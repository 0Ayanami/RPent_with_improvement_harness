# Copyright 2026 The RPent Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Loopback-only LIBERO trial worker; expose it to a local planner over SSH."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, ConfigDict, Field

from robots.libero.improvement import LiberoTrialRunner
from robots.libero.remote import (
    PROTOCOL_VERSION,
    checked_relative,
    materialize_snapshot,
    source_identity,
)
from rpent.improvement.models import ExecutionConfig, TrialResult, TrialSpec
from rpent.planner.base import PlannerResult
from rpent.utils.config import get_repo_root
from rpent.utils.logging import get_logger

logger = get_logger("libero.remote_worker")


class _Start(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    session_id: str = Field(pattern=r"^[a-f0-9]{32}$")
    source_hash: str
    spec: dict[str, Any]
    execution: dict[str, Any]
    snapshot: dict[str, Any]
    export_task_records: bool = False
    forbidden_initial_states: list[str] = Field(default_factory=list)


class _Call(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    call_id: int = Field(gt=0)
    name: str
    arguments: dict[str, Any]


class _Finish(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    finish_result: dict[str, Any] | None = None
    messages: list[dict[str, Any]] = Field(default_factory=list)
    stats: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None


def _artifacts(output: Path, *, final: bool) -> list[dict[str, Any]]:
    manifest = []
    for path in sorted(output.rglob("*")):
        if path.is_symlink() or not path.resolve().is_relative_to(output):
            raise ValueError("worker artifact escapes the episode output directory")
        if not path.is_file() or (not final and path.suffix in {".mp4", ".avi"}):
            continue
        relative = path.relative_to(output).as_posix()
        checked_relative(relative)
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        manifest.append(
            {
                "path": relative,
                "size": path.stat().st_size,
                "sha256": digest.hexdigest(),
            }
        )
    return manifest


class _Session:
    def __init__(self, request: _Start, directory: Path, lease_timeout_s: float):
        self.session_id = request.session_id
        self.directory = directory
        self.output = directory / "trial"
        self.snapshot = materialize_snapshot(directory / "snapshots", request.snapshot)
        self.spec = TrialSpec.model_validate_json(
            json.dumps({**request.spec, "output_dir": str(self.output)})
        )
        self.execution = ExecutionConfig.model_validate_json(
            json.dumps(request.execution)
        )
        if self.execution.planner != "codex":
            raise ValueError("the remote worker accepts Codex trials only")
        self.lease_timeout_s = lease_timeout_s
        self.last_seen = time.monotonic()
        self.ready = threading.Event()
        self.done = threading.Event()
        self.finished = threading.Event()
        self.cancelled = False
        self.toolkit = None
        self.info: dict[str, Any] | None = None
        self.result: TrialResult | None = None
        self.error: BaseException | None = None
        self.planner_result = PlannerResult()
        self.call_lock = threading.Lock()
        self.last_call = 0
        self.last_fingerprint = ""
        self.last_response: dict[str, Any] | None = None
        self.recipe_tag = ""
        self.planner_timeout_s = float(self.spec.budget.timeout_s)
        self.runner = LiberoTrialRunner(
            self.execution,
            export_task_records=request.export_task_records,
            forbidden_initial_states=set(request.forbidden_initial_states),
            planner_factory=self._planner_factory,
        )
        self.thread = threading.Thread(
            target=self._run, name=f"libero-{self.session_id}", daemon=True
        )

    def _planner_factory(self, *args: Any, **kwargs: Any) -> _Session:
        self.recipe_tag = kwargs["recipe_tag"]
        self.planner_timeout_s = kwargs["planner_timeout_s"]
        return self

    def solve(
        self, *, system_prompt: str, user_message: str, toolkit: Any, **kwargs: Any
    ) -> PlannerResult:
        self.toolkit = toolkit
        self.last_seen = time.monotonic()
        self.info = {
            **self.paths(),
            "session_id": self.session_id,
            "lease_timeout_s": self.lease_timeout_s,
            "recipe_tag": self.recipe_tag,
            "system_prompt": system_prompt,
            "user_message": user_message,
            "tools": [
                {
                    "name": tool.name,
                    "description": tool.description,
                    "input_schema": tool.input_schema,
                    "readonly": tool.readonly,
                }
                for tool in toolkit.list_tools()
            ],
        }
        self.ready.set()
        if not self.done.wait(self.planner_timeout_s):
            self.planner_result = PlannerResult(error="remote planner timed out")
            self.done.set()
        if self.cancelled:
            raise InterruptedError(
                self.planner_result.error or "remote trial cancelled"
            )
        return self.planner_result

    def paths(self) -> dict[str, str]:
        return {
            "output_dir": str(self.output.resolve()),
            "memory_dir": str(self.snapshot.corpus_dir),
            "repo_root": str(get_repo_root()),
        }

    def _run(self) -> None:
        try:
            self.result = self.runner.run(self.spec, self.snapshot)
        except BaseException as exc:
            self.error = exc
            logger.exception("remote trial %s failed", self.session_id)
        finally:
            self.toolkit = None
            self.last_response = None
            # The transcript is persisted by the native runner. Keeping every
            # planner conversation in this long-lived worker would accumulate
            # image/tool payloads across the full construction/evaluation run.
            self.planner_result = PlannerResult()
            self.finished.set()
            self.ready.set()

    def stop(self, *, lease_expired: bool = False) -> None:
        if self.finished.is_set() or self.done.is_set():
            return
        self.cancelled = not lease_expired
        self.planner_result = PlannerResult(
            error="remote client lease timed out"
            if lease_expired
            else "remote trial cancelled"
        )
        self.done.set()

    def response(self) -> dict[str, Any]:
        if self.error is not None:
            raise HTTPException(
                500, f"remote trial failed: {type(self.error).__name__}: {self.error}"
            )
        if self.result is None:
            raise HTTPException(409, "remote trial is still draining its tools")
        return {
            **self.paths(),
            "result": self.result.model_dump(mode="json"),
            "artifacts": _artifacts(self.output, final=True),
        }


class _Worker:
    def __init__(self, work_dir: Path, lease_timeout_s: float):
        self.work_dir = work_dir.resolve()
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.lease_timeout_s = lease_timeout_s
        self.source_hash = source_identity()
        self.sessions: dict[str, _Session] = {}
        self.lock = threading.Lock()
        self.shutdown = threading.Event()
        self.watchdog = threading.Thread(
            target=self._watch, name="libero-worker-lease", daemon=True
        )
        self.watchdog.start()

    def _watch(self) -> None:
        while not self.shutdown.wait(1):
            with self.lock:
                active = [
                    session
                    for session in self.sessions.values()
                    if not session.finished.is_set()
                ]
            for session in active:
                if (
                    session.ready.is_set()
                    and time.monotonic() - session.last_seen > self.lease_timeout_s
                ):
                    session.stop(lease_expired=True)

    def get(self, session_id: str) -> _Session:
        with self.lock:
            session = self.sessions.get(session_id)
        if session is None:
            raise HTTPException(404, "unknown remote trial session")
        session.last_seen = time.monotonic()
        return session

    def close(self) -> None:
        self.shutdown.set()
        for session in list(self.sessions.values()):
            session.stop()
        for session in list(self.sessions.values()):
            if session.thread.ident is not None:
                session.thread.join()
        self.watchdog.join(timeout=2)


def create_app(work_dir: Path, *, lease_timeout_s: float = 120) -> FastAPI:
    """Create one serial worker; CPU tests may replace its native trial runner."""
    if lease_timeout_s < 1:
        raise ValueError("lease_timeout_s must be at least one second")
    worker = _Worker(work_dir, lease_timeout_s)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        yield
        worker.close()

    app = FastAPI(lifespan=lifespan)
    app.state.worker = worker

    @app.get("/health")
    def health():
        with worker.lock:
            active = next(
                (
                    session.session_id
                    for session in worker.sessions.values()
                    if not session.finished.is_set()
                ),
                None,
            )
        return {
            "protocol": PROTOCOL_VERSION,
            "status": "ready",
            "source_hash": worker.source_hash,
            "active_trial": active,
        }

    @app.post("/trials/start")
    def start(request: _Start):
        if request.source_hash != worker.source_hash:
            raise HTTPException(
                409, "local and remote RPent source identities do not match"
            )
        with worker.lock:
            if request.session_id in worker.sessions:
                raise HTTPException(409, "session id was already used")
            if any(
                not session.finished.is_set() for session in worker.sessions.values()
            ):
                raise HTTPException(409, "the worker already has an active trial")
            directory = worker.work_dir / request.session_id
            try:
                directory.mkdir(exist_ok=False)
                session = _Session(request, directory, lease_timeout_s)
            except (ValueError, KeyError, TypeError, OSError) as exc:
                raise HTTPException(400, f"invalid remote trial: {exc}") from exc
            worker.sessions[request.session_id] = session
            session.thread.start()
        session.ready.wait()
        if session.finished.is_set():
            return session.response()
        return {**session.info, "artifacts": _artifacts(session.output, final=False)}

    @app.post("/trials/{session_id}/heartbeat")
    def heartbeat(session_id: str):
        worker.get(session_id)
        return {"ok": True}

    @app.post("/trials/{session_id}/tool")
    def tool(session_id: str, call: _Call):
        session = worker.get(session_id)
        fingerprint = json.dumps(call.model_dump(), sort_keys=True)
        with session.call_lock:
            if call.call_id == session.last_call:
                if (
                    fingerprint != session.last_fingerprint
                    or session.last_response is None
                ):
                    raise HTTPException(
                        409, "call id was already used with a different request"
                    )
                return session.last_response
            if call.call_id != session.last_call + 1:
                raise HTTPException(
                    409, "tool call ids must be contiguous and must not be replayed"
                )
            if session.done.is_set() or session.toolkit is None:
                raise HTTPException(409, "the trial no longer accepts tool calls")
            # Retire the id before dispatch. An exception during execution or
            # artifact capture must never make an action replayable.
            session.last_call = call.call_id
            session.last_fingerprint = fingerprint
            session.last_response = None
            toolkit = session.toolkit
            result = toolkit.execute_tool(call.name, call.arguments)
            response = {
                "data": result.data,
                "images": [
                    base64.b64encode(image).decode("ascii") for image in result.images
                ],
                "solved": toolkit.solved(),
                "artifacts": _artifacts(session.output, final=False),
            }
            session.last_call, session.last_fingerprint, session.last_response = (
                call.call_id,
                fingerprint,
                response,
            )
            return response

    @app.post("/trials/{session_id}/pause")
    def pause(session_id: str):
        session = worker.get(session_id)
        if session.toolkit is not None:
            session.toolkit.cancel_active_and_wait()
        return {"ok": True}

    @app.post("/trials/{session_id}/resume")
    def resume(session_id: str):
        session = worker.get(session_id)
        if session.toolkit is not None and not session.done.is_set():
            session.toolkit.resume_calls()
        return {"ok": True}

    @app.post("/trials/{session_id}/finish")
    def finish(session_id: str, result: _Finish):
        session = worker.get(session_id)
        with session.call_lock:
            if not session.done.is_set():
                session.planner_result = PlannerResult(**result.model_dump())
                session.done.set()
        session.finished.wait()
        return session.response()

    @app.post("/trials/{session_id}/cancel")
    def cancel(session_id: str):
        session = worker.get(session_id)
        session.stop()
        session.finished.wait()
        return session.response()

    @app.get("/trials/{session_id}/artifacts/{relative:path}")
    def artifact(session_id: str, relative: str):
        session = worker.get(session_id)
        try:
            path = session.output / checked_relative(relative)
            if (
                path.is_symlink()
                or not path.resolve().is_relative_to(session.output.resolve())
                or not path.is_file()
            ):
                raise ValueError("artifact is not a regular file within this episode")
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        return FileResponse(path)

    return app


def main() -> None:
    """Serve the worker on loopback; an SSH tunnel owns external access."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", choices=("127.0.0.1", "::1"), default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8870)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--lease-timeout-s", type=float, default=120)
    args = parser.parse_args()
    import uvicorn

    uvicorn.run(
        create_app(args.work_dir, lease_timeout_s=args.lease_timeout_s),
        host=args.host,
        port=args.port,
    )


if __name__ == "__main__":
    main()
