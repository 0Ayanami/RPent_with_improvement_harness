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

"""Local Codex planning against a LIBERO toolkit reached through an SSH tunnel."""

from __future__ import annotations

import base64
import hashlib
import json
import re
import threading
import time
import uuid
from pathlib import Path, PurePosixPath
from types import SimpleNamespace
from typing import Any

import httpx

from rpent.dashboard.events import NullDashboardEventSink
from rpent.evaluation.result import write_json_atomic
from rpent.improvement.models import ExecutionConfig, TrialCost, TrialResult, TrialSpec
from rpent.memory import MemoryManager
from rpent.memory.snapshots import MemorySnapshot, SnapshotStore, _checked_path
from rpent.planner.base import build_planner
from rpent.tools.base import ToolResult
from rpent.tools.toolkit import Toolkit
from rpent.utils.config import get_repo_root
from rpent.utils.logging import init_output_dir
from rpent.utils.serialization import strip_inline_images

PROTOCOL_VERSION = 1
_PORTABLE_PART = re.compile(r"[A-Za-z0-9_. -]+\Z")
_RESERVED = {"CON", "PRN", "AUX", "NUL"} | {
    f"{prefix}{number}" for prefix in ("COM", "LPT") for number in range(1, 10)
}


def source_identity(root: Path | None = None) -> str:
    """Hash shared source and LIBERO prompts, tolerating checkout line endings."""
    root = root or get_repo_root()
    files = [root / "pyproject.toml"]
    for directory in (root / "rpent", root / "robots" / "libero"):
        files.extend(
            path
            for path in directory.rglob("*")
            if path.is_file()
            and path.suffix in {".py", ".md", ".yaml", ".yml"}
            and "__pycache__" not in path.parts
            and "config" not in path.relative_to(directory).parts
        )
    digest = hashlib.sha256()
    for path in sorted(files, key=lambda item: item.relative_to(root).as_posix()):
        digest.update(path.relative_to(root).as_posix().encode("utf-8") + b"\0")
        digest.update(path.read_bytes().replace(b"\r\n", b"\n") + b"\0")
    return "sha256:" + digest.hexdigest()


def checked_relative(value: str) -> Path:
    """Accept only regular portable relative paths before any file is created."""
    parts = value.split("/")
    if (
        not value
        or len(parts) > 32
        or any(
            not _PORTABLE_PART.fullmatch(part)
            or part in {".", ".."}
            or part.endswith((".", " "))
            or part.split(".", 1)[0].upper() in _RESERVED
            for part in parts
        )
    ):
        raise ValueError(f"invalid portable artifact path: {value!r}")
    return Path(*PurePosixPath(value).parts)


def snapshot_payload(snapshot: MemorySnapshot) -> dict[str, Any]:
    """Transfer the original verified manifest and exact published file bytes."""
    SnapshotStore(snapshot.corpus_dir.parent.parent).verify(snapshot)
    return {
        "manifest": base64.b64encode(
            (snapshot.corpus_dir.parent / "manifest.json").read_bytes()
        ).decode("ascii"),
        "files": {
            path: base64.b64encode(
                (snapshot.corpus_dir / checked_relative(path)).read_bytes()
            ).decode("ascii")
            for path in snapshot.files
        },
    }


def materialize_snapshot(root: Path, payload: dict[str, Any]) -> MemorySnapshot:
    """Check transfer paths and revalidate the complete SnapshotStore manifest."""
    manifest_bytes = base64.b64decode(payload["manifest"], validate=True)
    manifest = json.loads(manifest_bytes)
    identity = manifest.get("snapshot_id", "")
    if not isinstance(identity, str) or not re.fullmatch(
        r"snapshot-[0-9a-f]{64}", identity
    ):
        raise ValueError("invalid transferred snapshot identity")
    files = payload["files"]
    if not isinstance(files, dict) or len(files) > 10000:
        raise ValueError("snapshot transfer requires at most 10000 files")
    if set(files) != set(manifest.get("files", {})):
        raise ValueError("snapshot transfer file list differs from its manifest")
    checked = {name: checked_relative(name) for name in files}
    if len({name.casefold() for name in checked}) != len(checked):
        raise ValueError("snapshot transfer has case-aliased paths")
    contents = {
        name: base64.b64decode(content, validate=True)
        for name, content in files.items()
    }
    if sum(map(len, contents.values())) > 64 * 1024 * 1024:
        raise ValueError("memory snapshot transfer exceeds 64 MiB")
    directory = root / identity
    directory.mkdir(parents=True, exist_ok=False)
    corpus = directory / "corpus"
    corpus.mkdir()
    for name, content in contents.items():
        target = corpus / checked[name]
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
    (directory / "manifest.json").write_bytes(manifest_bytes)
    return SnapshotStore(root).load(identity)


def replace_paths(value: Any, roots: list[tuple[str, str]]) -> Any:
    """Translate artifact and memory roots across the Windows/Linux boundary."""
    if isinstance(value, str):
        for source, destination in roots:
            value = value.replace(source, destination)
        return value
    if isinstance(value, list):
        return [replace_paths(item, roots) for item in value]
    if isinstance(value, dict):
        return {key: replace_paths(item, roots) for key, item in value.items()}
    return value


def remote_tool_arguments(
    arguments: dict[str, Any], roots: list[tuple[str, str]]
) -> dict[str, Any]:
    """Map the common file tools' path without rewriting text or recipe content."""
    result = dict(arguments)
    path = result.get("path")
    if not isinstance(path, str):
        return result
    normalized = path.replace("\\", "/")
    for remote, local in roots:
        prefix = local.replace("\\", "/").rstrip("/")
        windows = len(prefix) > 1 and prefix[1] == ":"
        compared = normalized.casefold() if windows else normalized
        expected = prefix.casefold() if windows else prefix
        if compared == expected or compared.startswith(expected + "/"):
            result["path"] = (
                remote.replace("\\", "/").rstrip("/") + normalized[len(prefix) :]
            )
            return result
    result["path"] = normalized
    return result


def _request(client: httpx.Client, method: str, path: str, **kwargs: Any) -> Any:
    response = client.request(method, path, **kwargs)
    response.raise_for_status()
    return response.json()


class RemoteToolkit(Toolkit):
    """Expose the worker's native tool schemas through the existing local MCP."""

    def __init__(
        self,
        client: httpx.Client,
        session_id: str,
        info: dict[str, Any],
        output: Path,
        snapshot: MemorySnapshot,
    ) -> None:
        self.client = client
        self.session_id = session_id
        self.output = output
        self._memory = MemoryManager(snapshot.corpus_dir, memory_access="read_only")
        self.roots = [
            (info["output_dir"], str(output)),
            (info["memory_dir"], str(snapshot.corpus_dir)),
            (info["repo_root"], str(get_repo_root())),
        ]
        self._declarations = tuple(
            SimpleNamespace(**replace_paths(declaration, self.roots))
            for declaration in info["tools"]
        )
        self._call_lock = threading.Lock()
        self._call_id = 0
        self._downloaded: dict[str, str] = {}
        self._solved = False

    def list_tools(self) -> tuple[Any, ...]:
        """Return the declarations validated and executed by the server toolkit."""
        return self._declarations

    def execute_tool(self, name: str, input_dict: dict[str, Any]) -> ToolResult:
        """Issue one serial high-level command; never retry an uncertain action."""
        with self._call_lock:
            self._call_id += 1
            response = _request(
                self.client,
                "POST",
                f"/trials/{self.session_id}/tool",
                json={
                    "call_id": self._call_id,
                    "name": name,
                    "arguments": remote_tool_arguments(input_dict, self.roots),
                },
            )
            self.sync_artifacts(response["artifacts"])
            self._solved = response["solved"]
            return ToolResult(
                data=replace_paths(response["data"], self.roots),
                images=[
                    base64.b64decode(item, validate=True) for item in response["images"]
                ],
            )

    def sync_artifacts(self, manifest: list[dict[str, Any]]) -> None:
        """Download changed artifacts with size/hash checks, without loading videos."""
        for item in manifest:
            relative = checked_relative(item["path"])
            if self._downloaded.get(item["path"]) == item["sha256"]:
                continue
            target = self.output / relative
            # A planner may read local artifacts. It must not create symlinks that
            # redirect a subsequent worker download outside this episode.
            _checked_path(target.parent)
            if target.is_symlink() or not target.resolve().is_relative_to(self.output):
                raise ValueError("artifact download escapes the local episode")
            target.parent.mkdir(parents=True, exist_ok=True)
            digest = hashlib.sha256()
            size = 0
            with self.client.stream(
                "GET", f"/trials/{self.session_id}/artifacts/{item['path']}"
            ) as response:
                response.raise_for_status()
                with target.open("wb") as stream:
                    for chunk in response.iter_bytes():
                        size += len(chunk)
                        if size > item["size"]:
                            raise ValueError(
                                "remote artifact exceeds its advertised size"
                            )
                        digest.update(chunk)
                        stream.write(chunk)
            if size != item["size"] or digest.hexdigest() != item["sha256"]:
                raise ValueError(
                    "remote artifact transfer failed integrity verification"
                )
            self._downloaded[item["path"]] = item["sha256"]

    def cancel_active_and_wait(self) -> None:
        """Drain server commands before the local planner exits."""
        _request(self.client, "POST", f"/trials/{self.session_id}/pause")

    def resume_calls(self) -> None:
        """Reopen worker admission after a drained pause."""
        _request(self.client, "POST", f"/trials/{self.session_id}/resume")

    def solved(self) -> bool:
        """Return the most recent native success reported by the worker."""
        return self._solved

    def close(self) -> None:
        """The owning trial finalizes and releases the remote toolkit."""


class RemoteLiberoTrialRunner:
    """Run local Codex sessions while all LIBERO primitives stay on the server."""

    def __init__(
        self,
        execution: ExecutionConfig,
        *,
        export_task_records: bool = False,
        forbidden_initial_states: set[str] | None = None,
    ) -> None:
        if execution.remote is None or execution.planner != "codex":
            raise ValueError(
                "remote LIBERO trials require a Codex planner and remote endpoint"
            )
        self.execution = execution
        self.export_task_records = export_task_records
        self.forbidden_initial_states = forbidden_initial_states

    def run(self, spec: TrialSpec, snapshot: MemorySnapshot) -> TrialResult:
        """Transfer one snapshot, run a fresh local planner and collect native evidence."""
        started = time.perf_counter()
        output = spec.output_dir.resolve()
        if output.is_relative_to(
            snapshot.corpus_dir.resolve()
        ) or snapshot.corpus_dir.resolve().is_relative_to(output):
            raise ValueError("trial output and memory corpus must have separate roots")
        if output.exists() and any(
            path.name not in {"trial_spec.json", "trial_events.jsonl", "selection.json"}
            for path in output.iterdir()
        ):
            raise ValueError(f"trial output must be fresh: {output}")
        output.mkdir(parents=True, exist_ok=True)
        init_output_dir(output)
        remote = self.execution.remote
        if remote is None:
            raise ValueError("remote endpoint is required")
        session_id = uuid.uuid4().hex
        stop_heartbeat = threading.Event()
        heartbeat = None
        toolkit = None
        planner_result = None
        worker_result = None
        failure = None
        interrupted = False
        with httpx.Client(
            base_url=remote.endpoint,
            timeout=remote.request_timeout_s,
            trust_env=False,
        ) as client:
            try:
                health = _request(client, "GET", "/health")
                if (
                    health.get("protocol") != PROTOCOL_VERSION
                    or health.get("source_hash") != source_identity()
                ):
                    raise ValueError(
                        "local and remote RPent source identities do not match"
                    )
                info = _request(
                    client,
                    "POST",
                    "/trials/start",
                    json={
                        "session_id": session_id,
                        "source_hash": source_identity(),
                        "spec": spec.model_dump(mode="json"),
                        "execution": self.execution.model_dump(
                            mode="json", exclude={"remote"}
                        ),
                        "snapshot": snapshot_payload(snapshot),
                        "export_task_records": self.export_task_records,
                        "forbidden_initial_states": sorted(
                            self.forbidden_initial_states or ()
                        ),
                    },
                )
                if "result" not in info:
                    toolkit = RemoteToolkit(client, session_id, info, output, snapshot)
                    toolkit.sync_artifacts(info["artifacts"])
                    heartbeat_interval = max(1, info["lease_timeout_s"] / 3)

                    def keepalive() -> None:
                        while not stop_heartbeat.wait(heartbeat_interval):
                            try:
                                _request(
                                    client,
                                    "POST",
                                    f"/trials/{session_id}/heartbeat",
                                    timeout=10,
                                )
                            except Exception:
                                # The worker lease bounds the remote episode if
                                # connectivity is lost during local reasoning.
                                return

                    heartbeat = threading.Thread(
                        target=keepalive, name="libero-lease", daemon=True
                    )
                    heartbeat.start()
                    planner = build_planner(
                        self.execution.planner,
                        output_dir=output,
                        recipe_tag=info["recipe_tag"],
                        robot_name="libero",
                        base_url=self.execution.base_url,
                        model=self.execution.model,
                        memory_dir=snapshot.corpus_dir,
                        max_tokens=self.execution.max_tokens,
                        reasoning_effort=self.execution.reasoning_effort,
                        planner_timeout_s=max(
                            0.001,
                            spec.budget.timeout_s - (time.perf_counter() - started),
                        ),
                        dashboard_events=NullDashboardEventSink(),
                        tools_only=True,
                    )
                    planner_result = planner.solve(
                        system_prompt=replace_paths(
                            info["system_prompt"], toolkit.roots
                        ),
                        user_message=replace_paths(info["user_message"], toolkit.roots),
                        toolkit=toolkit,
                        max_turns=spec.budget.max_turns,
                    )
                    info = _request(
                        client,
                        "POST",
                        f"/trials/{session_id}/finish",
                        json={
                            "finish_result": planner_result.finish_result,
                            "messages": strip_inline_images(planner_result.messages),
                            "stats": planner_result.stats,
                            "error": planner_result.error,
                        },
                    )
                worker_result = info["result"]
                if toolkit is None:
                    # Startup can fail before a toolkit exists; its diagnostic
                    # files still belong to this remote session.
                    toolkit = RemoteToolkit(
                        client, session_id, {**info, "tools": []}, output, snapshot
                    )
                toolkit.sync_artifacts(info["artifacts"])
                worker_result = replace_paths(worker_result, toolkit.roots)
                worker_result["spec"] = spec.model_dump(mode="json")
            except BaseException as exc:
                failure = f"{type(exc).__name__}: {exc}"
                interrupted = isinstance(exc, (KeyboardInterrupt, InterruptedError))
                try:
                    info = _request(
                        client,
                        "POST",
                        f"/trials/{session_id}/cancel",
                        timeout=min(30, remote.request_timeout_s),
                    )
                    if toolkit is not None:
                        toolkit.sync_artifacts(info["artifacts"])
                        worker_result = replace_paths(info["result"], toolkit.roots)
                        worker_result["spec"] = spec.model_dump(mode="json")
                except Exception:
                    pass
            finally:
                stop_heartbeat.set()
                if heartbeat is not None:
                    heartbeat.join(timeout=11)
        if worker_result is not None:
            result = TrialResult.model_validate_json(json.dumps(worker_result))
            if failure:
                result = result.model_copy(
                    update={
                        "outcome": "cancelled" if interrupted else "infra_error",
                        "error": failure,
                    }
                )
        else:
            result = TrialResult(
                spec=spec,
                outcome="cancelled" if interrupted else "infra_error",
                cleanup_ok=False,
                error=failure or "remote trial did not return a native result",
                cost=TrialCost(wall_time_s=time.perf_counter() - started),
                artifacts={"result": str(output / "trial.json")},
            )
        result = result.model_copy(
            update={
                "cost": result.cost.model_copy(
                    update={"wall_time_s": time.perf_counter() - started}
                )
            }
        )
        try:
            SnapshotStore(snapshot.corpus_dir.parent.parent).verify(snapshot)
        except Exception as exc:
            result = result.model_copy(
                update={
                    "outcome": "infra_error",
                    "error": f"snapshot integrity check failed: {exc}",
                }
            )
        write_json_atomic(output / "trial.json", result.model_dump(mode="json"))
        return result
