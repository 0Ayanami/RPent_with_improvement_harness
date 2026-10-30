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

"""Offline lifecycle and transfer evidence for local Codex/remote LIBERO trials."""

import base64
import json
import time
from contextlib import nullcontext
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from robots.libero import improvement, remote, remote_worker
from rpent.improvement.models import (
    ExecutionConfig,
    RemoteExecutionConfig,
    TaskCase,
    TrialBudget,
    TrialSpec,
    initial_state_key,
)
from rpent.memory.snapshots import SnapshotStore
from rpent.planner.base import PlannerResult
from rpent.tools.base import ToolResult
from rpent.tools.common import CommonTools


@pytest.fixture
def trial(tmp_path, monkeypatch):
    source = tmp_path / "source"
    source.mkdir()
    (source / "MEMORY.md").write_text("# Fixed corpus\n", encoding="utf-8")
    snapshot = SnapshotStore(tmp_path / "snapshots").capture(source)
    budget = TrialBudget(timeout_s=30, max_env_steps=2)
    spec = TrialSpec(
        trial_id="remote-1",
        experiment_id="offline",
        case=TaskCase(case_id="case-1", suite="libero_goal_task", task_id=0, seed=0),
        snapshot_id=snapshot.snapshot_id,
        snapshot_hash=snapshot.content_hash,
        scaffold_hash="offline",
        budget=budget,
        output_dir=tmp_path / "local",
    )
    execution = ExecutionConfig(
        planner="codex",
        model="offline",
        vla_identity="offline",
        budget=budget,
        remote=RemoteExecutionConfig(endpoint="http://127.0.0.1:8870"),
    )
    env = SimpleNamespace(
        terminated=False,
        truncated=False,
        resets=0,
        get_initial_state_id=lambda: "libero_goal_task:t0:i0:sha256:layout",
        get_task_language=lambda: "move the object",
    )

    def reset():
        env.resets += 1
        return {}, {}

    def step(action):
        env.terminated = True
        return {}, 0, True, False, {}

    env.reset, env.step = reset, step
    context = SimpleNamespace(
        snapshot=snapshot,
        spec=spec,
        execution=execution,
        env=env,
        daemons=[SimpleNamespace(stop=Mock()), SimpleNamespace(stop=Mock())],
        toolkit=None,
        calls=0,
        close=Mock(),
    )

    def init_runtime(args, output_dir, sink, components):
        assert not args.env_reset_on_connect
        return context.daemons, {"env": env, "model": SimpleNamespace(predict=Mock())}

    robot = improvement.get_robot_spec()
    monkeypatch.setattr(
        improvement, "get_robot_spec", lambda: replace(robot, init_runtime=init_runtime)
    )
    monkeypatch.setattr(improvement, "init_output_dir", lambda output: output)

    def make_toolkit(**kwargs):
        wrapped_env = kwargs["runtime_kwargs"]["env"]
        wrapped_env.reset()
        output = kwargs["state_output_dir"]
        memory = kwargs["memory"]

        def execute(name, arguments):
            context.calls += 1
            CommonTools(memory=memory).read_text_file(str(memory.root / "MEMORY.md"))
            wrapped_env.step([0] * 7)
            artifact = output / "observation.png"
            artifact.write_bytes(b"offline-png")
            return ToolResult(data={"artifact": str(artifact)}, images=[b"offline-png"])

        def recipe(tag):
            name = f"{tag}_recipe.jsonl"
            (output / name).write_text('{"action":"offline"}\n', encoding="utf-8")
            return name

        context.toolkit = SimpleNamespace(
            list_tools=lambda: (
                SimpleNamespace(
                    name="move",
                    description="Move once.",
                    input_schema={"type": "object", "properties": {}},
                    readonly=False,
                ),
            ),
            execute_tool=execute,
            solved=lambda: env.terminated,
            close=context.close,
            cancel_active_and_wait=Mock(),
            resume_calls=Mock(),
            write_recipe=recipe,
        )
        return context.toolkit

    monkeypatch.setattr(improvement, "get_toolkit", make_toolkit)
    monkeypatch.setattr(remote, "source_identity", lambda: "sha256:offline")
    monkeypatch.setattr(remote_worker, "source_identity", lambda: "sha256:offline")
    app = remote_worker.create_app(tmp_path / "worker", lease_timeout_s=2)
    context.app = app
    context.start = {
        "session_id": "a" * 32,
        "source_hash": "sha256:offline",
        "spec": spec.model_dump(mode="json"),
        "execution": execution.model_dump(mode="json", exclude={"remote"}),
        "snapshot": remote.snapshot_payload(snapshot),
        "export_task_records": True,
        "forbidden_initial_states": [],
    }
    with TestClient(app) as client:
        context.client = client
        yield context


def test_native_outcome_audit_recipe_and_duplicate_action(trial):
    client = trial.client
    assert client.get("/health").json()["protocol"] == 1
    started = client.post("/trials/start", json=trial.start)
    assert started.status_code == 200, started.text
    assert "move" in {tool["name"] for tool in started.json()["tools"]}
    path = f"/trials/{trial.start['session_id']}"
    call = {"call_id": 1, "name": "move", "arguments": {}}
    first = client.post(path + "/tool", json=call)
    again = client.post(path + "/tool", json=call)
    assert first.status_code == again.status_code == 200
    assert first.json() == again.json()
    assert trial.calls == 1
    assert base64.b64decode(first.json()["images"][0]) == b"offline-png"
    changed = client.post(path + "/tool", json={**call, "name": "other"})
    assert changed.status_code == 409
    finished = client.post(
        path + "/finish",
        json={"finish_result": {"status": "failure"}, "stats": {"turns_used": 2}},
    )
    assert finished.status_code == 200, finished.text
    result = finished.json()["result"]
    assert result["official_success"] is True
    assert result["cost"]["env_steps"] == 1
    assert result["cost"]["resets"] == 1
    assert result["cleanup_ok"]
    assert result["memory_reads"]
    assert {"task_recipe", "task_audit"} <= result["artifacts"].keys()
    assert client.get(path + "/artifacts/observation.png").content == b"offline-png"
    trial.close.assert_called_once()
    for daemon in trial.daemons:
        daemon.stop.assert_called_once()
    assert client.get("/health").json()["active_trial"] is None


def test_single_active_session_and_cancel_cleanup(trial):
    assert trial.client.post("/trials/start", json=trial.start).status_code == 200
    second = trial.client.post(
        "/trials/start", json={**trial.start, "session_id": "b" * 32}
    )
    assert second.status_code == 409
    result = trial.client.post(f"/trials/{trial.start['session_id']}/cancel").json()[
        "result"
    ]
    assert result["outcome"] == "cancelled"
    assert result["cleanup_ok"]
    trial.close.assert_called_once()


def test_tool_failure_after_action_does_not_allow_replay(trial):
    trial.client.post("/trials/start", json=trial.start).raise_for_status()
    execute = trial.toolkit.execute_tool

    def fail_after_action(name, arguments):
        execute(name, arguments)
        raise HTTPException(500, "response failed after environment action")

    trial.toolkit.execute_tool = fail_after_action
    path = f"/trials/{trial.start['session_id']}/tool"
    call = {"call_id": 1, "name": "move", "arguments": {}}
    assert trial.client.post(path, json=call).status_code == 500
    assert trial.client.post(path, json=call).status_code == 409
    assert trial.calls == 1


def test_worker_shutdown_cancels_and_drains_the_native_trial(trial):
    trial.client.post("/trials/start", json=trial.start).raise_for_status()
    session = trial.app.state.worker.sessions[trial.start["session_id"]]
    trial.app.state.worker.close()
    assert session.finished.is_set()
    assert session.result.outcome == "cancelled"
    assert session.result.cleanup_ok
    trial.close.assert_called_once()
    for daemon in trial.daemons:
        daemon.stop.assert_called_once()


def test_forbidden_initial_state_is_rejected_before_reset(trial):
    request = {
        **trial.start,
        "forbidden_initial_states": [
            initial_state_key(trial.env.get_initial_state_id())
        ],
    }
    response = trial.client.post("/trials/start", json=request)
    assert response.status_code == 200, response.text
    result = response.json()["result"]
    assert result["outcome"] == "infra_error"
    assert "overlaps construction" in result["error"]
    assert trial.env.resets == trial.calls == 0
    for daemon in trial.daemons:
        daemon.stop.assert_called_once()


def test_mismatched_source_is_rejected_without_starting_runtime(trial):
    response = trial.client.post(
        "/trials/start", json={**trial.start, "source_hash": "different"}
    )
    assert response.status_code == 409
    assert trial.env.resets == 0


@pytest.mark.parametrize(
    "path",
    [
        "../escape.md",
        "/absolute.md",
        "global/../escape.md",
        "global/a\\b.md",
        "global/CON.md",
    ],
)
def test_transferred_snapshot_paths_are_rejected_before_writing(trial, path):
    payload = dict(trial.start["snapshot"])
    manifest = json.loads(base64.b64decode(payload["manifest"]))
    manifest["files"] = {path: "sha256:" + "0" * 64}
    payload["manifest"] = base64.b64encode(json.dumps(manifest).encode()).decode()
    payload["files"] = {path: base64.b64encode(b"invalid").decode()}
    response = trial.client.post(
        "/trials/start", json={**trial.start, "snapshot": payload}
    )
    assert response.status_code == 400
    assert trial.env.resets == 0


def test_lease_expiration_cleans_the_native_trial(trial):
    trial.client.post("/trials/start", json=trial.start).raise_for_status()
    session = trial.app.state.worker.sessions[trial.start["session_id"]]
    session.last_seen = time.monotonic() - 10
    assert session.finished.wait(3)
    assert session.result.outcome == "budget_exceeded"
    assert "lease timed out" in session.result.error
    assert session.result.cleanup_ok
    trial.close.assert_called_once()


def test_local_runner_downloads_artifacts_and_restores_local_paths(trial, monkeypatch):
    # Reuse the already-started app lifespan through a small context wrapper.
    class BorrowedClient:
        def __enter__(self):
            return trial.client

        def __exit__(self, *args):
            return False

    monkeypatch.setattr(remote.httpx, "Client", lambda **kwargs: BorrowedClient())

    def build_planner(*args, **kwargs):
        assert kwargs["tools_only"] is True

        def solve(**call):
            assert str(trial.snapshot.corpus_dir) in call["system_prompt"]
            result = call["toolkit"].execute_tool("move", {})
            assert result.images == [b"offline-png"]
            assert (
                Path(result.data["artifact"])
                == trial.spec.output_dir / "observation.png"
            )
            assert Path(result.data["artifact"]).read_bytes() == b"offline-png"
            return PlannerResult(
                finish_result={"status": "success"}, stats={"turns_used": 1}
            )

        return SimpleNamespace(solve=solve)

    monkeypatch.setattr(remote, "build_planner", build_planner)
    result = remote.RemoteLiberoTrialRunner(
        trial.execution, export_task_records=True
    ).run(trial.spec, trial.snapshot)
    assert result.outcome == "completed", result.error
    assert result.official_success is True
    assert result.spec == trial.spec
    assert result.memory_reads
    for path in result.artifacts.values():
        assert Path(path).is_relative_to(trial.spec.output_dir)
        assert Path(path).is_file()
    assert Path(result.artifacts["task_audit"]).is_file()


def test_snapshot_bytes_are_verified_after_transfer(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "MEMORY.md").write_text("# Original\n", encoding="utf-8")
    snapshot = SnapshotStore(tmp_path / "local").capture(source)
    payload = remote.snapshot_payload(snapshot)
    payload["files"]["MEMORY.md"] = base64.b64encode(b"changed").decode()
    with pytest.raises(ValueError, match="hash mismatch"):
        remote.materialize_snapshot(tmp_path / "server", payload)


def test_local_planner_cancellation_collects_remote_cleanup(trial, monkeypatch):
    monkeypatch.setattr(
        remote.httpx, "Client", lambda **kwargs: nullcontext(trial.client)
    )

    def interrupted(*args, **kwargs):
        raise InterruptedError("operator cancelled local planning")

    monkeypatch.setattr(remote, "build_planner", interrupted)
    result = remote.RemoteLiberoTrialRunner(trial.execution).run(
        trial.spec, trial.snapshot
    )
    assert result.outcome == "cancelled"
    assert result.cleanup_ok
    assert "operator cancelled" in result.error
    trial.close.assert_called_once()
    for daemon in trial.daemons:
        daemon.stop.assert_called_once()


@pytest.mark.parametrize("separator", ["\\", "/"])
def test_windows_file_paths_map_entire_suffix_without_changing_content(separator):
    local = "D:/research/RPent/snapshots/hash/corpus"
    path = (local + "/task-specific/object_t0_s0.json").replace("/", separator)
    content = r"Keep literal \\n and D:\research\RPent\reference in this text."
    mapped = remote.remote_tool_arguments(
        {"path": path, "content": content}, [("/server/snapshot/corpus", local)]
    )
    assert mapped["path"] == "/server/snapshot/corpus/task-specific/object_t0_s0.json"
    assert mapped["content"] == content
    assert remote.remote_tool_arguments(
        {"prompt": content}, [("/server/snapshot/corpus", local)]
    ) == {"prompt": content}


def test_translated_windows_path_passes_the_native_memory_boundary(tmp_path):
    from rpent.memory import MemoryManager

    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "MEMORY.md").write_text("# Original memory\n", encoding="utf-8")
    memory = MemoryManager(
        corpus, memory_access="read_only", allowed_read_roots=[corpus]
    )
    roots = [(str(corpus), r"D:\local\snapshot\corpus")]
    mapped = remote.remote_tool_arguments(
        {"path": r"D:\local\snapshot\corpus\MEMORY.md"}, roots
    )
    result = CommonTools(memory=memory).read_text_file(mapped["path"])
    assert result.data["content"] == "# Original memory\n"
    denied = remote.remote_tool_arguments(
        {"path": r"D:\local\snapshot\corpus-other\MEMORY.md"}, roots
    )
    with pytest.raises(PermissionError):
        CommonTools(memory=memory).read_text_file(denied["path"])


def test_source_identity_tolerates_checkout_newlines(tmp_path):
    (tmp_path / "rpent").mkdir()
    (tmp_path / "robots" / "libero").mkdir(parents=True)
    (tmp_path / "pyproject.toml").write_bytes(b"name='offline'\n")
    source = tmp_path / "rpent" / "test.py"
    source.write_bytes(b"value = 1\n")
    first = remote.source_identity(tmp_path)
    source.write_bytes(b"value = 1\r\n")
    assert remote.source_identity(tmp_path) == first
    source.write_bytes(b"value = 2\n")
    assert remote.source_identity(tmp_path) != first
