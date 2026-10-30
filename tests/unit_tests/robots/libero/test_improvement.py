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

"""Offline evidence for independent, budgeted LIBERO improvement trials."""

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from robots.libero import improvement
from robots.libero import tools as libero_tools
from rpent.improvement import stream
from rpent.improvement.models import (
    ExecutionConfig,
    TaskCase,
    TrialBudget,
    TrialSpec,
    initial_state_key,
)
from rpent.memory.snapshots import SnapshotStore
from rpent.planner.base import PlannerResult
from rpent.tools.common import CommonTools


class _Env:
    def __init__(self, *, success=False, truncated=False, end_at=None):
        self.terminated = False
        self.truncated = False
        self.return_all_frames = False
        self.success = success
        self.end_at = end_at
        self.end_truncated = truncated
        self.steps = 0
        self.resets = 0

    def get_initial_state_id(self):
        return "libero_goal_task:t0:i0:sha256:layout"

    def reset(self):
        self.resets += 1
        self.terminated = self.truncated = False
        return {"states": np.zeros(8)}, {}

    def step(self, action):
        del action
        self.steps += 1
        self.terminated, self.truncated = self.success, self.end_truncated
        return {}, 0, self.terminated, self.truncated, {}

    def chunk_step(self, actions, **kwargs):
        del kwargs
        self.steps += len(actions)
        term = np.zeros(len(actions), dtype=bool)
        trunc = np.zeros(len(actions), dtype=bool)
        if self.success:
            term[self.end_at if self.end_at is not None else -1 :] = True
        self.terminated = bool(term.any())
        return {}, np.zeros(len(actions)), term, trunc, {}


@pytest.fixture
def trial(tmp_path, monkeypatch):
    source = tmp_path / "source"
    source.mkdir()
    (source / "MEMORY.md").write_text("# Fixed memory\n", encoding="utf-8")
    snapshot = SnapshotStore(tmp_path / "snapshots").capture(source)
    budget = TrialBudget(max_env_steps=4, max_vla_calls=2)
    spec = TrialSpec(
        trial_id="trial-1",
        experiment_id="offline",
        case=TaskCase(case_id="case-1", suite="libero_goal_task", task_id=0, seed=0),
        snapshot_id=snapshot.snapshot_id,
        snapshot_hash=snapshot.content_hash,
        scaffold_hash="fixed-scaffold",
        budget=budget,
        output_dir=tmp_path / "trial",
    )
    execution = ExecutionConfig(
        model="openai:offline", vla_identity="offline-vla", budget=budget
    )
    context = SimpleNamespace(
        spec=spec,
        snapshot=snapshot,
        execution=execution,
        env=_Env(),
        model=SimpleNamespace(predict=Mock(return_value=np.zeros((2, 7)))),
        owned=[SimpleNamespace(stop=Mock()), SimpleNamespace(stop=Mock())],
        toolkit=None,
        result=PlannerResult(stats={"turns_used": 2}),
        run=lambda toolkit: None,
        toolkit_error=None,
        close_error=None,
    )

    def init_runtime(args, output_dir, sink, components):
        assert not args.env_reset_on_connect
        assert args.memory_profile == "local"
        return context.owned, {"env": context.env, "model": context.model}

    real_spec = improvement.get_robot_spec()
    monkeypatch.setattr(
        improvement,
        "get_robot_spec",
        lambda: replace(real_spec, init_runtime=init_runtime),
    )

    def make_toolkit(**kwargs):
        assert kwargs["mode"] == "evaluation"
        assert kwargs["memory"].root == snapshot.corpus_dir
        assert kwargs["state_output_dir"] == context.spec.output_dir.resolve()
        env, model = kwargs["runtime_kwargs"]["env"], kwargs["runtime_kwargs"]["model"]
        env.reset()
        if context.toolkit_error:
            raise context.toolkit_error

        def close():
            if context.close_error:
                raise context.close_error

        context.toolkit = SimpleNamespace(
            env=env, model=model, memory=kwargs["memory"], close=Mock(side_effect=close)
        )
        return context.toolkit

    def build_planner(*args, **kwargs):
        assert args == ("api",)
        assert kwargs["memory_dir"] == snapshot.corpus_dir
        assert 0 < kwargs["planner_timeout_s"] <= budget.timeout_s

        def solve(**call):
            assert call["max_turns"] == context.spec.budget.max_turns
            assert str(snapshot.corpus_dir) in call["system_prompt"]
            context.run(call["toolkit"])
            return context.result

        return SimpleNamespace(solve=solve)

    monkeypatch.setattr(improvement, "get_toolkit", make_toolkit)
    monkeypatch.setattr(improvement, "build_planner", build_planner)
    return context


@pytest.mark.parametrize(
    ("native", "reported"), [(True, "failure"), (False, "success")]
)
def test_official_success_comes_from_environment_and_all_owned_services_close(
    trial, native, reported
):
    trial.env.success = native
    trial.result = PlannerResult(
        finish_result={"status": reported}, stats={"turns_used": 2}
    )
    trial.run = lambda toolkit: toolkit.env.step(np.zeros(7))
    result = improvement.LiberoTrialRunner(trial.execution).run(
        trial.spec, trial.snapshot
    )
    assert result.official_success is native
    assert result.success_source == "libero.env.terminated"
    assert result.outcome == "completed"
    assert result.cleanup_ok
    assert result.cost.env_steps == 1
    assert result.cost.resets == trial.env.resets == 1
    assert result.cost.vla_calls == 0
    assert result.cost.planner_requests == 2
    trial.toolkit.close.assert_called_once()
    for daemon in trial.owned:
        daemon.stop.assert_called_once()
    assert trial.spec.output_dir.joinpath("trial.json").is_file()


@pytest.mark.parametrize("failure", ["timed out after 1s", "connection refused"])
def test_planner_errors_preserve_native_outcome_and_cleanup(trial, failure):
    trial.result = PlannerResult(error=failure)
    result = improvement.LiberoTrialRunner(trial.execution).run(
        trial.spec, trial.snapshot
    )
    assert result.outcome == (
        "budget_exceeded" if "timed out" in failure else "infra_error"
    )
    assert result.error == failure
    assert result.official_success is False
    assert result.cleanup_ok
    trial.toolkit.close.assert_called_once()
    for daemon in trial.owned:
        daemon.stop.assert_called_once()


def test_toolkit_startup_failure_still_cleans_owned_daemons(trial):
    trial.toolkit_error = RuntimeError("state capture unavailable")
    result = improvement.LiberoTrialRunner(trial.execution).run(
        trial.spec, trial.snapshot
    )
    assert result.outcome == "infra_error"
    assert "state capture unavailable" in result.error
    for daemon in trial.owned:
        daemon.stop.assert_called_once()


def test_cleanup_failure_does_not_skip_other_owned_processes(trial):
    trial.close_error = RuntimeError("recording cleanup failed")
    trial.owned[-1].stop.side_effect = RuntimeError("stop failed")
    result = improvement.LiberoTrialRunner(trial.execution).run(
        trial.spec, trial.snapshot
    )
    assert result.outcome == "infra_error"
    assert not result.cleanup_ok
    assert "recording cleanup failed" in result.error and "stop failed" in result.error
    for daemon in trial.owned:
        daemon.stop.assert_called_once()


def test_large_chunk_is_refused_before_environment_entry_but_prediction_is_charged(
    trial,
):
    trial.model.predict.return_value = np.zeros((5, 7))

    def run(toolkit):
        toolkit.env.chunk_step(toolkit.model.predict({}))

    trial.run = run
    result = improvement.LiberoTrialRunner(trial.execution).run(
        trial.spec, trial.snapshot
    )
    assert result.outcome == "budget_exceeded"
    assert result.cost.env_steps == trial.env.steps == 0
    assert result.cost.vla_calls == 1
    assert result.cleanup_ok


def test_vla_budget_is_shared_across_primitive_calls(trial):
    def run(toolkit):
        for _ in range(3):
            toolkit.model.predict({})

    trial.run = run
    result = improvement.LiberoTrialRunner(trial.execution).run(
        trial.spec, trial.snapshot
    )
    assert result.outcome == "budget_exceeded"
    assert trial.model.predict.call_count == result.cost.vla_calls == 2
    assert result.cost.env_steps == 0


def test_failed_vla_call_is_charged_and_infrastructure_failure_is_retained(trial):
    trial.model.predict.side_effect = ConnectionError("prediction failed")
    trial.run = lambda toolkit: toolkit.model.predict({})
    result = improvement.LiberoTrialRunner(trial.execution).run(
        trial.spec, trial.snapshot
    )
    assert result.outcome == "infra_error"
    assert result.cost.vla_calls == 1
    assert result.cleanup_ok


def test_chunk_termination_does_not_invent_an_executed_prefix(trial):
    trial.env.success, trial.env.end_at = True, 0
    trial.run = lambda toolkit: toolkit.env.chunk_step(np.zeros((4, 7)))
    result = improvement.LiberoTrialRunner(trial.execution).run(
        trial.spec, trial.snapshot
    )
    assert result.official_success is True
    assert result.cost.env_steps is None
    assert "env_steps" in result.cost.unknown
    assert (
        trial.spec.output_dir.joinpath("budget.json")
        .read_text()
        .find('"environment_steps_admitted": 4')
        >= 0
    )


def test_truncation_is_not_task_success(trial):
    trial.env.end_truncated = True
    trial.run = lambda toolkit: toolkit.env.step(np.zeros(7))
    result = improvement.LiberoTrialRunner(trial.execution).run(
        trial.spec, trial.snapshot
    )
    assert result.truncated and not result.terminated
    assert result.official_success is False


def test_request_limit_without_finish_is_budget_exceeded(trial):
    trial.result = PlannerResult(stats={"turns_used": trial.spec.budget.max_turns})
    result = improvement.LiberoTrialRunner(trial.execution).run(
        trial.spec, trial.snapshot
    )
    assert result.outcome == "budget_exceeded"
    assert result.cost.planner_requests == trial.spec.budget.max_turns


def test_native_flags_are_read_after_toolkit_drains(trial):
    def run(toolkit):
        toolkit.close.side_effect = lambda: setattr(trial.env, "terminated", True)

    trial.run = run
    result = improvement.LiberoTrialRunner(trial.execution).run(
        trial.spec, trial.snapshot
    )
    assert result.official_success is True


def test_borrowed_services_are_never_stopped(trial):
    trial.owned = []
    trial.env.shutdown = Mock()
    trial.model.shutdown = Mock()
    result = improvement.LiberoTrialRunner(trial.execution).run(
        trial.spec, trial.snapshot
    )
    assert result.cleanup_ok
    trial.env.shutdown.assert_not_called()
    trial.model.shutdown.assert_not_called()


def test_codex_trial_dispatch_and_remote_planner_bridge_share_native_outcome(trial):
    execution = trial.execution.model_copy(
        update={"planner": "codex", "model": "gpt-5.5"}
    )
    captured = {}

    def factory(planner, **kwargs):
        captured.update(planner=planner, **kwargs)
        return SimpleNamespace(
            solve=lambda **call: PlannerResult(finish_result={"status": "success"})
        )

    runner = improvement.LiberoTrialRunner(execution, planner_factory=factory)
    result = runner.run(trial.spec, trial.snapshot)
    assert captured["planner"] == "codex"
    assert captured["model"] == "gpt-5.5"
    assert captured["tools_only"] is True
    assert result.outcome == "completed"
    assert result.official_success is False
    assert result.cleanup_ok


def test_transcript_omits_inline_images_without_changing_planner_messages(trial):
    message = {
        "role": "user",
        "content": [
            {"type": "input_image", "image_url": "data:image/png;base64,AAAA"},
            "data:image/png;base64,BBBB",
        ],
    }
    trial.result = PlannerResult(messages=[message])
    result = improvement.LiberoTrialRunner(trial.execution).run(
        trial.spec, trial.snapshot
    )
    transcript = json.loads(
        trial.spec.output_dir.joinpath("transcript.json").read_text()
    )
    assert "AAAA" not in str(transcript) and "BBBB" not in str(transcript)
    assert "AAAA" in str(message) and "BBBB" in str(message)
    assert result.outcome == "completed"


def test_only_snapshot_and_trial_files_are_visible_and_reads_are_audited(
    trial, tmp_path
):
    hidden = tmp_path / "meta.md"
    hidden.write_text("hidden improvement evidence", encoding="utf-8")

    def run(toolkit):
        tools = CommonTools(memory=toolkit.memory)
        tools.read_text_file(str(trial.snapshot.corpus_dir / "MEMORY.md"), max_chars=5)
        with pytest.raises(PermissionError):
            tools.read_text_file(str(hidden))
        with pytest.raises(PermissionError):
            tools.write_text_file(str(hidden), "overwrite")

    trial.run = run
    result = improvement.LiberoTrialRunner(trial.execution).run(
        trial.spec, trial.snapshot
    )
    assert len(result.memory_reads) == 1
    assert result.memory_reads[0]["truncated"]
    assert result.memory_reads[0]["trial_id"] == trial.spec.trial_id
    assert hidden.read_text() == "hidden improvement evidence"


def test_reserved_runtime_options_cannot_change_case_or_memory(trial):
    execution = trial.execution.model_copy(update={"robot_options": {"seed": 99}})
    result = improvement.LiberoTrialRunner(execution).run(trial.spec, trial.snapshot)
    assert result.outcome == "infra_error"
    assert "unsupported trial robot options" in result.error
    assert trial.toolkit is None
    for daemon in trial.owned:
        daemon.stop.assert_not_called()


def test_fixed_prompt_guides_remain_readable_and_audited(trial):
    def run(toolkit):
        tools = CommonTools(memory=toolkit.memory)
        result = tools.read_text_file("robots/libero/guides/env_calibration.md")
        assert result.data["content"]

    trial.run = run
    result = improvement.LiberoTrialRunner(trial.execution).run(
        trial.spec, trial.snapshot
    )
    assert result.outcome == "completed"
    assert len(result.memory_reads) == 1
    assert result.memory_reads[0]["relative_path"] is None


def test_reusing_a_trial_output_is_rejected_before_starting_runtime(trial):
    runner = improvement.LiberoTrialRunner(trial.execution)
    runner.run(trial.spec, trial.snapshot)
    with pytest.raises(ValueError, match="must be fresh"):
        runner.run(trial.spec, trial.snapshot)
    assert trial.env.resets == 1


def test_deployment_selection_metadata_is_permitted_before_startup(trial):
    trial.spec.output_dir.mkdir()
    (trial.spec.output_dir / "selection.json").write_text("{}", encoding="utf-8")
    (trial.spec.output_dir / "trial_spec.json").write_text(
        trial.spec.model_dump_json(), encoding="utf-8"
    )
    result = improvement.LiberoTrialRunner(trial.execution).run(
        trial.spec, trial.snapshot
    )
    assert result.outcome == "completed"
    assert trial.env.resets == 1


def test_native_success_exports_consumable_audit_and_native_recipe(trial):
    trial.env.success = True
    trial.env.get_task_language = Mock(return_value="Move the bowl to the plate")
    commands = [{"action": "move_to", "xyz": [0.1, 0.0, 0.2]}]
    recorded = [
        SimpleNamespace(
            step_idx=0, command=None, result=None, artifacts={}, terminated=False
        ),
        SimpleNamespace(
            step_idx=1,
            command=commands[0],
            result={"executed_steps": 1},
            artifacts={},
            terminated=True,
        ),
    ]

    def run(toolkit):
        def write_recipe(tag):
            # Export after close drains the scheduler, while the native trace
            # remains available and the owned environment is still alive.
            toolkit.close.assert_called_once()
            for daemon in trial.owned:
                daemon.stop.assert_not_called()
            return libero_tools.write_recipe_from_states(
                SimpleNamespace(records=lambda: recorded),
                tag,
                output_dir=trial.spec.output_dir,
            )

        toolkit.write_recipe = Mock(side_effect=write_recipe)
        toolkit.env.step(np.zeros(7))

    trial.run = run
    result = improvement.LiberoTrialRunner(
        trial.execution, export_task_records=True
    ).run(trial.spec, trial.snapshot)
    assert result.outcome == "completed" and result.cleanup_ok
    assert result.official_success is True
    tag = "goal_task_t0_s0"
    audit = json.loads(Path(result.artifacts["task_audit"]).read_text("utf-8"))
    assert audit["libero_terminated"] is True
    assert audit["task_language"] == "Move the bowl to the plate"
    assert audit["suite"] == trial.spec.case.suite
    assert audit["task_id"] == trial.spec.case.task_id
    assert audit["seed"] == trial.spec.case.seed
    assert audit["source_trial"] == trial.spec.trial_id
    assert audit["initial_state_id"] == result.initial_state_id
    assert audit["snapshot_hash"] == trial.snapshot.content_hash
    recipe = Path(result.artifacts["task_recipe"])
    assert recipe.name == f"{tag}_recipe.jsonl"
    assert [json.loads(line) for line in recipe.read_text().splitlines()] == commands
    assert trial.env.resets == 1

    # Exercise the actual stream importer and snapshot publisher; native reset
    # is represented by step-zero observation and never becomes a recipe action.
    records = stream._task_records(result)
    assert set(records) == {
        f"task-specific/{tag}.json",
        f"task-specific/{tag}_recipe.jsonl",
    }
    store = SnapshotStore(trial.snapshot.corpus_dir.parent.parent)
    published = store.publish_task_records(trial.snapshot, records)
    for path, content in records.items():
        assert (published.corpus_dir / path).read_bytes() == content


def test_native_failure_exports_no_success_records_even_when_planner_reports_success(
    trial,
):
    trial.result = PlannerResult(finish_result={"status": "success"})
    exporter = Mock(side_effect=AssertionError("failed episode must not export"))

    def run(toolkit):
        toolkit.write_recipe = exporter
        toolkit.env.step(np.zeros(7))

    trial.run = run
    result = improvement.LiberoTrialRunner(
        trial.execution, export_task_records=True
    ).run(trial.spec, trial.snapshot)
    assert result.official_success is False
    assert result.outcome == "completed" and result.cleanup_ok
    assert "task_audit" not in result.artifacts
    assert "task_recipe" not in result.artifacts
    assert stream._task_records(result) == {}
    exporter.assert_not_called()


def test_native_record_export_failure_still_closes_owned_services(trial):
    trial.env.success = True

    def run(toolkit):
        toolkit.write_recipe = Mock(side_effect=OSError("recipe storage failed"))
        toolkit.env.step(np.zeros(7))

    trial.run = run
    result = improvement.LiberoTrialRunner(
        trial.execution, export_task_records=True
    ).run(trial.spec, trial.snapshot)
    assert result.official_success is True
    assert result.outcome == "infra_error" and not result.cleanup_ok
    assert "task record export failed: recipe storage failed" in result.error
    assert "task_audit" not in result.artifacts
    assert "task_recipe" not in result.artifacts
    for daemon in trial.owned:
        daemon.stop.assert_called_once()


@pytest.mark.parametrize("catalog_index", [0, 999])
def test_forbidden_native_state_stops_before_reset_or_planner_and_closes_services(
    trial, catalog_index
):
    forbidden = initial_state_key(
        trial.env.get_initial_state_id().replace(":i0:", f":i{catalog_index}:")
    )
    result = improvement.LiberoTrialRunner(
        trial.execution,
        export_task_records=True,
        forbidden_initial_states={forbidden},
    ).run(trial.spec, trial.snapshot)
    assert result.outcome == "infra_error" and result.cleanup_ok
    assert "initial state overlaps construction" in result.error
    assert result.initial_state_id == trial.env.get_initial_state_id()
    assert result.official_success is None
    assert trial.toolkit is None
    assert trial.env.resets == result.cost.resets == 0
    assert trial.env.steps == 0
    assert result.cost.vla_calls == 0
    assert result.cost.planner_requests is None
    trial.model.predict.assert_not_called()
    for daemon in trial.owned:
        daemon.stop.assert_called_once()
