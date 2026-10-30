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

"""Baseline integration requires recorded actions rather than a success claim."""

import json

import pytest

from robots.libero import baseline_smoke
from rpent.evaluation.result import write_json_atomic
from rpent.improvement.models import TrialCost, TrialResult


@pytest.fixture
def runner(monkeypatch):
    class Runner:
        calls = []
        finish = True
        action = True

        def __init__(self, execution):
            self.execution = execution

        def run(self, spec, snapshot):
            self.calls.append((self.execution, spec, snapshot))
            output = spec.output_dir
            output.mkdir()
            write_json_atomic(
                output / "transcript.json",
                {
                    "finish": {"_finish": True, "status": "stuck"}
                    if self.finish
                    else None,
                },
            )
            write_json_atomic(
                output / "states.json",
                {
                    "steps": [
                        {
                            "command": {"action": "pi0_pick"},
                            "result": {"chunks_used": 1 if self.action else 0},
                        }
                    ]
                },
            )
            write_json_atomic(
                output / "budget.json",
                {
                    "environment_steps_admitted": 5 if self.action else 0,
                },
            )
            (output / "agentview.png").write_bytes(b"offline-image")
            return TrialResult(
                spec=spec,
                outcome="completed",
                official_success=False,
                initial_state_id="libero_object:t0:i0:sha256:layout",
                success_source="libero.env.terminated",
                cleanup_ok=True,
                cost=TrialCost(vla_calls=1 if self.action else 0),
            )

    monkeypatch.setattr(baseline_smoke, "RemoteLiberoTrialRunner", Runner)
    return Runner


def test_ordinary_episode_and_failed_task_can_pass_integration(tmp_path, runner):
    memory = tmp_path / "original-memory"
    memory.mkdir()
    (memory / "MEMORY.md").write_text("# Original frozen corpus\n", encoding="utf-8")
    output = tmp_path / "run"
    assert (
        baseline_smoke.main(
            [
                "--memory-dir",
                str(memory),
                "--memory-provenance",
                "original fixed commit",
                "--vla-identity",
                "fixed-checkpoint",
                "--output-dir",
                str(output),
            ]
        )
        == 0
    )
    report = json.loads((output / "baseline-smoke.json").read_text("utf-8"))
    assert report["status"] == "passed"
    assert report["native_success"] is False
    assert report["finish_status"] == "stuck"
    assert report["admitted_env_steps"] == 5
    config = json.loads((output / "baseline-config.json").read_text("utf-8"))
    assert config["adaptive_memory_enabled"] is False
    assert config["memory_provenance"] == "original fixed commit"
    assert (memory / "MEMORY.md").read_text("utf-8") == "# Original frozen corpus\n"
    assert len(runner.calls) == 1
    execution, spec, _ = runner.calls[0]
    assert execution.planner == "codex"
    assert execution.model == "gpt-5.5"
    assert spec.case.suite == "libero_object"


@pytest.mark.parametrize("missing", ["action", "finish"])
def test_claimed_completion_without_chain_evidence_fails(tmp_path, runner, missing):
    setattr(runner, missing, False)
    output = tmp_path / "run"
    assert (
        baseline_smoke.main(
            [
                "--empty-memory",
                "--vla-identity",
                "fixed-checkpoint",
                "--output-dir",
                str(output),
            ]
        )
        == 2
    )
    report = json.loads((output / "baseline-smoke.json").read_text("utf-8"))
    assert report["status"] == "failed"
    assert not report["checks"][
        "vla_action_recorded" if missing == "action" else "finish"
    ]


def test_existing_output_is_preserved(tmp_path, runner):
    output = tmp_path / "existing"
    output.mkdir()
    keep = output / "keep.txt"
    keep.write_text("preserve", encoding="utf-8")
    assert (
        baseline_smoke.main(
            [
                "--empty-memory",
                "--vla-identity",
                "fixed-checkpoint",
                "--output-dir",
                str(output),
            ]
        )
        == 2
    )
    assert keep.read_text("utf-8") == "preserve"
    assert not runner.calls
