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

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from rpent.improvement.controller import ImprovementHarness, check, load_config, prepare
from rpent.improvement.deployment import select_snapshot
from rpent.improvement.models import (
    ExecutionConfig,
    ExperiencePatch,
    ExperimentConfig,
    ImprovementProtocol,
    TaskCase,
    TrialCost,
    TrialResult,
)
from rpent.improvement.proposer import ProposalResult
from rpent.memory.snapshots import MemoryEdit


@pytest.fixture
def config(tmp_path: Path) -> ExperimentConfig:
    root = tmp_path / "source"
    (root / "global").mkdir(parents=True)
    metadata = {
        "id": "grasp",
        "scope": "global",
        "kind": "failure",
        "title": "Grasp recovery",
        "applies_when": "Before transport",
        "confidence": "single-shot",
        "evidence": {"cells": ["previous"]},
    }
    (root / "global" / "grasp.md").write_text(
        "---\n" + yaml.safe_dump(metadata) + "---\nInspect before moving.\n",
        encoding="utf-8",
    )
    (root / "MEMORY.md").write_text(
        "# Memory\n[Grasp](global/grasp.md)\n", encoding="utf-8"
    )
    cases = [
        TaskCase(case_id=f"case_{i}", suite="libero_spatial_task", task_id=0, seed=i)
        for i in range(2)
    ]
    return ExperimentConfig(
        experiment_id="offline_contract",
        output_dir=tmp_path / "experiment",
        baseline_memory=root,
        execution=ExecutionConfig(model="test:model", vla_identity="fake-only"),
        protocol=ImprovementProtocol(
            protocol_id="single_patch_v1",
            probe_cases=cases[:1],
            confirmation_cases=cases[1:],
        ),
        proposer_model="test:proposer",
        library_dir=tmp_path / "library",
    )


def make_patch(baseline, *, stale=False) -> ExperiencePatch:
    return ExperiencePatch(
        patch_id="inspect_grasp",
        base_snapshot_hash=baseline.content_hash,
        edit=MemoryEdit(
            operation="revise",
            path="global/grasp.md",
            expected_old_hash="sha256:" + "0" * 64
            if stale
            else baseline.files["global/grasp.md"],
            body="Verify the object is held before transport.\n",
        ),
        hypotheses=["Transport may start after an empty grasp"],
        expected_effect="Prevent transport after empty grasp",
        falsifying_observation="Extra checks block a valid grasp",
        evidence_refs=["reference_case_0"],
    )


class Proposer:
    def __init__(self, *, stale=False):
        self.calls = []
        self.stale = stale

    def propose(self, *, baseline, evidence, lessons, rejection):
        self.calls.append({"lessons": lessons, "rejection": rejection})
        return ProposalResult(
            patch=make_patch(baseline, stale=self.stale),
            requests=1,
            input_tokens=10,
            output_tokens=20,
            elapsed_s=0.01,
        )


class Runner:
    def __init__(
        self,
        *,
        reads=True,
        alias=False,
        cleanup=True,
        regression=False,
        unknown=False,
        crash=False,
    ):
        self.calls = []
        (
            self.reads,
            self.alias,
            self.cleanup,
            self.regression,
            self.unknown,
            self.crash,
        ) = reads, alias, cleanup, regression, unknown, crash

    def run(self, spec, snapshot):
        self.calls.append(spec)
        if self.crash:
            raise OSError("simulated process interruption")
        candidate = snapshot.parent_id is not None
        reads = (
            [
                {
                    "relative_path": "global/grasp.md",
                    "source_hash": snapshot.files["global/grasp.md"],
                    "provided_chars": 10,
                    "truncated": False,
                }
            ]
            if self.reads
            else []
        )
        success = (not candidate) if self.regression else candidate
        return TrialResult(
            spec=spec,
            outcome="completed",
            official_success=None if self.unknown else success,
            success_source="offline_fake",
            initial_state_id="layout_0"
            if self.alias
            else f"{spec.case.suite}:{spec.case.task_id}:{spec.case.seed}",
            cleanup_ok=self.cleanup,
            cost=TrialCost(wall_time_s=0.1, env_steps=2, vla_calls=1),
            memory_reads=reads,
        )


def test_full_adaptation_is_independent_and_preserves_e0(config):
    original = (config.baseline_memory / "global/grasp.md").read_bytes()
    proposer, runner = Proposer(), Runner()
    harness = ImprovementHarness(config, runner, proposer)
    report = harness.run()
    assert report.decision == "accepted"
    assert [spec.trial_id for spec in runner.calls] == [
        "reference_case_0",
        "probe_case_0",
        "confirmation_base_case_1",
        "confirmation_case_1",
    ]
    assert report.total_cost["env_steps"] == 8
    assert report.total_cost["input_tokens"] is None
    assert report.total_cost["all_model_input_tokens"] is None
    assert report.total_cost["proposals"][0]["requests"] == 1
    assert (config.baseline_memory / "global/grasp.md").read_bytes() == original
    assert (config.output_dir / "qualification.json").is_file()
    assert len(list((config.library_dir / "entries").glob("*.json"))) == 1
    assert proposer.calls[0]["lessons"] == []
    for spec in runner.calls:
        assert (spec.output_dir / "trial_result.json").is_file()
        assert (
            json.loads(
                (spec.output_dir / "trial_events.jsonl").read_text().splitlines()[-1]
            )["run_state"]
            == "finalized"
        )


def test_cost_report_preserves_known_tokens_when_one_trial_has_no_usage(config):
    from rpent.improvement.stream import _aggregate_costs

    class PartialUsageRunner(Runner):
        def run(self, spec, snapshot):
            result = super().run(spec, snapshot)
            return result.model_copy(
                update={
                    "cost": result.cost.model_copy(
                        update={"input_tokens": None if len(self.calls) == 1 else 20}
                    )
                }
            )

    report = ImprovementHarness(config, PartialUsageRunner(), Proposer()).run()
    costs = report.total_cost
    assert costs["all_model_input_tokens"] is None
    assert costs["known_subtotals"]["all_model_input_tokens"] == 70
    assert costs["known_subtotals"]["all_model_output_tokens"] == 20
    combined = _aggregate_costs([costs, costs])
    assert combined["all_model_input_tokens"] is None
    assert combined["known_subtotals"]["all_model_input_tokens"] == 140


@pytest.mark.parametrize(
    "options,expected,count",
    [
        ({"reads": False}, "insufficient_evidence", 2),
        ({"unknown": True}, "insufficient_evidence", 1),
        ({"alias": True}, "insufficient_evidence", 4),
        ({"regression": True}, "rejected", 2),
        ({"cleanup": False}, "insufficient_evidence", 1),
    ],
)
def test_gate_retains_fallback_for_incomplete_or_negative_evidence(
    config, options, expected, count
):
    runner = Runner(**options)
    report = ImprovementHarness(config, runner, Proposer()).run()
    assert report.decision == expected
    assert len(runner.calls) == count


def test_trial_budget_reserved_before_any_model_or_robot_call(config):
    config = config.model_copy(
        update={"protocol": config.protocol.model_copy(update={"max_trials": 3})}
    )
    runner, proposer = Runner(), Proposer()
    report = ImprovementHarness(config, runner, proposer).run()
    assert report.decision == "insufficient_evidence"
    assert not runner.calls and not proposer.calls


def test_invalid_proposal_retries_are_finite_and_costed(config):
    proposer, runner = Proposer(stale=True), Runner()
    report = ImprovementHarness(config, runner, proposer).run()
    assert report.decision == "invalid"
    assert len(proposer.calls) == 2
    assert proposer.calls[1]["rejection"]
    assert len(runner.calls) == 1
    assert sum(reply["requests"] for reply in report.total_cost["proposals"]) == 2


def test_interrupted_trial_is_not_finalized_or_reused(config):
    harness = ImprovementHarness(config, Runner(crash=True), Proposer())
    with pytest.raises(OSError, match="interruption"):
        harness.run()
    trial = config.output_dir / "trials/reference_case_0"
    assert not (trial / "trial_result.json").exists()
    last = json.loads((trial / "trial_events.jsonl").read_text().splitlines()[-1])
    assert last == {"run_state": "interrupted", "cost": "unknown"}
    with pytest.raises(ValueError, match="already started"):
        ImprovementHarness(config, Runner(), Proposer()).run()


def test_deployment_revalidates_scope_budget_and_snapshot(config):
    harness = ImprovementHarness(config, Runner(), Proposer())
    report = harness.run()
    case = TaskCase(
        case_id="deployment", suite="libero_spatial_task", task_id=0, seed=10
    )
    selection, snapshot = select_snapshot(config, report, case, harness.store)
    assert selection.used_candidate
    assert snapshot.content_hash == report.candidate_hash
    rejected = report.model_copy(update={"decision": "rejected"})
    selection, _ = select_snapshot(config, rejected, case, harness.store)
    assert not selection.used_candidate
    changed = config.model_copy(
        update={"execution": config.execution.model_copy(update={"max_tokens": 42})}
    )
    with pytest.raises(ValueError, match="scaffold"):
        select_snapshot(changed, report, case, harness.store)
    with pytest.raises(ValueError, match="scope"):
        select_snapshot(
            config, report, case.model_copy(update={"task_id": 1}), harness.store
        )
    (snapshot.corpus_dir / "global/grasp.md").write_text("tampered", encoding="utf-8")
    with pytest.raises(ValueError):
        select_snapshot(config, report, case, harness.store)


def test_static_check_and_relative_config_need_no_robot_runtime(config, tmp_path):
    result = check(config)
    assert result["behavior_verified"] is False
    _, baseline = prepare(config)
    patch_file = tmp_path / "patch.json"
    patch_file.write_text(make_patch(baseline).model_dump_json(), encoding="utf-8")
    manual = config.model_copy(
        update={"patch_file": patch_file, "proposer_model": None}
    )
    assert check(manual)["candidate_hash"] != baseline.content_hash
    data = manual.model_dump(mode="json")
    data.update(
        output_dir="experiment",
        baseline_memory="source",
        patch_file="patch.json",
        library_dir="library",
    )
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    assert load_config(path).baseline_memory == config.baseline_memory


def test_truncated_candidate_read_cannot_qualify(config):
    class TruncatedRunner(Runner):
        def run(self, spec, snapshot):
            result = super().run(spec, snapshot)
            return result.model_copy(
                update={
                    "memory_reads": [
                        {**event, "truncated": True} for event in result.memory_reads
                    ]
                }
            )

    report = ImprovementHarness(config, TruncatedRunner(), Proposer()).run()
    assert report.decision == "insufficient_evidence"


def test_scaffold_drift_prevents_further_trials_or_adoption(config, monkeypatch):
    class DriftingRunner(Runner):
        def run(self, spec, snapshot):
            result = super().run(spec, snapshot)
            monkeypatch.setenv("LIBERO_TYPE", "changed-during-trial")
            return result

    runner = DriftingRunner()
    report = ImprovementHarness(config, runner, Proposer()).run()
    assert report.decision == "invalid"
    assert len(runner.calls) == 1
    assert any("scaffold changed" in reason for reason in report.reasons)


def test_snapshot_mutation_retains_measured_cost_and_blocks_proposal(config):
    class MutatingRunner(Runner):
        def run(self, spec, snapshot):
            result = super().run(spec, snapshot)
            (snapshot.corpus_dir / "MEMORY.md").write_text("changed", encoding="utf-8")
            return result

    proposer = Proposer()
    report = ImprovementHarness(config, MutatingRunner(), proposer).run()
    assert report.decision == "insufficient_evidence"
    assert report.total_cost["env_steps"] == 2
    assert not proposer.calls
    result = json.loads(
        (config.output_dir / "trials/reference_case_0/trial_result.json").read_text()
    )
    assert result["outcome"] == "infra_error"


def test_cli_check_run_and_independent_deployment(config, tmp_path, monkeypatch):
    from robots.libero import improvement as runtime
    from rpent.cli.improvement import main
    from rpent.improvement import proposer

    monkeypatch.setattr(runtime, "LiberoTrialRunner", lambda execution: Runner())
    monkeypatch.setattr(proposer, "APIProposer", lambda *args: Proposer())
    path = tmp_path / "experiment.yaml"
    path.write_text(yaml.safe_dump(config.model_dump(mode="json")), encoding="utf-8")
    assert main(["check", "--config", str(path)]) == 0
    assert main(["run", "--config", str(path)]) == 0
    deploy = tmp_path / "deployment"
    assert (
        main(
            [
                "deploy",
                "--config",
                str(path),
                "--report",
                str(config.output_dir / "qualification.json"),
                "--suite",
                "libero_spatial_task",
                "--task",
                "0",
                "--seed",
                "2",
                "--output-dir",
                str(deploy),
            ]
        )
        == 0
    )
    result = json.loads((deploy / "trial_result.json").read_text())
    assert result["spec"]["case"]["seed"] == 2
    assert result["spec"]["trial_id"] == "deployment"
    assert len(list((config.library_dir / "entries").glob("*.json"))) == 1
