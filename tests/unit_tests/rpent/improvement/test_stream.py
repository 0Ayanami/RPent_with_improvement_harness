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

"""Offline construction streams and independently frozen paired evaluation."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from rpent.improvement import stream
from rpent.improvement.models import ExecutionConfig, TaskCase, TrialCost, TrialResult
from rpent.improvement.ppc import PPCController
from rpent.improvement.ppc_models import (
    MemoryRevision,
    PPCEditObservation,
    PPCEditResponse,
    PPCLesson,
    PPCReflection,
    PPCReflectionResponse,
)
from rpent.memory.snapshots import MemoryEdit, SnapshotStore


@pytest.fixture
def config(tmp_path):
    source = tmp_path / "initial"
    (source / "global").mkdir(parents=True)
    metadata = {
        "id": "rule",
        "scope": "global",
        "kind": "strategy",
        "title": "Observe the grasp",
        "applies_when": "before transport",
        "confidence": "single-shot",
        "evidence": {"cells": ["offline-fixture"]},
    }
    (source / "global/rule.md").write_text(
        "---\n" + yaml.safe_dump(metadata) + "---\nObserve before transport.\n",
        encoding="utf-8",
    )
    (source / "MEMORY.md").write_text("# Memory\n", encoding="utf-8")
    execution = ExecutionConfig(model="test:executor", vla_identity="offline-only")
    return stream.StreamConfig(
        experiment_id="stream_contract",
        output_dir=tmp_path / "experiment",
        initial_memory=stream.MemorySource(path=source, provenance="offline fixture"),
        reference_memory=stream.MemorySource(path=source, provenance="same fixture"),
        execution=execution,
        evaluation_execution=execution,
        editor=stream.EditorConfig(model="test:editor"),
    )


@pytest.fixture
def small_stream(config, monkeypatch):
    def cases(self, *, construction):
        return [
            TaskCase(
                case_id=f"libero_spatial_t{task}_s{seed}",
                suite="libero_spatial",
                task_id=task,
                seed=seed,
            )
            for task in (0, 1)
            for seed in ([0] if construction else [1, 2])
        ]

    monkeypatch.setattr(stream.StreamConfig, "cases", cases)
    # Concurrent source edits are outside this fixture's execution contract.
    monkeypatch.setattr(stream, "scaffold_fingerprint", lambda _: "offline-scaffold")
    return config


class Editor:
    def __init__(self):
        self.prefixes = []

    def edit(self, *, stage, baseline, evidence, lessons, rejection):
        task = evidence["task"]["task_id"]
        reference = evidence["trials"]["base"]["trial_id"]
        return PPCEditResponse(
            revision=MemoryRevision(
                revision_id=f"task{task}_{stage}",
                base_snapshot_hash=baseline.content_hash,
                edits=[
                    MemoryEdit(
                        operation="revise",
                        path="global/rule.md",
                        expected_old_hash=baseline.files["global/rule.md"],
                        body=f"Task {task}, stage {stage}: verify before transport.\n",
                    )
                ],
                notes="Offline memory candidate",
                hypotheses=["Verification may avoid an empty grasp"],
                expected_effect="Avoid empty transport",
                falsifying_observation="A valid grasp is blocked",
                evidence_refs=[reference],
            ),
            requests=1,
        )

    def reflect(self, *, baseline, evidence, lessons):
        self.prefixes.append(json.loads(json.dumps(list(lessons))))
        task = evidence["task"]["task_id"]
        return PPCReflectionResponse(
            reflection=PPCReflection(
                edits=[
                    PPCEditObservation(
                        path="global/rule.md",
                        candidates=["final"],
                        change="Observe before transport",
                        observed="Offline final execution returned native evidence",
                        verdict="confounded",
                        evidence_refs=[evidence["trials"]["final"]["trial_id"]],
                    )
                ],
                lesson=PPCLesson(
                    cue=f"task {task}",
                    lesson=f"Observed recovery in task {task}",
                    applies_when="before transport",
                    evidence_refs=[evidence["trials"]["final"]["trial_id"]],
                    uncertainty="Offline fake establishes scheduling only",
                ),
            ),
            requests=1,
        )


class Runner:
    def __init__(self, *, alias=None, identity_suffix="", changed_stage=None):
        self.calls = []
        self.alias = alias
        self.identity_suffix = identity_suffix
        self.changed_stage = changed_stage

    def run(self, spec, snapshot):
        self.calls.append(spec)
        case = spec.case
        stage = spec.output_dir.name
        index = case.seed
        content_seed = case.seed
        if case.seed and self.alias == "construction":
            content_seed = 0
            index = 99
        elif case.seed and self.alias == "evaluation":
            content_seed = 1
        suffix = self.identity_suffix
        if stage == self.changed_stage:
            suffix += "changed-stage"
        digest = hashlib.sha256(
            f"{case.task_id}:{content_seed}:{suffix}".encode()
        ).hexdigest()
        identity = f"{case.suite}:t{case.task_id}:i{index}:sha256:{digest}"
        success = stage != "final" or case.task_id == 0
        artifacts = {}
        if success:
            tag = f"{case.suite.replace('libero_', '')}_t{case.task_id}_s{case.seed}"
            audit = spec.output_dir / f"{tag}.json"
            recipe = spec.output_dir / f"{tag}_recipe.jsonl"
            audit.write_text(
                json.dumps(
                    {
                        "suite": case.suite,
                        "task_id": case.task_id,
                        "seed": case.seed,
                        "libero_terminated": True,
                        "source_trial": spec.trial_id,
                        "initial_state_id": identity,
                    }
                ),
                encoding="utf-8",
            )
            recipe.write_text('{"action":"move_to","xyz":[0,0,1]}\n', encoding="utf-8")
            artifacts = {"task_audit": str(audit), "task_recipe": str(recipe)}
        return TrialResult(
            spec=spec,
            outcome="completed",
            official_success=success,
            success_source="offline_fake",
            initial_state_id=identity,
            cleanup_ok=True,
            cost=TrialCost(wall_time_s=0.01, env_steps=1, vla_calls=0),
            artifacts=artifacts,
        )


def build(config):
    editor, runner = Editor(), Runner()
    controller = PPCController(
        runner, editor, SnapshotStore(config.output_dir / "snapshots")
    )
    report = stream.construct(config, controller)
    return report, editor, runner


def payload(path):
    return json.loads(path.read_text("utf-8"))["payload"]


@pytest.mark.parametrize(
    "benchmark,suites,total", [("libero", 4, 400), ("libero_pro", 8, 800)]
)
def test_full_paper_cases_preserve_seed_split(config, benchmark, suites, total):
    config = config.model_copy(update={"benchmark": benchmark})
    construction = config.cases(construction=True)
    evaluation = config.cases(construction=False)
    assert len(construction) == suites * 10
    assert len(evaluation) == total
    assert {case.seed for case in construction} == {0}
    assert {case.seed for case in evaluation} == set(range(1, 11))
    assert len({case.case_id for case in evaluation}) == total
    assert {case.suite for case in evaluation} == set(config.suites)


@pytest.mark.parametrize("seeds", [[0], [0, 1], [-1, 1], [1, 1]])
def test_invalid_evaluation_seed_split_is_rejected(config, seeds):
    fields = config.model_dump()
    fields["evaluation_seeds"] = seeds
    with pytest.raises(ValueError, match="evaluation seeds"):
        stream.StreamConfig.model_validate(fields)


def test_offline_plan_imports_no_gpu_simulator_or_editor(config, tmp_path):
    config_file = tmp_path / "experiment.yaml"
    config_file.write_text(
        yaml.safe_dump(config.model_dump(mode="json")), encoding="utf-8"
    )
    script = """
import importlib.abc
import socket
import sys

class BlockRuntime(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        blocked = ('torch', 'rlinf', 'robosuite', 'libero', 'transformers',
                   'rpent.improvement.ppc', 'rpent.improvement.ppc_proposer',
                   'rpent.improvement.proposer',
                   'rpent.robots.components.pi05_vla_server',
                   'rpent.robots.components.sam3_server')
        if any(fullname == name or fullname.startswith(name + '.') for name in blocked):
            raise AssertionError('offline plan imported ' + fullname)

sys.meta_path.insert(0, BlockRuntime())
def no_network(*args, **kwargs):
    raise AssertionError('offline plan opened a network connection')
socket.create_connection = no_network
from rpent.cli.improvement import main
assert main(['plan', '--config', sys.argv[1]]) == 0
"""
    completed = subprocess.run(
        [sys.executable, "-c", script, str(config_file)],
        cwd=Path(__file__).resolve().parents[4],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    planned = payload(config.output_dir / "plan.json")
    assert planned["expected_rollouts"] == 160
    assert len(planned["evaluation_cases"]) == 400
    assert planned["behavior_verified"] is False


def test_construction_freezes_prefix_and_only_selected_successful_task_records(
    small_stream,
):
    report, editor, runner = build(small_stream)
    assert report["status"] == "frozen"
    assert report["tasks"] == 2 and report["rollouts"] == 8
    assert [spec.output_dir.name for spec in runner.calls] == [
        "base",
        "inv",
        "dom",
        "final",
    ] * 2
    first, second = small_stream.cases(construction=True)
    root = small_stream.output_dir / "construction"
    prefix0 = payload(root / first.case_id / "meta_prefix.json")
    prefix1 = payload(root / second.case_id / "meta_prefix.json")
    assert prefix0["entries"] == [] and prefix0["selected"] == []
    assert [entry["task_id"] for entry in prefix1["entries"]] == [0]
    assert len(prefix1["selected"]) == 1
    assert editor.prefixes[0] == []
    assert [lesson["cue"] for lesson in editor.prefixes[1]] == ["task 0"]
    assert payload(root / first.case_id / "meta_prefix.json") == prefix0

    manifest, store, snapshot = stream.load_frozen(
        small_stream.output_dir / "frozen-ppc_meta.json"
    )
    assert manifest.snapshot_hash == report["snapshot_hash"]
    assert len(manifest.construction_states) == 2
    records = {name for name in snapshot.files if name.startswith("task-specific/")}
    assert records == {
        "task-specific/spatial_t0_s0.json",
        "task-specific/spatial_t0_s0_recipe.jsonl",
    }
    audit = json.loads(
        (snapshot.corpus_dir / "task-specific/spatial_t0_s0.json").read_text()
    )
    assert audit["source_trial"] == f"{first.case_id}_final"
    assert audit["libero_terminated"] is True
    assert "stage compose" in (snapshot.corpus_dir / "global/rule.md").read_text()
    assert not any("meta" in name or "reflection" in name for name in snapshot.files)
    store.verify(snapshot)


def test_meta_lessons_transfer_across_suites_without_other_experiment_history(
    small_stream, monkeypatch
):
    original_cases = stream.StreamConfig.cases

    def cross_suite_cases(self, *, construction):
        return [
            case.model_copy(
                update={
                    "suite": "libero_object",
                    "case_id": case.case_id.replace("libero_spatial", "libero_object"),
                }
            )
            if case.task_id == 1
            else case
            for case in original_cases(self, construction=construction)
        ]

    class OtherExperimentEditor(Editor):
        def reflect(self, **kwargs):
            response = super().reflect(**kwargs)
            lesson = response.reflection.lesson.model_copy(
                update={"cue": "other-experiment-only"}
            )
            return response.model_copy(
                update={
                    "reflection": response.reflection.model_copy(
                        update={"lesson": lesson}
                    )
                }
            )

    monkeypatch.setattr(stream.StreamConfig, "cases", cross_suite_cases)
    other = small_stream.model_copy(
        update={
            "experiment_id": "other_experiment",
            "output_dir": small_stream.output_dir.parent / "other-experiment",
        }
    )
    stream.construct(
        other,
        PPCController(
            Runner(),
            OtherExperimentEditor(),
            SnapshotStore(other.output_dir / "snapshots"),
        ),
    )
    assert list((other.output_dir / "construction/meta").glob("*.json"))

    _, editor, _ = build(small_stream)
    first, second = small_stream.cases(construction=True)
    assert first.suite == "libero_spatial" and second.suite == "libero_object"
    assert editor.prefixes[0] == []
    assert [lesson["cue"] for lesson in editor.prefixes[1]] == ["task 0"]
    selected = payload(
        small_stream.output_dir / "construction" / second.case_id / "meta_prefix.json"
    )["selected"]
    assert [lesson["cue"] for lesson in selected] == ["task 0"]
    assert "other-experiment-only" not in json.dumps(editor.prefixes)


def test_frozen_evaluation_survives_missing_construction_and_calls_no_editor(
    small_stream, monkeypatch
):
    _, editor, _ = build(small_stream)
    (small_stream.output_dir / "construction").rename(
        small_stream.output_dir / "archived-construction"
    )

    def forbidden(*args, **kwargs):
        raise AssertionError("frozen evaluation entered construction/editor")

    monkeypatch.setattr(PPCController, "run", forbidden)
    monkeypatch.setattr(Editor, "edit", forbidden)
    monkeypatch.setattr(Editor, "reflect", forbidden)
    monkeypatch.setattr(stream, "construct", forbidden)
    monkeypatch.setattr(stream, "plan", forbidden)
    original_prefixes = list(editor.prefixes)
    reports = []
    for arm in ("reference", "ppc_meta"):
        runner = Runner()
        directory = small_stream.output_dir / f"eval-{arm}"
        report = stream.evaluate(
            small_stream.output_dir / f"frozen-{arm}.json", directory, runner
        )
        assert report["status"] == "completed"
        assert report["requested"] == report["completed"] == 4
        assert len({spec.snapshot_hash for spec in runner.calls}) == 1
        assert {spec.case.seed for spec in runner.calls} == {1, 2}
        reports.append(directory / "evaluation_report.json")
    comparison = stream.compare(reports, small_stream.output_dir / "comparison.json")
    assert comparison["requested_per_arm"] == 4
    assert {row["arm"] for row in comparison["arms"]} == {"reference", "ppc_meta"}
    assert editor.prefixes == original_prefixes


@pytest.mark.parametrize("alias", ["construction", "evaluation"])
def test_frozen_evaluation_rejects_native_state_aliases_even_at_different_indices(
    small_stream, alias
):
    build(small_stream)
    report = stream.evaluate(
        small_stream.output_dir / "frozen-ppc_meta.json",
        small_stream.output_dir / "eval-alias",
        Runner(alias=alias),
    )
    assert report["status"] == "invalid"
    assert "overlaps" in report["error"]
    assert report["completed"] < report["requested"]


def test_construction_rejects_different_native_states_between_task_rollouts(
    small_stream,
):
    runner = Runner(changed_stage="inv")
    controller = PPCController(
        runner,
        Editor(),
        SnapshotStore(small_stream.output_dir / "snapshots"),
    )
    with pytest.raises(ValueError, match="PPC task did not complete"):
        stream.construct(small_stream, controller)
    assert len(runner.calls) == 2
    case = small_stream.cases(construction=True)[0]
    report = payload(
        small_stream.output_dir / "construction" / case.case_id / "task_report.json"
    )
    assert any("different native initial state" in error for error in report["errors"])
    assert not (small_stream.output_dir / "frozen-ppc_meta.json").exists()


def test_comparison_rejects_different_native_layouts_across_arms(small_stream):
    build(small_stream)
    paths = []
    for arm, suffix in (("reference", "original"), ("ppc_meta", "other-layout")):
        directory = small_stream.output_dir / f"eval-{arm}"
        report = stream.evaluate(
            small_stream.output_dir / f"frozen-{arm}.json",
            directory,
            Runner(identity_suffix=suffix),
        )
        assert report["status"] == "completed"
        paths.append(directory / "evaluation_report.json")
    with pytest.raises(ValueError, match="native initial states differ"):
        stream.compare(paths, small_stream.output_dir / "comparison.json")


@pytest.mark.parametrize("target", ["manifest", "index", "leaf", "task_audit"])
def test_frozen_inputs_reject_tampering_before_episode_admission(small_stream, target):
    build(small_stream)
    frozen = small_stream.output_dir / "frozen-ppc_meta.json"
    _, _, snapshot = stream.load_frozen(frozen)
    if target == "manifest":
        record = json.loads(frozen.read_text("utf-8"))
        record["payload"]["source_provenance"] = "unsealed replacement"
        frozen.write_text(json.dumps(record), encoding="utf-8")
    else:
        names = {
            "index": "MEMORY.md",
            "leaf": "global/rule.md",
            "task_audit": "task-specific/spatial_t0_s0.json",
        }
        (snapshot.corpus_dir / names[target]).write_text("tampered", encoding="utf-8")
    runner = Runner()
    with pytest.raises(ValueError):
        stream.evaluate(frozen, small_stream.output_dir / "eval-tampered", runner)
    assert runner.calls == []
