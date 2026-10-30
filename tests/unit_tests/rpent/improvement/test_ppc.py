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

"""Offline PPC scheduling, evidence isolation, fallback and model boundary tests."""

from __future__ import annotations

import asyncio
import copy
import json
from pathlib import Path

import pytest
from pydantic_ai.models.test import TestModel

from rpent.improvement.models import TaskCase, TrialBudget, TrialCost, TrialResult
from rpent.improvement.ppc import PPCController, model_evidence
from rpent.improvement.ppc_models import (
    MemoryRevision,
    PPCEditObservation,
    PPCEditResponse,
    PPCLesson,
    PPCReflection,
    PPCReflectionResponse,
)
from rpent.improvement.ppc_proposer import (
    APIEditor,
    CodexEditor,
    _leaf_contract,
    _payload,
    build_editor,
)
from rpent.memory.snapshots import MemoryEdit, SnapshotStore


@pytest.fixture
def setup(tmp_path):
    source = tmp_path / "source"
    (source / "global").mkdir(parents=True)
    for name in ("grasp", "observe"):
        (source / "global" / f"{name}.md").write_text(
            f"---\nid: {name}\nscope: global\nkind: strategy\ntitle: {name}\n"
            "applies_when: after grasp\nconfidence: single-shot\n"
            "evidence:\n  cells: [prior]\n---\nInspect before transport.\n",
            encoding="utf-8",
        )
    store = SnapshotStore(tmp_path / "snapshots")
    return (
        store,
        store.capture(source),
        TaskCase(case_id="build_task", suite="libero_spatial_task", task_id=0, seed=0),
    )


def revision(baseline, stage, trial_id="build_task_base"):
    names = {"inv": ["grasp"], "dom": ["observe"], "compose": ["grasp", "observe"]}[
        stage
    ]
    return MemoryRevision(
        revision_id=f"{stage}_revision",
        base_snapshot_hash=baseline.content_hash,
        edits=[
            MemoryEdit(
                operation="revise",
                path=f"global/{name}.md",
                expected_old_hash=baseline.files[f"global/{name}.md"],
                body=f"{stage}: observe object and gripper before transport.",
            )
            for name in names
        ],
        notes=f"{stage}: grounded in the observed grasp.",
        hypotheses=["The original rule lacks an observable trigger."],
        expected_effect="Avoid unobserved transport.",
        falsifying_observation="Transport still starts without an observation.",
        evidence_refs=[trial_id],
    )


class Runner:
    def __init__(self, *, failed_cleanup=None, stopped=None):
        self.calls = []
        self.failed_cleanup = failed_cleanup
        self.stopped = stopped

    def run(self, spec, snapshot):
        label = spec.output_dir.name
        self.calls.append((spec, snapshot))
        transcript = spec.output_dir / "transcript.json"
        transcript.write_text(
            json.dumps(
                {"messages": [{"tool": "observe", "terminated": label == "base"}]}
            ),
            encoding="utf-8",
        )
        private = spec.output_dir / "trial.json"
        private.write_text(
            '{"official_success":true,"private_marker":"SECRET"}', encoding="utf-8"
        )
        return TrialResult(
            spec=spec,
            outcome="budget_exceeded" if self.stopped == label else "completed",
            cleanup_ok=self.failed_cleanup != label,
            official_success=label == "base",  # Final regresses: PPC must not rank it.
            success_source="SECRET",
            initial_state_id="same-native-initial-state",
            cost=TrialCost(wall_time_s=0.1, env_steps=3, vla_calls=1),
            memory_reads=[
                {
                    "relative_path": "global/grasp.md",
                    "source_hash": snapshot.files["global/grasp.md"],
                }
            ],
            artifacts={"transcript": str(transcript), "result": str(private)},
        )


class Editor:
    def __init__(self, *, invalid=False, unknown_refs=False, no_reflection=False):
        self.calls = []
        self.reflections = []
        self.invalid = invalid
        self.unknown_refs = unknown_refs
        self.no_reflection = no_reflection

    def edit(self, *, stage, baseline, evidence, lessons, rejection=None):
        self.calls.append(
            (
                stage,
                baseline,
                copy.deepcopy(evidence),
                copy.deepcopy(lessons),
                rejection,
            )
        )
        # Attempts to modify one model context must not contaminate another.
        evidence["injected"] = "isolated"
        if lessons:
            lessons[0]["cue"] = "mutated"
        if self.invalid:
            return PPCEditResponse(
                requests=1, elapsed_s=0.01, error="invalid model output"
            )
        return PPCEditResponse(
            requests=1,
            input_tokens=10,
            output_tokens=20,
            revision=revision(
                baseline,
                stage,
                "held_out_trial" if self.unknown_refs else "build_task_base",
            ),
        )

    def reflect(self, *, baseline, evidence, lessons):
        self.reflections.append(copy.deepcopy(evidence))
        if self.no_reflection:
            return PPCReflectionResponse(requests=1, error="provider unavailable")
        if self.invalid or self.unknown_refs:
            return PPCReflectionResponse(
                requests=1, reflection=PPCReflection(notes="NO_ENTRY")
            )
        refs = ["build_task_base", "build_task_final"]
        return PPCReflectionResponse(
            requests=1,
            input_tokens=10,
            output_tokens=20,
            reflection=PPCReflection(
                edits=[
                    PPCEditObservation(
                        path="global/grasp.md",
                        candidates=["final"],
                        change="Added observation trigger",
                        observed="The final trace contains an observation",
                        verdict="confounded",
                        evidence_refs=refs,
                    )
                ],
                lesson=PPCLesson(
                    cue="unobserved transport",
                    lesson="Use an observable grasp check",
                    applies_when="a grasp precedes transport",
                    evidence_refs=refs,
                    uncertainty="One rollout and multiple edits cannot isolate the cause.",
                ),
            ),
        )


def run(setup, tmp_path, runner=None, editor=None, **kwargs):
    store, baseline, case = setup
    runner, editor = runner or Runner(), editor or Editor()
    result = PPCController(runner, editor, store).run(
        case,
        baseline,
        tmp_path / "run",
        TrialBudget(),
        scaffold_fingerprint="fixed_H0",
        **kwargs,
    )
    return result, runner, editor


def test_full_ppc_uses_independent_candidates_composes_and_runs_final(setup, tmp_path):
    _, baseline, _ = setup
    original = (baseline.corpus_dir / "global/grasp.md").read_bytes()
    prefix = {"cue": "original", "lesson": "prior task only"}
    result, runner, editor = run(setup, tmp_path, meta_lessons=(prefix,))
    assert result.status == "completed"
    assert [spec.output_dir.name for spec, _ in runner.calls] == [
        "base",
        "inv",
        "dom",
        "final",
    ]
    assert [call[0] for call in editor.calls] == ["inv", "dom", "compose"]
    assert all(call[1].snapshot_id == baseline.snapshot_id for call in editor.calls)
    assert editor.calls[0][2] == editor.calls[1][2]
    assert "injected" not in editor.calls[1][2]
    assert prefix["cue"] == editor.calls[1][3][0]["cue"] == "original"
    composed = editor.calls[2][2]
    assert set(composed["trials"]) == {"base", "inv", "dom"}
    assert set(composed["candidates"]) == {"inv", "dom"}
    assert (
        "inv: observe" in composed["candidates"]["inv"]["memory_leaves"][0]["content"]
    )
    assert result.selected_snapshot_id == result.final_id
    assert result.selected_snapshot_id != baseline.snapshot_id
    assert result.trials["base"].official_success is True
    assert result.trials["final"].official_success is False
    assert set(editor.reflections[0]["trials"]) == {"base", "inv", "dom", "final"}
    assert result.reflection.lesson.evidence_refs == [
        "build_task_base",
        "build_task_final",
    ]
    assert result.total_cost["trials"] == 4
    assert result.total_cost["env_steps"] == 12
    assert result.total_cost["improvement_requests"] == 4
    assert result.total_cost["input_tokens"] is None
    assert (baseline.corpus_dir / "global/grasp.md").read_bytes() == original
    exposed = json.dumps([call[2] for call in editor.calls] + editor.reflections)
    assert "official_success" not in exposed
    assert "success_source" not in exposed
    assert "SECRET" not in exposed
    assert '"terminated": true' in exposed
    assert (tmp_path / "run/final/trial_result.json").exists()
    assert (tmp_path / "run/ppc_task.json").exists()


@pytest.mark.parametrize("failure", ["invalid", "unknown_refs"])
def test_invalid_stages_retry_once_then_probe_base(setup, tmp_path, failure):
    result, runner, editor = run(setup, tmp_path, editor=Editor(**{failure: True}))
    assert result.status == "completed"
    assert len(runner.calls) == 4
    assert len(editor.calls) == 6
    assert result.selected_snapshot_id == result.baseline_id
    assert all(stage.fell_back for stage in result.stages.values())
    assert all(stage.kept_base for stage in result.stages.values())
    assert editor.calls[1][4]
    assert result.total_cost["improvement_requests"] == 7


def test_cleanup_failure_stops_before_another_episode_and_counts_prior_edits(
    setup, tmp_path
):
    result, runner, editor = run(setup, tmp_path, runner=Runner(failed_cleanup="inv"))
    assert result.status == "stopped"
    assert len(runner.calls) == 2
    assert len(editor.calls) == 2
    assert not editor.reflections
    assert result.total_cost["improvement_requests"] == 2
    assert result.selected_snapshot_id == result.baseline_id
    assert "cleanup" in result.errors[0]


def test_episode_budget_failure_stops_before_model_calls(setup, tmp_path):
    result, runner, editor = run(setup, tmp_path, runner=Runner(stopped="base"))
    assert result.status == "stopped"
    assert len(runner.calls) == 1
    assert not editor.calls


def test_reflection_provider_failure_does_not_fabricate_a_lesson(setup, tmp_path):
    result, _, _ = run(setup, tmp_path, editor=Editor(no_reflection=True))
    assert result.status == "completed"
    assert result.reflection is None
    assert result.reflection_error == "provider unavailable"
    assert result.reflection_cost.requests == 1
    assert result.total_cost["improvement_requests"] == 4


def test_reflection_exception_preserves_final_and_marks_unknown_cost(setup, tmp_path):
    class CrashingReflection(Editor):
        def reflect(self, **kwargs):
            raise RuntimeError("reflection transport vanished")

    result, runner, _ = run(setup, tmp_path, editor=CrashingReflection())
    assert result.status == "completed"
    assert len(runner.calls) == 4
    assert result.selected_snapshot_id == result.final_id
    assert result.reflection is None
    assert "reflection transport vanished" in result.reflection_error
    assert result.reflection_cost.requests is None
    assert result.total_cost["improvement_requests"] is None
    assert result.total_cost["known_subtotals"]["improvement_requests"] == 3


@pytest.mark.parametrize(
    "changed",
    [
        {"outcome": "infra_error"},
        {"official_success": None},
        {"initial_state_id": None},
        {"initial_state_id": "different-layout"},
    ],
)
def test_invalid_probe_stops_before_dom_probe_or_compose(setup, tmp_path, changed):
    class InvalidProbe(Runner):
        def run(self, spec, snapshot):
            result = super().run(spec, snapshot)
            return (
                result.model_copy(update=changed)
                if spec.output_dir.name == "inv"
                else result
            )

    result, runner, editor = run(setup, tmp_path, runner=InvalidProbe())
    assert result.status == "stopped"
    assert len(runner.calls) == 2
    assert len(editor.calls) == 2
    assert not editor.reflections
    assert result.total_cost["trials"] == 2


def test_h0_mutation_stops_and_preserves_completed_trial_cost(setup, tmp_path):
    store, baseline, case = setup
    h0 = ["before"]

    class ChangingRunner(Runner):
        def run(self, spec, snapshot):
            result = super().run(spec, snapshot)
            h0[0] = "after"
            return result

    result = PPCController(ChangingRunner(), Editor(), store).run(
        case,
        baseline,
        tmp_path / "run",
        TrialBudget(),
        scaffold_fingerprint=lambda: h0[0],
    )
    assert result.status == "invalid"
    assert result.total_cost["trials"] == 1
    assert "scaffold changed" in result.errors[0]


def test_model_evidence_cannot_read_private_result_as_transcript(setup, tmp_path):
    result, _, _ = run(setup, tmp_path)
    trial = result.trials["base"]
    forged = trial.model_copy(
        update={"artifacts": {"transcript": trial.artifacts["result"]}}
    )
    with pytest.raises(ValueError, match="invalid transcript"):
        model_evidence(forged)


def test_duplicate_leaf_revision_is_rejected(setup):
    _, baseline, _ = setup
    payload = revision(baseline, "inv").model_dump()
    payload["edits"] *= 2
    with pytest.raises(ValueError, match="duplicate"):
        MemoryRevision.model_validate(payload)


class CapturingModel(TestModel):
    async def request(self, messages, model_settings, model_request_parameters):
        self.captured_messages = messages
        return await super().request(messages, model_settings, model_request_parameters)


def test_api_editor_uses_one_request_without_function_tools(setup, monkeypatch):
    import rpent.planner.base as base

    _, baseline, _ = setup
    model = CapturingModel(
        custom_output_args=revision(baseline, "compose").model_dump(mode="json")
    )
    monkeypatch.setattr(base, "build_api_model", lambda *args: model)
    result = APIEditor("test:model").edit(
        stage="compose",
        baseline=baseline,
        evidence={"both_candidates": True},
        lessons=(),
    )
    assert result.error is None
    assert len(result.revision.edits) == 2
    assert result.requests == 1
    assert model.last_model_request_parameters.function_tools == []
    payload = json.loads(
        next(
            part.content
            for message in model.captured_messages
            for part in message.parts
            if part.part_kind == "user-prompt"
        )
    )
    assert len(payload["memory_leaves"]) == 2
    assert payload["evidence"]["both_candidates"] is True


def test_api_editor_failure_counts_attempt_without_output_retry(setup, monkeypatch):
    import rpent.planner.base as base

    class FailingModel(TestModel):
        async def request(self, *args, **kwargs):
            raise RuntimeError("offline provider failure")

    _, baseline, _ = setup
    monkeypatch.setattr(base, "build_api_model", lambda *args: FailingModel())
    result = APIEditor("test:model").edit(
        stage="inv", baseline=baseline, evidence={}, lessons=()
    )
    assert result.revision is None
    assert result.requests == 1
    assert "offline provider failure" in result.error
    assert result.input_tokens is None


def test_api_reflection_can_decline_lesson(setup, monkeypatch):
    import rpent.planner.base as base

    _, baseline, _ = setup
    model = CapturingModel(
        custom_output_args={
            "edits": [],
            "lesson": None,
            "notes": "No supported transferable contrast.",
        }
    )
    monkeypatch.setattr(base, "build_api_model", lambda *args: model)
    result = APIEditor("test:model").reflect(baseline=baseline, evidence={}, lessons=())
    assert result.error is None
    assert result.requests == 1
    assert result.reflection.lesson is None


@pytest.fixture
def codex_call(monkeypatch):
    import openai_codex

    state = {"events": [], "interrupted": False, "closed": False, "wait": False}

    class Turn:
        async def stream(self):
            for event in state["events"]:
                yield event
            if state["wait"]:
                await asyncio.Event().wait()

        async def interrupt(self):
            state["interrupted"] = True

    class Thread:
        async def turn(self, prompt, **kwargs):
            state["prompt"] = json.loads(prompt)
            state["turn_options"] = kwargs
            return Turn()

    class Client:
        def __init__(self, config):
            state["config"] = config

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            state["closed"] = True

        async def thread_start(self, **kwargs):
            state["thread_options"] = kwargs
            return Thread()

    monkeypatch.setattr(openai_codex, "AsyncCodex", Client)
    return state


def _codex_events(payload, *, output_tokens=23):
    return [
        {
            "method": "thread/tokenUsage/updated",
            "payload": {
                "token_usage": {
                    "total": {"input_tokens": 101, "output_tokens": output_tokens}
                }
            },
        },
        {
            "method": "item/completed",
            "payload": {
                "item": {
                    "type": "agentMessage",
                    "text": json.dumps({"payload": json.dumps(payload)}),
                }
            },
        },
        {"method": "turn/completed", "payload": {"turn": {"status": "completed"}}},
    ]


def test_codex_editor_structured_edit_reuses_auth_and_reports_observed_cost(
    setup, codex_call
):
    _, baseline, _ = setup
    proposed = revision(baseline, "inv")
    codex_call["events"] = _codex_events(proposed.model_dump(mode="json"))
    result = build_editor(planner="codex", model="gpt-5.5").edit(
        stage="inv", baseline=baseline, evidence={}, lessons=()
    )
    assert result.error is None
    assert result.revision == proposed
    assert (result.requests, result.input_tokens, result.output_tokens) == (1, 101, 23)
    assert codex_call["thread_options"]["model"] == "gpt-5.5"
    assert codex_call["thread_options"]["ephemeral"]
    assert "INV proposer" in codex_call["thread_options"]["base_instructions"]
    assert codex_call["turn_options"]["output_schema"]["additionalProperties"] is False
    assert (
        codex_call["prompt"]["context"]["base_snapshot_hash"] == baseline.content_hash
    )
    assert "mcp_servers={}" in codex_call["config"].config_overrides
    assert "features.shell_tool=false" in codex_call["config"].config_overrides
    assert codex_call["closed"]


@pytest.mark.parametrize("failure", ["invalid", "tools", "tokens", "timeout"])
def test_codex_editor_failures_retain_cost_and_stop_sdk_session(
    setup, codex_call, failure
):
    _, baseline, _ = setup
    codex_call["events"] = _codex_events(
        revision(baseline, "inv").model_dump(mode="json")
    )
    expected = {
        "invalid": "ValidationError",
        "tools": "tool call",
        "tokens": "max_tokens",
        "timeout": "TimeoutError",
    }
    kwargs = {}
    if failure == "invalid":
        codex_call["events"] = _codex_events({"revision_id": "invalid"})
    elif failure == "tools":
        codex_call["events"].insert(
            1,
            {
                "method": "item/completed",
                "payload": {
                    "item": {"type": "commandExecution", "command": "forbidden"}
                },
            },
        )
    elif failure == "tokens":
        kwargs["max_tokens"] = 1
    else:
        codex_call["events"] = []
        codex_call["wait"] = True
        kwargs["timeout_s"] = 0.01
    response = CodexEditor("gpt-5.5", **kwargs).edit(
        stage="inv", baseline=baseline, evidence={}, lessons=()
    )
    assert response.revision is None
    assert expected[failure] in response.error
    assert response.requests == (None if failure == "timeout" else 1)
    assert codex_call["interrupted"] is (failure in {"tools", "timeout"})
    assert codex_call["closed"]


def test_codex_reflection_can_decline_lesson(setup, codex_call):
    _, baseline, _ = setup
    codex_call["events"] = _codex_events(
        {"edits": [], "lesson": None, "notes": "No contrast."}
    )
    response = CodexEditor("gpt-5.5").reflect(
        baseline=baseline, evidence={}, lessons=()
    )
    assert response.error is None
    assert response.reflection.lesson is None
    assert isinstance(build_editor(model="test:model"), APIEditor)


def test_codex_configuration_uses_local_provider_settings():
    from rpent.improvement.models import ExecutionConfig
    from rpent.improvement.stream import EditorConfig

    assert EditorConfig(model="test:api").planner == "api"
    assert ExecutionConfig(model="test:api", vla_identity="offline").planner == "api"
    for config_type, kwargs in (
        (EditorConfig, {}),
        (ExecutionConfig, {"vla_identity": "offline"}),
    ):
        with pytest.raises(ValueError, match="CODEX_BASE_URL"):
            config_type(
                planner="codex",
                model="gpt-5.5",
                base_url="http://localhost:1",
                **kwargs,
            )


def test_empty_memory_prompt_supplies_valid_add_schema_and_both_layer_examples(
    tmp_path, monkeypatch
):
    import rpent.planner.base as base

    source = tmp_path / "empty"
    source.mkdir()
    store = SnapshotStore(tmp_path / "snapshots")
    baseline = store.capture(source)
    examples = _leaf_contract()["valid_add_examples"]
    assert {
        MemoryEdit.model_validate(example).metadata["scope"] for example in examples
    } == {"global", "task-family"}
    edit = copy.deepcopy(examples[0])
    edit["metadata"]["evidence"]["cells"] = ["build_task_base"]
    proposed = MemoryRevision(
        revision_id="first_leaf",
        base_snapshot_hash=baseline.content_hash,
        edits=[MemoryEdit.model_validate(edit)],
        notes="The base trace shows unobserved transport.",
        hypotheses=["A check may prevent premature transport."],
        expected_effect="Observe before transport",
        falsifying_observation="Transport still starts without observation",
        evidence_refs=["build_task_base"],
    )
    model = CapturingModel(custom_output_args=proposed.model_dump(mode="json"))
    monkeypatch.setattr(base, "build_api_model", lambda *args: model)
    response = APIEditor("test:model").edit(
        stage="inv", baseline=baseline, evidence={}, lessons=()
    )
    assert response.error is None
    payload = json.loads(
        next(
            part.content
            for message in model.captured_messages
            for part in message.parts
            if part.part_kind == "user-prompt"
        )
    )
    assert payload["memory_leaves"] == []
    assert "evidence" in payload["leaf_contract"]["global_required_metadata"]
    assert set(payload["leaf_contract"]["task_family_required_metadata"]) >= {
        "suite",
        "regime",
        "task_id",
        "task_language",
    }
    assert len(payload["leaf_contract"]["valid_add_examples"]) == 2
    candidate = store.apply_many(baseline, response.revision.edits)
    assert "global/grasp_observation.md" in candidate.files


def test_memory_sources_prioritize_read_leaves_and_keep_required_candidate_sources(
    setup,
):
    store, baseline, _ = setup
    large = store.apply_many(
        baseline,
        [
            MemoryEdit(
                operation="revise",
                path=path,
                expected_old_hash=baseline.files[path],
                body="x" * 15_000,
            )
            for path in ("global/grasp.md", "global/observe.md")
        ],
    )
    evidence = {
        "trials": {"base": {"memory_reads": [{"relative_path": "global/observe.md"}]}}
    }
    payload = json.loads(_payload(large, evidence, ()))
    assert [leaf["path"] for leaf in payload["memory_leaves"]] == ["global/observe.md"]
    assert payload["omitted_memory_leaves"] == [
        {"path": "global/grasp.md", "old_hash": large.files["global/grasp.md"]}
    ]
    evidence["candidates"] = {
        "inv": {"revision": revision(large, "inv").model_dump(mode="json")}
    }
    composed = json.loads(_payload(large, evidence, ()))
    assert "global/grasp.md" in {leaf["path"] for leaf in composed["memory_leaves"]}


def test_evidence_keeps_whole_head_and_tail_records_and_omission_counts(
    setup, tmp_path
):
    result, _, _ = run(setup, tmp_path)
    trial = result.trials["base"]
    messages = [{"role": "system", "content": "duplicate system prompt" * 3000}]
    messages.extend(
        {"role": "tool", "content": f"event-{index}:" + "x" * 600}
        for index in range(30)
    )
    Path(trial.artifacts["transcript"]).write_text(
        json.dumps({"messages": messages}), encoding="utf-8"
    )
    packet = model_evidence(trial)["artifacts"][0]
    visible = packet["data"]
    assert packet["omitted_system_messages"] == 1
    assert visible["total_records"] == 30
    assert visible["omitted_records"] > 0
    retained = {record["index"] for record in visible["records"]}
    assert 0 in retained and 29 in retained
    assert retained.isdisjoint(visible["omitted_indices"])
    assert len(retained) + visible["omitted_records"] == 30
    assert "duplicate system prompt" not in json.dumps(packet)
