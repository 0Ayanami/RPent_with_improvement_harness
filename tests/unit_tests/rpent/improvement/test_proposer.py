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

"""Structured proposals use one tool-free request and account for failures."""

import json

import pytest
from pydantic_ai.models.test import TestModel

from rpent.improvement.proposer import (
    MAX_PROMPT_CHARS,
    SYSTEM_PROMPT,
    APIProposer,
    manual_proposal,
)
from rpent.memory.snapshots import SnapshotStore


@pytest.fixture
def baseline(tmp_path):
    source = tmp_path / "memory"
    (source / "global").mkdir(parents=True)
    (source / "global" / "recovery.md").write_text(
        "---\nid: recovery\nscope: global\nkind: failure\ntitle: Recovery\n"
        "applies_when: repeated grasp failure\nconfidence: single-shot\n"
        "evidence:\n  cells: [prior_episode]\n---\nObserve before retrying.\n",
        encoding="utf-8",
    )
    return SnapshotStore(tmp_path / "snapshots").capture(source)


def _patch(baseline):
    return {
        "patch_id": "patch1",
        "base_snapshot_hash": baseline.content_hash,
        "edit": {
            "operation": "revise",
            "path": "global/recovery.md",
            "expected_old_hash": baseline.files["global/recovery.md"],
            "body": "Observe the gripper after grasping before retrying.",
        },
        "hypotheses": ["The existing recovery rule lacks a visible trigger."],
        "expected_effect": "Avoid repeating the same unobserved recovery.",
        "falsifying_observation": "The same repetition occurs with the new rule read.",
        "evidence_refs": ["trial1/artifacts/states.json"],
    }


class CapturingModel(TestModel):
    async def request(self, messages, model_settings, model_request_parameters):
        self.captured_messages = messages
        return await super().request(messages, model_settings, model_request_parameters)


def _model(monkeypatch, output):
    import rpent.planner.base as base

    model = CapturingModel(custom_output_args=output)
    monkeypatch.setattr(base, "build_api_model", lambda *args: model)
    return model


def _payload(model):
    return json.loads(
        next(
            part.content
            for message in model.captured_messages
            for part in message.parts
            if part.part_kind == "user-prompt"
        )
    )


def test_proposal_receives_complete_leaf_and_has_no_function_tools(
    monkeypatch, baseline
):
    model = _model(monkeypatch, _patch(baseline))
    result = APIProposer("test:model").propose(
        baseline=baseline,
        evidence={"trial_ids": ["trial1"]},
        lessons=[],
        rejection="The previous patch had no falsifiable prediction.",
    )
    assert result.error is None
    assert result.patch.edit.expected_old_hash == baseline.files["global/recovery.md"]
    assert result.requests == 1
    assert result.input_tokens > 0
    assert model.last_model_request_parameters.function_tools == []
    payload = _payload(model)
    assert "Observe before retrying." in payload["memory_leaves"][0]["content"]
    assert (
        payload["memory_leaves"][0]["old_hash"] == baseline.files["global/recovery.md"]
    )
    assert "falsifiable" in payload["previous_rejection"]


def test_unknown_fields_are_rejected_without_an_automatic_retry(monkeypatch, baseline):
    output = _patch(baseline) | {"modify_system_prompt": True}
    model = _model(monkeypatch, output)
    result = APIProposer("test:model").propose(
        baseline=baseline, evidence={}, lessons=[]
    )
    assert result.patch is None
    assert result.error
    assert result.requests == 1
    assert model.last_model_request_parameters.function_tools == []
    assert manual_proposal(output).error


def test_provider_failure_still_counts_the_attempt(monkeypatch, baseline):
    import rpent.planner.base as base

    class FailingModel(TestModel):
        async def request(self, *args, **kwargs):
            raise RuntimeError("offline provider failure")

    monkeypatch.setattr(base, "build_api_model", lambda *args: FailingModel())
    result = APIProposer("test:model").propose(
        baseline=baseline, evidence={}, lessons=[]
    )
    assert result.patch is None
    assert "offline provider failure" in result.error
    assert result.requests == 1
    assert result.input_tokens is None
    assert result.output_tokens is None


def test_baseline_hash_cannot_be_changed(monkeypatch, baseline):
    _model(monkeypatch, _patch(baseline) | {"base_snapshot_hash": "another baseline"})
    result = APIProposer("test:model").propose(
        baseline=baseline, evidence={}, lessons=[]
    )
    assert result.patch is None
    assert "baseline snapshot hash" in result.error
    assert result.requests == 1


def test_truncation_preserves_all_artifact_references(monkeypatch, baseline):
    model = _model(monkeypatch, _patch(baseline))
    result = APIProposer("test:model").propose(
        baseline=baseline,
        evidence={
            "trajectory": "repeated tool output " * 10_000,
            "artifacts": {"states": "trial1/artifacts/states.json"},
            "evidence_refs": ["trial1", "trial2"],
        },
        lessons=[],
    )
    assert result.error is None
    payload = _payload(model)
    assert payload["evidence"]["truncated"] is True
    assert payload["evidence"]["artifact_refs"] == [
        "trial1",
        "trial1/artifacts/states.json",
        "trial2",
    ]
    assert len(json.dumps(payload)) + len(SYSTEM_PROMPT) < MAX_PROMPT_CHARS


def test_manual_patch_uses_identical_validation_without_model_calls(baseline):
    patch = _patch(baseline)
    result = manual_proposal(json.dumps(patch))
    assert result.error is None
    assert result.patch.patch_id == "patch1"
    assert result.requests == 0
    bad = patch | {"edit": patch["edit"] | {"path": "../../systemprompt.md"}}
    assert manual_proposal(bad).patch is None
    assert manual_proposal(bad).error


def test_trial_text_artifacts_are_read_and_inline_images_omitted(
    monkeypatch, baseline, tmp_path
):
    model = _model(monkeypatch, _patch(baseline))
    directory = tmp_path / "trial1"
    directory.mkdir()
    (directory / "transcript.json").write_text(
        json.dumps(
            {
                "messages": [
                    {
                        "role": "tool",
                        "content": "grasp returned without holding the cup",
                    },
                    {"type": "image_url", "image_url": "data:image/png;base64,PRIVATE"},
                ]
            }
        ),
        encoding="utf-8",
    )
    result = APIProposer("test:model").propose(
        baseline=baseline,
        evidence={
            "trials": [
                {
                    "spec": {"trial_id": "trial1", "output_dir": str(directory)},
                    "artifacts": {"transcript": "transcript.json"},
                }
            ]
        },
        lessons=[],
    )
    assert result.error is None
    rendered = json.dumps(_payload(model)["artifact_evidence"])
    assert "without holding the cup" in rendered
    assert "PRIVATE" not in rendered
    assert "omitted_from_text_evidence" in rendered


def test_trial_artifact_path_cannot_escape_its_output(monkeypatch, baseline, tmp_path):
    _model(monkeypatch, _patch(baseline))
    directory = tmp_path / "trial1"
    directory.mkdir()
    outside = tmp_path / "private.json"
    outside.write_text('{"private": "not trial evidence"}', encoding="utf-8")
    result = APIProposer("test:model").propose(
        baseline=baseline,
        evidence={
            "trials": [
                {
                    "spec": {"trial_id": "trial1", "output_dir": str(directory)},
                    "artifacts": {"states": str(outside)},
                }
            ]
        },
        lessons=[],
    )
    assert result.patch is None
    assert result.requests == 0
    assert "leaves its output directory" in result.error
