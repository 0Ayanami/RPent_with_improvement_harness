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

"""Task-local lesson retrieval and report-bound immutable publication."""

import json

import pytest

from rpent.improvement.library import ExperienceLibrary
from rpent.improvement.models import (
    ImprovementExperience,
    QualificationReport,
    TaskCase,
    fingerprint,
)


def _report(decision="accepted", *, suite="libero10", task_id=1):
    return QualificationReport(
        experiment_id="experiment1",
        protocol_hash="protocol1",
        scaffold_hash="scaffold1",
        baseline_id="baseline",
        baseline_hash="basehash",
        candidate_id="candidate",
        candidate_hash="candidatehash",
        patch_id="patch1",
        decision=decision,
        reasons=["Observed the predeclared paired executions."],
        trial_ids=["trial1", "trial2"],
        scope=[TaskCase(case_id="case1", suite=suite, task_id=task_id, seed=1)],
        total_cost={"wall_time_s": 12},
    )


def _lesson(identity, report, **updates):
    record = {
        "experience_id": identity,
        "suite": report.scope[0].suite,
        "task_id": report.scope[0].task_id,
        "cue": "When recovery repeats after a grasp",
        "lesson": "Inspect whether the memory rule was actually read before revising it.",
        "evidence_refs": ["trial1/artifacts/states.json", "trial2"],
        "decision": report.decision,
        "report_hash": fingerprint(report),
    }
    record.update(updates)
    return ImprovementExperience(**record)


@pytest.mark.parametrize(
    "decision", ["accepted", "rejected", "insufficient_evidence", "invalid"]
)
def test_all_decisions_can_preserve_cited_lessons(tmp_path, decision):
    report = _report(decision)
    lesson = _lesson("lesson1", report)
    library = ExperienceLibrary(tmp_path)
    path = library.append(lesson, report=report)
    assert library.append(lesson, report=report) == path
    assert library.freeze().retrieve(report.scope[0], "grasp", 10_000) == [lesson]
    assert len(list((tmp_path / "entries").glob("*.json"))) == 1
    assert not list(tmp_path.rglob("MEMORY.md"))


def test_frozen_prefix_and_returned_records_are_independent(tmp_path):
    report = _report()
    library = ExperienceLibrary(tmp_path)
    library.append(_lesson("lesson1", report), report=report)
    frozen = library.freeze()
    before_hash = frozen.content_hash
    library.append(_lesson("lesson2", report), report=report)
    retrieved = frozen.retrieve(report.scope[0], "grasp", 10_000)
    retrieved[0].evidence_refs.append("future evidence")
    again = frozen.retrieve(report.scope[0], "grasp", 10_000)
    assert len(again) == 1
    assert "future evidence" not in again[0].evidence_refs
    assert frozen.content_hash == before_hash
    assert library.freeze().content_hash != before_hash


def test_retrieval_is_scoped_deterministic_and_bounded(tmp_path):
    library = ExperienceLibrary(tmp_path)
    report = _report()
    other = _report(task_id=2)
    foreign = _report(suite="libero_spatial")
    library.append(_lesson("z_same_task", report), report=report)
    library.append(_lesson("a_other_task", other), report=other)
    library.append(_lesson("foreign", foreign), report=foreign)
    frozen = library.freeze()
    lessons = frozen.retrieve(report.scope[0], "grasp", 10_000)
    assert [item.experience_id for item in lessons] == ["z_same_task", "a_other_task"]
    encoded = json.dumps(
        lessons[0].model_dump(mode="json"),
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    )
    assert frozen.retrieve(report.scope[0], "grasp", len(encoded)) == lessons[:1]
    assert frozen.retrieve(report.scope[0], "grasp", 0) == []
    with pytest.raises(ValueError, match="nonnegative"):
        frozen.retrieve(report.scope[0], "grasp", -1)
    assert ExperienceLibrary(tmp_path).freeze().content_hash == frozen.content_hash


@pytest.mark.parametrize(
    "updates, match",
    [
        ({"report_hash": "other"}, "report_hash"),
        ({"decision": "rejected"}, "decision"),
        ({"task_id": 999}, "scope"),
        ({"evidence_refs": ["unrelated_trial/result.json"]}, "outside"),
    ],
)
def test_unrelated_report_or_evidence_is_rejected(tmp_path, updates, match):
    report = _report()
    with pytest.raises(ValueError, match=match):
        ExperienceLibrary(tmp_path).append(
            _lesson("lesson1", report, **updates), report=report
        )
    assert not (tmp_path / "entries").exists()


def test_existing_identity_cannot_be_replaced(tmp_path):
    report = _report()
    library = ExperienceLibrary(tmp_path)
    original = _lesson("lesson1", report)
    path = library.append(original, report=report)
    with pytest.raises(ValueError, match="already differs"):
        library.append(
            _lesson("lesson1", report, lesson="Changed lesson"), report=report
        )
    assert ImprovementExperience.model_validate_json(path.read_text()) == original


def test_freeze_rejects_changed_report(tmp_path):
    report = _report()
    library = ExperienceLibrary(tmp_path)
    library.append(_lesson("lesson1", report), report=report)
    path = tmp_path / "reports" / f"{fingerprint(report)}.json"
    content = json.loads(path.read_text())
    content["reasons"] = ["Changed after the lesson was written"]
    path.write_text(json.dumps(content), encoding="utf-8")
    with pytest.raises(ValueError, match="report_hash"):
        library.freeze()
