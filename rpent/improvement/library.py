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

"""Immutable improvement lessons, kept outside the execution memory corpus.

The task-prefix library and evidence-supported lesson contract adapt Harness2
``core/library.py`` at commit ``1cfe254e8d25b15f37048c3730134686abe60335``.
Frozen views contain their records, so appending a later task cannot change the
lessons available to an improvement attempt already in progress.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

from rpent.improvement.models import (
    ImprovementExperience,
    QualificationReport,
    TaskCase,
    fingerprint,
)


def _encoded(value: dict | list) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def _write_immutable(path: Path, text: str) -> Path:
    """Publish complete bytes under the library's single-writer contract."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_text(encoding="utf-8") != text:
            raise ValueError(f"immutable library record already differs: {path}")
        return path
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        temporary.write_text(text, encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def _check_evidence(
    experience: ImprovementExperience, report: QualificationReport
) -> None:
    if experience.report_hash != fingerprint(report):
        raise ValueError("lesson report_hash does not match its qualification report")
    if experience.decision != report.decision:
        raise ValueError("lesson decision does not match its qualification report")
    if experience.patch is not None and experience.patch.patch_id != report.patch_id:
        raise ValueError("lesson patch does not match its qualification report")
    if not any(
        case.suite == experience.suite and case.task_id == experience.task_id
        for case in report.scope
    ):
        raise ValueError("lesson task is outside its qualification report scope")
    for reference in experience.evidence_refs:
        if (
            "\\" in reference
            or any(part in {".", "..", ""} for part in reference.split("/"))
            or not any(
                reference == trial_id
                or reference.startswith(f"{trial_id}/")
                or reference.startswith(f"trials/{trial_id}/")
                for trial_id in report.trial_ids
            )
        ):
            raise ValueError(
                f"lesson evidence is outside its report trials: {reference}"
            )


def _report_path(root: Path, digest: str) -> Path:
    if not re.fullmatch(r"[0-9a-f]{64}", digest):
        raise ValueError("lesson report_hash must be a SHA-256 content hash")
    return root / "reports" / f"{digest}.json"


@dataclass(frozen=True)
class LibrarySnapshot:
    """A content-addressed view whose retrieval never reads the mutable library."""

    content_hash: str
    _records: tuple[str, ...]

    def retrieve(
        self, case: TaskCase, query: str, char_budget: int
    ) -> list[ImprovementExperience]:
        """Return compatible lessons in deterministic relevance and size order.

        Only the requested suite is eligible. Same-task lessons precede other
        tasks; query-word overlap breaks ties, then the stable experience ID.
        The budget counts the complete canonical JSON of every returned lesson.
        Oversized lessons are skipped rather than partially shown to a model.
        """
        if char_budget < 0:
            raise ValueError("char_budget must be nonnegative")
        words = set(re.findall(r"\w+", query.casefold()))
        ranked = []
        for text in self._records:
            lesson = ImprovementExperience.model_validate_json(text)
            if lesson.suite != case.suite:
                continue
            searchable = set(
                re.findall(r"\w+", f"{lesson.cue} {lesson.lesson}".casefold())
            )
            ranked.append(
                (
                    lesson.task_id != case.task_id,
                    -len(words & searchable),
                    lesson.experience_id,
                    text,
                )
            )
        selected = []
        remaining = char_budget
        for _other_task, _overlap, _identity, text in sorted(ranked):
            if len(text) <= remaining:
                selected.append(ImprovementExperience.model_validate_json(text))
                remaining -= len(text)
        return selected


class ExperienceLibrary:
    """Append cited lessons and freeze task-local views of all previous lessons.

    This single-writer store contains analysis metadata and reports. Its root
    must never be passed to the execution planner as a memory corpus.
    """

    def __init__(self, root: Path | str) -> None:
        self.root = Path(root)

    def append(
        self,
        experience: ImprovementExperience,
        *,
        report: QualificationReport,
    ) -> Path:
        """Store a lesson from any decision, idempotently for the same ID/content.

        Args:
            experience: A lesson citing trials within the report's task scope.
            report: The immutable qualification evidence supporting the lesson.

        Returns:
            The lesson record path outside any execution corpus.

        """
        experience = ImprovementExperience.model_validate(experience.model_dump())
        report = QualificationReport.model_validate(report.model_dump())
        _check_evidence(experience, report)
        _write_immutable(
            _report_path(self.root, experience.report_hash),
            _encoded(report.model_dump(mode="json")),
        )
        return _write_immutable(
            self.root / "entries" / f"{experience.experience_id}.json",
            _encoded(experience.model_dump(mode="json")),
        )

    def freeze(self) -> LibrarySnapshot:
        """Validate persisted evidence and materialize an immutable library version."""
        records = []
        for path in sorted((self.root / "entries").glob("*.json")):
            lesson = ImprovementExperience.model_validate_json(
                path.read_text(encoding="utf-8")
            )
            if path.stem != lesson.experience_id:
                raise ValueError(f"library record ID differs from filename: {path}")
            report_path = _report_path(self.root, lesson.report_hash)
            report = QualificationReport.model_validate_json(
                report_path.read_text(encoding="utf-8")
            )
            _check_evidence(lesson, report)
            records.append(lesson.model_dump(mode="json"))
        digest = fingerprint(records)
        _write_immutable(
            self.root / "versions" / f"{digest}.json",
            _encoded({"schema_version": 1, "content_hash": digest, "entries": records}),
        )
        return LibrarySnapshot(digest, tuple(_encoded(record) for record in records))
