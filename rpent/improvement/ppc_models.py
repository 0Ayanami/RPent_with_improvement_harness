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

"""Typed records for memory-only k=1 propose, probe, compose and reflection."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import Field, model_validator

from rpent.improvement.models import Record, TaskCase, TrialResult
from rpent.memory.snapshots import MemoryEdit

PPCStage = Literal["inv", "dom", "compose"]


class MemoryRevision(Record):
    """A complete batch of leaf edits relative to one verified base snapshot.

    An empty batch explicitly keeps the base. Composer batches are relative to
    that same base, rather than incremental patches against either candidate.
    """

    schema_version: Literal[1] = 1
    revision_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_-]*$")
    base_snapshot_hash: str = Field(min_length=1)
    edits: list[MemoryEdit] = Field(default_factory=list)
    notes: str = Field(min_length=1)
    hypotheses: list[str] = Field(default_factory=list)
    expected_effect: str = ""
    falsifying_observation: str = ""
    evidence_refs: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_batch(self) -> MemoryRevision:
        """Reject ambiguous targets and unsupported, untestable changed batches."""
        paths = [edit.path.casefold() for edit in self.edits]
        if len(paths) != len(set(paths)):
            raise ValueError("revision contains duplicate edit paths")
        if self.edits and not all(
            (
                self.hypotheses,
                self.expected_effect.strip(),
                self.falsifying_observation.strip(),
                self.evidence_refs,
            )
        ):
            raise ValueError(
                "changed revisions require hypotheses, expected_effect, "
                "falsifying_observation and evidence_refs"
            )
        return self


class PPCModelCost(Record):
    """Observed model cost; unknown counters remain null, including on failure."""

    requests: int | None = Field(default=0, ge=0)
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    elapsed_s: float = Field(default=0.0, ge=0)
    cost_usd: float | None = Field(default=None, ge=0)


class PPCLesson(Record):
    """At most one reusable, evidence-cited lesson from a completed task."""

    cue: str = Field(min_length=1)
    lesson: str = Field(min_length=1)
    applies_when: str = Field(min_length=1)
    evidence_refs: list[str] = Field(min_length=1)
    counterevidence: list[str] = Field(default_factory=list)
    uncertainty: str = Field(min_length=1)
    confidence: Literal["low", "medium", "high"] = "low"


class PPCEditObservation(Record):
    """An execution contrast for one edited leaf, with attribution uncertainty."""

    path: str = Field(min_length=1)
    candidates: list[Literal["inv", "dom", "final"]] = Field(min_length=1)
    change: str = Field(min_length=1)
    observed: str = Field(min_length=1)
    verdict: Literal["helped", "hurt", "no_observed_effect", "confounded"]
    evidence_refs: list[str] = Field(min_length=1)


class PPCReflection(Record):
    """Task-end edit observations and an optional transferable lesson."""

    edits: list[PPCEditObservation] = Field(default_factory=list)
    lesson: PPCLesson | None = None
    notes: str = ""


class PPCEditResponse(PPCModelCost):
    """One admitted editing call, including failed and rejected requests."""

    revision: MemoryRevision | None = None
    error: str | None = None


class PPCReflectionResponse(PPCModelCost):
    """One reflection request; no lesson is an ordinary supported result."""

    reflection: PPCReflection | None = None
    error: str | None = None


class PPCStageResult(Record):
    """Materialized stage decision and all semantic attempts, in order."""

    stage: PPCStage
    snapshot_id: str
    snapshot_hash: str
    revision: MemoryRevision | None = None
    kept_base: bool = False
    fell_back: bool = False
    attempts: list[PPCEditResponse] = Field(default_factory=list)
    error: str | None = None


class PPCTaskResult(Record):
    """Four-rollout task record, independent of native-success ranking."""

    schema_version: Literal[1] = 1
    case: TaskCase
    status: Literal["completed", "stopped", "invalid"]
    baseline_id: str
    baseline_hash: str
    scaffold_hash: str
    inv_id: str | None = None
    dom_id: str | None = None
    final_id: str | None = None
    selected_snapshot_id: str
    trials: dict[str, TrialResult] = Field(default_factory=dict)
    stages: dict[str, PPCStageResult] = Field(default_factory=dict)
    reflection: PPCReflection | None = None
    reflection_cost: PPCModelCost = Field(default_factory=PPCModelCost)
    reflection_error: str | None = None
    total_cost: dict[str, Any] = Field(default_factory=dict)
    errors: list[str] = Field(default_factory=list)
