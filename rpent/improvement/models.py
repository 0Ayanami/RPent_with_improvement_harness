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

"""Versioned contracts for memory proposals, trials and adoption evidence."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from rpent.memory.snapshots import MemoryEdit

Decision = Literal["accepted", "rejected", "insufficient_evidence", "invalid"]


def initial_state_key(identity: str) -> str:
    """Compare layout bytes within a task even when catalog indices differ."""
    return re.sub(r":i[0-9]+(?=:sha256:)", "", identity)


def fingerprint(value: Any) -> str:
    """Hash JSON-compatible input independently of dictionary insertion order."""
    if isinstance(value, BaseModel):
        value = value.model_dump(mode="json")
    encoded = json.dumps(
        value, sort_keys=True, ensure_ascii=False, separators=(",", ":")
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


class Record(BaseModel):
    """Reject unknown fields in persisted records and model responses."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class TaskCase(Record):
    """One requested LIBERO episode; resolved layout identity comes from the runner."""

    case_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_-]*$")
    suite: str = Field(min_length=1)
    task_id: int = Field(ge=0)
    seed: int = Field(ge=0)


class TrialBudget(Record):
    """Admission limits for an episode, with a timeout that includes tool draining."""

    max_turns: int = Field(default=50, gt=0)
    timeout_s: int = Field(default=1200, gt=0)
    max_env_steps: int | None = Field(default=None, gt=0)
    max_vla_calls: int | None = Field(default=None, gt=0)


class RemoteExecutionConfig(Record):
    """Loopback endpoint forwarded to the trial worker over SSH."""

    endpoint: str = "http://127.0.0.1:8870"
    request_timeout_s: int = Field(default=1200, gt=0, le=6000)

    @model_validator(mode="after")
    def validate_endpoint(self) -> RemoteExecutionConfig:
        from urllib.parse import urlsplit

        url = urlsplit(self.endpoint)
        if (
            url.scheme != "http"
            or url.hostname not in {"127.0.0.1", "localhost", "::1"}
            or url.username is not None
            or url.password is not None
            or url.path not in {"", "/"}
            or url.query
            or url.fragment
        ):
            raise ValueError("remote endpoint must be a loopback HTTP SSH tunnel URL")
        if url.port is None:
            raise ValueError("remote endpoint requires an explicit forwarded port")
        return self


class ExecutionConfig(Record):
    """Frozen execution settings; endpoint identities must describe the actual services."""

    robot: Literal["libero"] = "libero"
    planner: Literal["api", "codex"] = "api"
    model: str = Field(min_length=1)
    vla_identity: str = Field(min_length=1)
    base_url: str | None = None
    max_tokens: int = Field(default=8192, gt=0)
    reasoning_effort: Literal["none", "low", "medium", "high", "xhigh"] = "none"
    robot_options: dict[str, Any] = Field(default_factory=dict)
    budget: TrialBudget = Field(default_factory=TrialBudget)
    remote: RemoteExecutionConfig | None = None

    @model_validator(mode="after")
    def validate_codex_endpoint(self) -> ExecutionConfig:
        """Keep Codex endpoint selection consistent with the ordinary planner."""
        if self.planner == "codex" and self.base_url is not None:
            raise ValueError("Codex uses CODEX_BASE_URL; omit execution.base_url")
        if self.remote is not None and self.planner != "codex":
            raise ValueError("remote trials currently require planner=codex")
        return self


class TrialSpec(Record):
    """All inputs needed to identify one independent execution."""

    schema_version: Literal[1] = 1
    trial_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_-]*$")
    experiment_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_-]*$")
    case: TaskCase
    snapshot_id: str
    snapshot_hash: str
    scaffold_hash: str
    budget: TrialBudget
    output_dir: Path


class TrialCost(Record):
    """Measured costs; unavailable counters remain null rather than zero."""

    wall_time_s: float = Field(default=0.0, ge=0)
    planner_requests: int | None = Field(default=None, ge=0)
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    env_steps: int | None = Field(default=None, ge=0)
    vla_calls: int | None = Field(default=None, ge=0)
    resets: int | None = Field(default=None, ge=0)
    unknown: list[str] = Field(default_factory=list)


class TrialResult(Record):
    """Official outcome and cleanup evidence, independent of the planner's finish text."""

    schema_version: Literal[1] = 1
    spec: TrialSpec
    outcome: Literal["completed", "budget_exceeded", "cancelled", "infra_error"]
    official_success: bool | None = None
    success_source: str = "unavailable"
    initial_state_id: str | None = None
    terminated: bool = False
    truncated: bool = False
    cleanup_ok: bool = False
    cost: TrialCost = Field(default_factory=TrialCost)
    memory_reads: list[dict[str, Any]] = Field(default_factory=list)
    artifacts: dict[str, str] = Field(default_factory=dict)
    error: str | None = None


class ExperiencePatch(Record):
    """One leaf edit with a testable prediction and references to observed evidence."""

    schema_version: Literal[1] = 1
    patch_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_-]*$")
    base_snapshot_hash: str
    edit: MemoryEdit
    hypotheses: list[str] = Field(min_length=1)
    expected_effect: str = Field(min_length=1)
    falsifying_observation: str = Field(min_length=1)
    evidence_refs: list[str] = Field(min_length=1)


class ImprovementProtocol(Record):
    """Predeclared probe and held-out confirmation cases for one memory edit."""

    schema_version: Literal[1] = 1
    protocol_id: str = Field(min_length=1)
    probe_cases: list[TaskCase] = Field(min_length=1)
    confirmation_cases: list[TaskCase] = Field(min_length=1)
    regression_cases: list[TaskCase] = Field(default_factory=list)
    max_trials: int = Field(default=20, gt=0)
    max_proposals: int = Field(default=2, ge=1, le=2)
    meta_char_budget: int = Field(default=8000, ge=0)
    require_candidate_read: bool = True

    @model_validator(mode="after")
    def distinct_cases(self) -> ImprovementProtocol:
        cases = self.probe_cases + self.confirmation_cases + self.regression_cases
        ids = [case.case_id for case in cases]
        keys = [(case.suite, case.task_id, case.seed) for case in cases]
        if len(set(ids)) != len(ids) or len(set(keys)) != len(keys):
            raise ValueError(
                "case IDs and requested episode conditions must be distinct"
            )
        proposed_tasks = {(case.suite, case.task_id) for case in self.probe_cases}
        confirmed_tasks = {
            (case.suite, case.task_id) for case in self.confirmation_cases
        }
        if not proposed_tasks.issubset(confirmed_tasks):
            raise ValueError("each probed task requires held-out confirmation")
        return self


class QualificationReport(Record):
    """A decision bound to complete snapshots, execution settings and a protocol."""

    schema_version: Literal[1] = 1
    experiment_id: str
    protocol_hash: str
    scaffold_hash: str
    baseline_id: str
    baseline_hash: str
    candidate_id: str | None = None
    candidate_hash: str | None = None
    patch_id: str | None = None
    decision: Decision
    reasons: list[str]
    trial_ids: list[str]
    scope: list[TaskCase]
    total_cost: dict[str, Any]
    limitations: list[str] = Field(default_factory=list)


class ImprovementExperience(Record):
    """A cited observation about an edit, including counterevidence and limitations."""

    schema_version: Literal[1] = 1
    experience_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_-]*$")
    robot: Literal["libero"] = "libero"
    suite: str
    task_id: int = Field(ge=0)
    cue: str = Field(min_length=1)
    lesson: str = Field(min_length=1)
    evidence_refs: list[str] = Field(min_length=1)
    counterevidence: list[str] = Field(default_factory=list)
    decision: Decision
    report_hash: str
    patch: ExperiencePatch | None = None


class ExperimentConfig(Record):
    """Portable, explicit inputs for a manual or automatic improvement run."""

    schema_version: Literal[1] = 1
    experiment_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_-]*$")
    output_dir: Path
    baseline_memory: Path
    execution: ExecutionConfig
    protocol: ImprovementProtocol
    patch_file: Path | None = None
    proposer_model: str | None = None
    proposer_base_url: str | None = None
    library_dir: Path | None = None

    @model_validator(mode="after")
    def proposal_source(self) -> ExperimentConfig:
        if (self.patch_file is None) == (self.proposer_model is None):
            raise ValueError("set exactly one of patch_file or proposer_model")
        return self
