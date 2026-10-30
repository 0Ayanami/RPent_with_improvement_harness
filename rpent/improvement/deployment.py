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

"""Resolve an explicit qualification report to a compatible deployment snapshot."""

from __future__ import annotations

from rpent.improvement.controller import scaffold_fingerprint
from rpent.improvement.models import (
    ExperimentConfig,
    QualificationReport,
    Record,
    TaskCase,
    fingerprint,
)
from rpent.memory.snapshots import MemorySnapshot, SnapshotStore


class DeploymentSelection(Record):
    """A configuration choice; deployment must still execute a new episode."""

    snapshot_id: str
    snapshot_hash: str
    fallback_id: str
    report_hash: str
    used_candidate: bool
    reason: str


def select_snapshot(
    config: ExperimentConfig,
    report: QualificationReport,
    case: TaskCase,
    store: SnapshotStore,
) -> tuple[DeploymentSelection, MemorySnapshot]:
    """Verify code, budget, task scope and snapshot content before selection."""
    if report.experiment_id != config.experiment_id:
        raise ValueError("report belongs to a different experiment")
    if report.scaffold_hash != scaffold_fingerprint(config):
        raise ValueError(
            "execution scaffold, model settings or budget changed; requalification is required"
        )
    if report.protocol_hash != fingerprint(config.protocol):
        raise ValueError("qualification protocol changed")
    protocol = config.protocol
    fallback_tasks = {
        (item.suite, item.task_id)
        for item in protocol.probe_cases
        + protocol.confirmation_cases
        + protocol.regression_cases
    }
    key = (case.suite, case.task_id)
    if key not in fallback_tasks:
        raise ValueError(
            "neither candidate nor fallback has a declared scope for this task"
        )
    fallback = store.load(report.baseline_id)
    if fallback.content_hash != report.baseline_hash:
        raise ValueError("fallback snapshot hash does not match qualification")
    scope = {(item.suite, item.task_id) for item in report.scope}
    use_candidate = report.decision == "accepted" and key in scope
    selected = fallback
    if use_candidate:
        if not report.candidate_id or not report.candidate_hash:
            raise ValueError("accepted report is missing the qualified candidate")
        selected = store.load(report.candidate_id)
        if selected.content_hash != report.candidate_hash:
            raise ValueError("candidate snapshot hash does not match qualification")
        if selected.parent_id != fallback.snapshot_id:
            raise ValueError("candidate does not descend from the declared E0")
    selection = DeploymentSelection(
        snapshot_id=selected.snapshot_id,
        snapshot_hash=selected.content_hash,
        fallback_id=fallback.snapshot_id,
        report_hash=fingerprint(report),
        used_candidate=use_candidate,
        reason="accepted snapshot in qualified task scope"
        if use_candidate
        else "qualification did not authorize the candidate; using declared E0",
    )
    return selection, selected
