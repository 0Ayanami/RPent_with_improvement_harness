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

"""Memory-only Harness2 k=1 recursion with task-end evidence reflection.

The schedule adapts ``core/{recursion,selection}.py`` at Harness2 revision
``1cfe254e8d25b15f37048c3730134686abe60335``. Normal tool observations may
contain native terminal signals; private evaluation records are not model input.
"""

from __future__ import annotations

import copy
import hashlib
import json
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, Protocol

from rpent.evaluation.result import write_json_atomic
from rpent.improvement.controller import TrialRunner, _sum_costs
from rpent.improvement.models import TaskCase, TrialBudget, TrialResult, TrialSpec
from rpent.improvement.ppc_models import (
    MemoryRevision,
    PPCEditResponse,
    PPCLesson,
    PPCModelCost,
    PPCReflectionResponse,
    PPCStage,
    PPCStageResult,
    PPCTaskResult,
)
from rpent.improvement.proposer import _without_images
from rpent.memory.snapshots import MemorySnapshot, SnapshotStore

LessonInput = PPCLesson | Mapping[str, Any]
MAX_ARTIFACT_BYTES = 2_000_000
ARTIFACT_RECORD_CHARS = 6000


class PPCEditor(Protocol):
    """Model boundary; the controller owns semantic repair and validation."""

    def edit(
        self,
        *,
        stage: PPCStage,
        baseline: MemorySnapshot,
        evidence: dict[str, Any],
        lessons: Sequence[LessonInput],
        rejection: str | None = None,
    ) -> PPCEditResponse:
        """Propose one complete batch relative to the supplied baseline."""
        ...

    def reflect(
        self,
        *,
        baseline: MemorySnapshot,
        evidence: dict[str, Any],
        lessons: Sequence[LessonInput],
    ) -> PPCReflectionResponse:
        """Inspect the whole task record after its final rollout."""
        ...


def memory_sources(snapshot: MemorySnapshot) -> list[dict[str, str]]:
    """Read complete editable leaves, verifying their expected source hashes."""
    leaves = []
    for relative, expected in sorted(snapshot.files.items()):
        if not relative.startswith(("global/", "task-family/")):
            continue
        payload = (snapshot.corpus_dir / relative).read_bytes()
        if "sha256:" + hashlib.sha256(payload).hexdigest() != expected:
            raise ValueError(f"memory source changed: {relative}")
        leaves.append(
            {"path": relative, "old_hash": expected, "content": payload.decode("utf-8")}
        )
    return leaves


def model_evidence(result: TrialResult) -> dict[str, Any]:
    """Project ordinary execution evidence without dumping private trial results.

    Only explicitly named, trial-owned transcript/states JSON is read. Normal
    tool messages, including their observable completion signals, are retained.
    Native evaluation flags and separate result, budget or grader files are not
    supplied to improvement models.
    """
    packets = []
    directory = result.spec.output_dir.resolve()
    for name in ("transcript", "states"):
        reference = result.artifacts.get(name)
        if not reference:
            continue
        path = Path(reference)
        resolved = (path if path.is_absolute() else directory / path).resolve()
        if resolved.name != f"{name}.json" or not resolved.is_relative_to(directory):
            raise ValueError(f"invalid {name} evidence artifact: {reference}")
        packet: dict[str, Any] = {"name": name, "source_ref": str(resolved)}
        try:
            with resolved.open("rb") as file:
                payload = file.read(MAX_ARTIFACT_BYTES + 1)
            if len(payload) > MAX_ARTIFACT_BYTES:
                raise ValueError("artifact exceeds the bounded JSON parsing byte limit")
            data = _without_images(json.loads(payload))
            if not isinstance(data, dict):
                raise ValueError("artifact must contain a JSON object")
            records = (
                data.get("messages", [])
                if name == "transcript"
                else data.get("steps", [])
            )
            if not isinstance(records, list):
                raise ValueError("artifact messages/steps must be a list")
            system_count = 0
            if name == "transcript":
                system_count = sum(
                    isinstance(record, dict) and record.get("role") == "system"
                    for record in records
                )
                records = [
                    record
                    for record in records
                    if not isinstance(record, dict) or record.get("role") != "system"
                ]
                if data.get("finish") is not None:
                    records = [*records, {"finish": data["finish"]}]
            visible = _bounded_records(records, ARTIFACT_RECORD_CHARS)
            packet.update(
                data=visible,
                omitted_system_messages=system_count,
                source_bytes=len(payload),
                source_sha256=hashlib.sha256(payload).hexdigest(),
            )
        except (OSError, ValueError) as exc:
            packet["unavailable"] = f"{type(exc).__name__}: {exc}"
        packets.append(packet)
    return {
        "trial_id": result.spec.trial_id,
        "case": result.spec.case.model_dump(mode="json"),
        "snapshot_hash": result.spec.snapshot_hash,
        "memory_reads": copy.deepcopy(result.memory_reads),
        "execution_error": result.error,
        "artifacts": packets,
    }


def _bounded_records(records: list[Any], char_budget: int) -> dict[str, Any]:
    """Select whole first/last records without breaking JSON or hiding omissions."""
    remaining = char_budget
    selected = {}
    order = []
    for index in range((len(records) + 1) // 2):
        order.append(index)
        opposite = len(records) - 1 - index
        if opposite != index:
            order.append(opposite)
    for index in order:
        record = records[index]
        length = len(json.dumps(record, ensure_ascii=False))
        if length <= remaining:
            selected[index] = record
            remaining -= length
    omitted = [index for index in range(len(records)) if index not in selected]
    return {
        "records": [
            {"index": index, "data": record}
            for index, record in sorted(selected.items())
        ],
        "total_records": len(records),
        "omitted_records": len(omitted),
        "omitted_indices": omitted,
    }


def _validate_refs(refs: Sequence[str], known: set[str]) -> None:
    for ref in refs:
        parts = ref.split("/")
        if "\\" in ref or any(part in {"", ".", ".."} for part in parts):
            raise ValueError(f"invalid evidence reference: {ref}")
        trial_id = parts[1] if parts[0] == "trials" and len(parts) > 1 else parts[0]
        if trial_id not in known:
            raise ValueError(f"evidence does not reference an observed trial: {ref}")


class _StopPPC(RuntimeError):
    """Stop episode admission after a recorded failure or invariant violation."""


class PPCController:
    """Run base, independent INV/DOM probes, composed final and reflection.

    Environment episodes run serially. Every editing stage has at most two
    semantic attempts; invalid output falls back to the original task baseline.
    The one final is adopted without ranking native-success measurements.
    """

    def __init__(
        self, runner: TrialRunner, editor: PPCEditor, snapshot_store: SnapshotStore
    ) -> None:
        self.runner = runner
        self.editor = editor
        self.snapshot_store = snapshot_store

    def run(
        self,
        case: TaskCase,
        baseline: MemorySnapshot,
        output_dir: Path | str,
        budget: TrialBudget,
        meta_lessons: Sequence[LessonInput] = (),
        scaffold_fingerprint: str | Callable[[], str] = "",
    ) -> PPCTaskResult:
        """Build one task revision and persist its complete execution evidence.

        Args:
            case: One build episode condition, reused by all four rollouts.
            baseline: Verified task-start execution corpus.
            output_dir: Fresh directory containing stage and trial records.
            budget: The same admission limits for each rollout.
            meta_lessons: Read-only prefix from tasks finished before this task.
            scaffold_fingerprint: H0 identity, or a callable checked at boundaries.

        Returns:
            A completed task with the composed snapshot selected, or a stopped
            record that retains the baseline and all costs observed so far.

        Raises:
            ValueError: Inputs are invalid or the output directory was used.
        """
        started = time.monotonic()
        store = self.snapshot_store
        baseline = store.verify(baseline)
        root = Path(output_dir).resolve()
        if root.exists() and any(root.iterdir()):
            raise ValueError("PPC output directory already contains records")
        root.mkdir(parents=True, exist_ok=True)
        h0 = (
            scaffold_fingerprint()
            if callable(scaffold_fingerprint)
            else scaffold_fingerprint
        )
        if not h0:
            raise ValueError("a nonempty execution scaffold fingerprint is required")
        # Round-trip records once: models and fakes cannot mutate the caller's prefix.
        lessons = tuple(
            copy.deepcopy(
                lesson.model_dump(mode="json")
                if isinstance(lesson, PPCLesson)
                else dict(lesson)
            )
            for lesson in meta_lessons
        )
        trials: dict[str, TrialResult] = {}
        stages: dict[str, PPCStageResult] = {}
        admitted_responses: list[dict[str, Any]] = []
        snapshots = {"base": baseline}
        reflection = None
        reflection_response = PPCReflectionResponse()
        status = "completed"
        errors = []

        def invariant() -> None:
            store.verify(baseline)
            if callable(scaffold_fingerprint) and scaffold_fingerprint() != h0:
                raise ValueError("execution scaffold changed during PPC")

        def event(directory: Path, state: str, **fields: Any) -> None:
            with (directory / "trial_events.jsonl").open("a", encoding="utf-8") as file:
                file.write(json.dumps({"run_state": state, **fields}) + "\n")

        def roll(label: str, snapshot: MemorySnapshot) -> TrialResult:
            invariant()
            store.verify(snapshot)
            directory = root / label
            directory.mkdir()
            spec = TrialSpec(
                trial_id=f"{case.case_id}_{label}",
                experiment_id=case.case_id,
                case=case,
                snapshot_id=snapshot.snapshot_id,
                snapshot_hash=snapshot.content_hash,
                scaffold_hash=h0,
                budget=budget,
                output_dir=directory,
            )
            write_json_atomic(
                directory / "trial_spec.json", spec.model_dump(mode="json")
            )
            event(directory, "running")
            try:
                result = self.runner.run(spec, snapshot)
                if result.spec != spec:
                    raise ValueError(
                        "runner returned results for different trial inputs"
                    )
            except BaseException:
                event(directory, "interrupted", cost="unknown")
                raise
            trials[label] = result
            write_json_atomic(
                directory / "trial_result.json", result.model_dump(mode="json")
            )
            event(directory, "finalized", cost=result.cost.model_dump(mode="json"))
            store.verify(snapshot)
            invariant()
            if not result.cleanup_ok:
                raise _StopPPC("trial cleanup failed; refusing another episode")
            if result.outcome != "completed":
                raise _StopPPC(f"{label} execution stopped: {result.outcome}")
            if result.official_success is None or result.initial_state_id is None:
                raise _StopPPC(f"{label} execution has unavailable native evidence")
            if (
                label != "base"
                and result.initial_state_id != trials["base"].initial_state_id
            ):
                raise _StopPPC(
                    f"{label} execution resolved to a different native initial state"
                )
            return result

        def candidate_evidence(label: str, stage: PPCStageResult) -> dict[str, Any]:
            snapshot = snapshots[label]
            changed = (
                {edit.path for edit in stage.revision.edits}
                if stage.revision
                else set()
            )
            return {
                "snapshot_id": snapshot.snapshot_id,
                "snapshot_hash": snapshot.content_hash,
                "inherits_unedited_leaves_from": baseline.content_hash,
                "memory_leaves": [
                    leaf for leaf in memory_sources(snapshot) if leaf["path"] in changed
                ],
                "removed_paths": sorted(changed - set(snapshot.files)),
                "revision": stage.revision.model_dump(mode="json")
                if stage.revision
                else None,
                "kept_base": stage.kept_base,
                "fell_back": stage.fell_back,
                "error": stage.error,
            }

        def evidence(labels: Sequence[str]) -> dict[str, Any]:
            return {
                "task": case.model_dump(mode="json"),
                "base_snapshot_hash": baseline.content_hash,
                "trials": {label: model_evidence(trials[label]) for label in labels},
                "candidates": {
                    label: candidate_evidence(label, stage)
                    for label, stage in stages.items()
                    if label in labels
                },
            }

        def edit(stage: PPCStage, observed: dict[str, Any]) -> MemorySnapshot:
            write_json_atomic(root / f"{stage}_evidence.json", observed)
            attempts = []
            rejection = None
            candidate = baseline
            revision = None
            for attempt in range(1, 3):
                invariant()
                reply = self.editor.edit(
                    stage=stage,
                    baseline=baseline,
                    evidence=copy.deepcopy(observed),
                    lessons=copy.deepcopy(lessons),
                    rejection=rejection,
                )
                reply = PPCEditResponse.model_validate(reply.model_dump())
                attempts.append(reply)
                admitted_responses.append(reply.model_dump(mode="json"))
                write_json_atomic(
                    root / f"{stage}_attempt_{attempt}.json",
                    reply.model_dump(mode="json"),
                )
                invariant()
                try:
                    if reply.revision is None:
                        raise ValueError(reply.error or "no revision returned")
                    proposed = MemoryRevision.model_validate(
                        reply.revision.model_dump()
                    )
                    if proposed.base_snapshot_hash != baseline.content_hash:
                        raise ValueError(
                            "revision changed the task baseline snapshot hash"
                        )
                    _validate_refs(
                        proposed.evidence_refs,
                        {t.spec.trial_id for t in trials.values()},
                    )
                    candidate = (
                        store.apply_many(baseline, proposed.edits)
                        if proposed.edits
                        else baseline
                    )
                    revision = proposed
                    rejection = None
                    break
                except (ValueError, OSError) as exc:
                    rejection = str(exc)
            record = PPCStageResult(
                stage=stage,
                snapshot_id=candidate.snapshot_id,
                snapshot_hash=candidate.content_hash,
                revision=revision,
                kept_base=candidate.content_hash == baseline.content_hash,
                fell_back=revision is None,
                attempts=attempts,
                error=rejection,
            )
            stages[stage] = record
            snapshots["final" if stage == "compose" else stage] = candidate
            write_json_atomic(
                root / f"{stage}_stage.json", record.model_dump(mode="json")
            )
            return candidate

        write_json_atomic(
            root / "task_manifest.json",
            {
                "case": case.model_dump(mode="json"),
                "baseline_id": baseline.snapshot_id,
                "baseline_hash": baseline.content_hash,
                "scaffold_hash": h0,
                "budget": budget.model_dump(mode="json"),
                "meta_lessons": list(lessons),
                "k": 1,
                "feedback": "normal tool observations only; private evaluation records excluded",
            },
        )
        try:
            roll("base", baseline)
            shared = evidence(["base"])
            inv = edit("inv", shared)
            dom = edit("dom", shared)
            roll("inv", inv)
            roll("dom", dom)
            final = edit("compose", evidence(["base", "inv", "dom"]))
            roll("final", final)
            record = evidence(["base", "inv", "dom", "final"])
            record["candidates"]["final"] = candidate_evidence(
                "final", stages["compose"]
            )
            record["adoption"] = {
                "label": "final",
                "reason": "k=1 has one composed final",
            }
            write_json_atomic(root / "reflection_evidence.json", record)
            invariant()
            reflection_started = time.monotonic()
            try:
                reflection_response = self.editor.reflect(
                    baseline=baseline,
                    evidence=copy.deepcopy(record),
                    lessons=copy.deepcopy(lessons),
                )
                reflection_response = PPCReflectionResponse.model_validate(
                    reflection_response.model_dump()
                )
            except Exception as exc:
                reflection_response = PPCReflectionResponse(
                    requests=None,
                    elapsed_s=time.monotonic() - reflection_started,
                    error=f"{type(exc).__name__}: {exc}; model request cost unknown",
                )
            admitted_responses.append(reflection_response.model_dump(mode="json"))
            if reflection_response.reflection is not None:
                proposed = reflection_response.reflection
                known = {t.spec.trial_id for t in trials.values()}
                try:
                    for observation in proposed.edits:
                        _validate_refs(observation.evidence_refs, known)
                        for label in observation.candidates:
                            stage = stages.get("compose" if label == "final" else label)
                            if (
                                stage is None
                                or stage.revision is None
                                or observation.path
                                not in {edit.path for edit in stage.revision.edits}
                            ):
                                raise ValueError(
                                    "reflection attributes an unobserved candidate edit"
                                )
                    if proposed.lesson is not None:
                        _validate_refs(proposed.lesson.evidence_refs, known)
                        if not proposed.edits:
                            raise ValueError(
                                "reflection lesson has no supporting edit observations"
                            )
                    reflection = proposed
                except ValueError as exc:
                    reflection_response = reflection_response.model_copy(
                        update={"reflection": None, "error": str(exc)}
                    )
            invariant()
        except _StopPPC as exc:
            status = "stopped"
            errors.append(str(exc))
        except (ValueError, OSError) as exc:
            status = "invalid"
            errors.append(f"{type(exc).__name__}: {exc}")
        total_cost = _sum_costs(list(trials.values()), admitted_responses)
        total_cost["adaptation_wall_time_s"] = time.monotonic() - started
        result = PPCTaskResult(
            case=case,
            status=status,
            baseline_id=baseline.snapshot_id,
            baseline_hash=baseline.content_hash,
            scaffold_hash=h0,
            inv_id=snapshots["inv"].snapshot_id if "inv" in snapshots else None,
            dom_id=snapshots["dom"].snapshot_id if "dom" in snapshots else None,
            final_id=snapshots["final"].snapshot_id if "final" in snapshots else None,
            selected_snapshot_id=snapshots["final"].snapshot_id
            if status == "completed"
            else baseline.snapshot_id,
            trials=trials,
            stages=stages,
            reflection=reflection,
            reflection_cost=PPCModelCost.model_validate(
                reflection_response.model_dump(include=set(PPCModelCost.model_fields))
            ),
            reflection_error=reflection_response.error,
            total_cost=total_cost,
            errors=errors,
        )
        write_json_atomic(
            root / "reflection.json", reflection_response.model_dump(mode="json")
        )
        write_json_atomic(root / "ppc_task.json", result.model_dump(mode="json"))
        return result
