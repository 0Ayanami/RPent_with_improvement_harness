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

"""Serial, single-patch adaptation with an independent confirmation gate.

The phase separation follows Harness2's schedule/recursion/selection/library
at google-research/harness2, revision 1cfe254e8d25b15f37048c3730134686abe60335.
The editable surface, native feedback and confirmation gate are RPent adaptations.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

import yaml

from rpent.evaluation.result import write_json_atomic
from rpent.improvement.models import (
    ExecutionConfig,
    ExperiencePatch,
    ExperimentConfig,
    ImprovementExperience,
    QualificationReport,
    TaskCase,
    TrialResult,
    TrialSpec,
    fingerprint,
)
from rpent.memory.snapshots import MemorySnapshot, SnapshotStore
from rpent.utils.config import get_repo_root

if TYPE_CHECKING:
    from rpent.improvement.proposer import ProposalResult


class TrialRunner(Protocol):
    """A robot adapter owns the episode and drains it before returning."""

    def run(self, spec: TrialSpec, snapshot: MemorySnapshot) -> TrialResult:
        """Execute an independent episode using only the supplied corpus."""
        ...


class PatchProposer(Protocol):
    """Return a checked patch and the costs of every admitted model request."""

    def propose(
        self,
        *,
        baseline: MemorySnapshot,
        evidence: dict[str, Any],
        lessons: list[ImprovementExperience],
        rejection: str | None,
    ) -> ProposalResult:
        """Produce one proposal; the controller owns semantic retries."""
        ...


class _TrialStopped(RuntimeError):
    """The controller stopped admission after a recorded trial or budget gate."""


def load_config(path: Path | str) -> ExperimentConfig:
    """Read strict YAML/JSON configuration, resolving paths beside the config file."""
    path = Path(path).resolve()
    config = ExperimentConfig.model_validate(yaml.safe_load(path.read_text("utf-8")))
    updates = {}
    for field in ("output_dir", "baseline_memory", "patch_file", "library_dir"):
        value = getattr(config, field)
        if value is not None:
            updates[field] = (path.parent / value).resolve()
    return config.model_copy(update=updates)


def scaffold_fingerprint(config: ExperimentConfig | ExecutionConfig) -> str:
    """Fingerprint execution code, settings and relevant service configuration."""
    execution = config if isinstance(config, ExecutionConfig) else config.execution
    root = get_repo_root()
    files = {}
    for directory in (root / "rpent", root / "robots" / "libero"):
        for path in sorted(directory.rglob("*.py")):
            files[path.relative_to(root).as_posix()] = hashlib.sha256(
                path.read_bytes()
            ).hexdigest()
    for name in ("pyproject.toml",):
        if (root / name).is_file():
            files[name] = hashlib.sha256((root / name).read_bytes()).hexdigest()
    for path in sorted((root / "robots" / "libero" / "guides").glob("*.md")):
        files[path.relative_to(root).as_posix()] = hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
    # Model services cannot be introspected for checkpoint bytes; require the
    # explicit vla_identity as well as recording the resolved launch settings.
    environment = {
        name: os.environ.get(name)
        for name in (
            "PI05_CHECKPOINT_PATH",
            "LIBERO_TYPE",
            "RPENT_RLINF_ROOT",
            "RLINF_REPO_PATH",
        )
    }
    dependencies = {}
    for name in (
        "pydantic-ai-slim",
        "pydantic-ai-harness",
        "openai-codex",
        "numpy",
        "rlinf",
    ):
        try:
            dependencies[name] = version(name)
        except PackageNotFoundError:
            dependencies[name] = None
    return fingerprint(
        {
            "code": files,
            "execution": execution.model_dump(mode="json"),
            "environment": environment,
            "dependencies": dependencies,
        }
    )


def read_patch(path: Path | str) -> ExperiencePatch:
    """Parse a hand-authored patch using the same schema as automatic proposals."""
    return ExperiencePatch.model_validate(yaml.safe_load(Path(path).read_text("utf-8")))


def prepare(config: ExperimentConfig) -> tuple[SnapshotStore, MemorySnapshot]:
    """Materialize E0 without starting a planner, model or simulator."""
    output = config.output_dir.resolve()
    source = config.baseline_memory.resolve()
    if (
        output == source
        or output.is_relative_to(source)
        or source.is_relative_to(output)
    ):
        raise ValueError(
            "baseline_memory and experiment output must be separate directory trees"
        )
    output.mkdir(parents=True, exist_ok=True)
    store = SnapshotStore(output / "snapshots")
    return store, store.capture(source)


def check(config: ExperimentConfig) -> dict[str, Any]:
    """Validate a manual candidate and report frozen identities without robot I/O."""
    store, baseline = prepare(config)
    result = {
        "schema_version": 1,
        "baseline_id": baseline.snapshot_id,
        "baseline_hash": baseline.content_hash,
        "scaffold_hash": scaffold_fingerprint(config),
        "protocol_hash": fingerprint(config.protocol),
        "leaf_hashes": dict(baseline.files),
        "behavior_verified": False,
    }
    if config.patch_file:
        patch = read_patch(config.patch_file)
        if patch.base_snapshot_hash != baseline.content_hash:
            raise ValueError("patch base_snapshot_hash does not match E0")
        candidate = store.apply(baseline, patch.edit)
        result.update(
            candidate_id=candidate.snapshot_id, candidate_hash=candidate.content_hash
        )
    write_json_atomic(config.output_dir / "static_check.json", result)
    return result


def _valid(result: TrialResult) -> bool:
    return (
        result.outcome == "completed"
        and result.cleanup_ok
        and result.official_success is not None
        and result.initial_state_id is not None
    )


def _sum_costs(trials: list[TrialResult], proposals: list[dict]) -> dict[str, Any]:
    costs: dict[str, Any] = {}
    for field in (
        "wall_time_s",
        "planner_requests",
        "input_tokens",
        "output_tokens",
        "env_steps",
        "vla_calls",
        "resets",
    ):
        observed = [getattr(trial.cost, field) for trial in trials]
        known_subtotal = sum(value for value in observed if value is not None)
        costs[field] = known_subtotal
        if any(value is None for value in observed):
            costs[field] = None
            costs.setdefault("known_subtotals", {})[field] = known_subtotal
            costs.setdefault("incomplete_counters", []).append(field)
    requests = [reply["requests"] for reply in proposals]
    costs["improvement_requests"] = (
        sum(requests) if all(value is not None for value in requests) else None
    )
    if costs["improvement_requests"] is None:
        costs.setdefault("known_subtotals", {})["improvement_requests"] = sum(
            value for value in requests if value is not None
        )
        costs.setdefault("incomplete_counters", []).append("improvement_requests")
    costs["improvement_wall_time_s"] = sum(reply["elapsed_s"] for reply in proposals)
    for field in ("input_tokens", "output_tokens"):
        values = [reply[field] for reply in proposals]
        total = sum(values) if all(value is not None for value in values) else None
        costs[f"improvement_{field}"] = total
        if total is None:
            costs.setdefault("known_subtotals", {})[f"improvement_{field}"] = sum(
                value for value in values if value is not None
            )
        execution_total = costs[field]
        costs[f"all_model_{field}"] = (
            execution_total + total
            if execution_total is not None and total is not None
            else None
        )
        if costs[f"all_model_{field}"] is None:
            known = costs.get("known_subtotals", {})
            costs.setdefault("known_subtotals", {})[f"all_model_{field}"] = (
                execution_total if execution_total is not None else known[field]
            ) + (total if total is not None else known[f"improvement_{field}"])
    dollar_costs = [reply.get("cost_usd") for reply in proposals]
    costs["improvement_cost_usd"] = (
        sum(dollar_costs) if all(value is not None for value in dollar_costs) else None
    )
    if costs["improvement_cost_usd"] is None:
        costs.setdefault("known_subtotals", {})["improvement_cost_usd"] = sum(
            value for value in dollar_costs if value is not None
        )
    costs["trials"] = len(trials)
    costs["proposals"] = proposals
    return costs


class ImprovementHarness:
    """Freeze evidence libraries, propose one edit, probe and independently confirm."""

    def __init__(
        self,
        config: ExperimentConfig,
        runner: TrialRunner,
        proposer: PatchProposer | None = None,
    ) -> None:
        self.config = ExperimentConfig.model_validate(config.model_dump())
        self.runner = runner
        self.proposer = proposer
        self.trials: list[TrialResult] = []
        self.proposals: list[dict] = []
        self.store: SnapshotStore | None = None
        self.baseline: MemorySnapshot | None = None
        self.candidate: MemorySnapshot | None = None
        self.patch: ExperiencePatch | None = None
        self.scaffold_hash = scaffold_fingerprint(config)
        self.protocol_hash = fingerprint(config.protocol)
        self.started = 0.0

    def _trial(
        self, role: str, case: TaskCase, snapshot: MemorySnapshot
    ) -> TrialResult:
        if scaffold_fingerprint(self.config) != self.scaffold_hash:
            raise _TrialStopped("execution scaffold changed during adaptation")
        if fingerprint(self.config.protocol) != self.protocol_hash:
            raise _TrialStopped("qualification protocol changed during adaptation")
        if len(self.trials) >= self.config.protocol.max_trials:
            raise _TrialStopped("adaptation trial budget exhausted")
        self.store.verify(snapshot)
        trial_id = f"{role}_{case.case_id}"
        directory = self.config.output_dir / "trials" / trial_id
        if directory.exists():
            raise ValueError(
                f"trial directory already exists: {directory}; use a fresh experiment output"
            )
        directory.mkdir(parents=True)
        spec = TrialSpec(
            trial_id=trial_id,
            experiment_id=self.config.experiment_id,
            case=case,
            snapshot_id=snapshot.snapshot_id,
            snapshot_hash=snapshot.content_hash,
            scaffold_hash=self.scaffold_hash,
            budget=self.config.execution.budget,
            output_dir=directory,
        )
        write_json_atomic(directory / "trial_spec.json", spec.model_dump(mode="json"))
        self._event(directory, "running")
        try:
            result = self.runner.run(spec, snapshot)
            if result.spec != spec:
                raise ValueError("runner returned results for different trial inputs")
            try:
                self.store.verify(snapshot)
            except (ValueError, OSError) as exc:
                result = result.model_copy(
                    update={
                        "outcome": "infra_error",
                        "error": f"snapshot integrity check failed: {exc}",
                    }
                )
        except BaseException:
            # Never manufacture a finalized trial after an unaccounted crash.
            self._event(directory, "interrupted", cost="unknown")
            raise
        write_json_atomic(
            directory / "trial_result.json", result.model_dump(mode="json")
        )
        self._event(directory, "finalized", cost=result.cost.model_dump(mode="json"))
        self.trials.append(result)
        if not result.cleanup_ok:
            raise _TrialStopped(
                "trial cleanup failed; refusing to start another episode"
            )
        return result

    @staticmethod
    def _event(directory: Path, state: str, **fields: Any) -> None:
        with (directory / "trial_events.jsonl").open("a", encoding="utf-8") as file:
            file.write(json.dumps({"run_state": state, **fields}) + "\n")

    def _read_candidate(self, result: TrialResult) -> bool:
        if not self.config.protocol.require_candidate_read:
            return True
        path = (
            "MEMORY.md"
            if self.patch.edit.operation == "retire"
            else self.patch.edit.path
        )
        expected = self.candidate.files.get(path)
        return any(
            event.get("relative_path") == path
            and event.get("source_hash") == expected
            and event.get("provided_chars", 0) > 0
            and event.get("truncated") is False
            for event in result.memory_reads
        )

    def _pair_problem(self, base: TrialResult, candidate: TrialResult) -> str | None:
        if not _valid(base) or not _valid(candidate):
            return "paired trial has an unavailable verdict, budget failure or infrastructure error"
        if base.initial_state_id != candidate.initial_state_id:
            return "baseline and candidate did not resolve to the same native initial state"
        if not self._read_candidate(candidate):
            return (
                "candidate edit was not observed through the audited memory read path"
            )
        if base.official_success and not candidate.official_success:
            return "candidate regressed on a matched baseline success"
        return None

    def _report(self, decision: str, reasons: list[str]) -> QualificationReport:
        if scaffold_fingerprint(self.config) != self.scaffold_hash:
            decision = "invalid"
            reasons = [*reasons, "execution scaffold changed during adaptation"]
        if fingerprint(self.config.protocol) != self.protocol_hash:
            decision = "invalid"
            reasons = [*reasons, "qualification protocol changed during adaptation"]
        report = QualificationReport(
            experiment_id=self.config.experiment_id,
            protocol_hash=self.protocol_hash,
            scaffold_hash=self.scaffold_hash,
            baseline_id=self.baseline.snapshot_id,
            baseline_hash=self.baseline.content_hash,
            candidate_id=self.candidate.snapshot_id if self.candidate else None,
            candidate_hash=self.candidate.content_hash if self.candidate else None,
            patch_id=self.patch.patch_id if self.patch else None,
            decision=decision,
            reasons=reasons,
            trial_ids=[trial.spec.trial_id for trial in self.trials],
            scope=self.config.protocol.confirmation_cases
            + self.config.protocol.regression_cases,
            total_cost={
                **_sum_costs(self.trials, self.proposals),
                "adaptation_wall_time_s": time.monotonic() - self.started,
            },
            limitations=[
                "Native-success-assisted, single-patch adaptation; not the full Harness2 PPC protocol.",
                "Matched initial states do not imply deterministic VLA execution.",
                "Qualification covers the configured LIBERO task identities and execution budget only; statistical benefit is unverified.",
                "Remote service checkpoint identity is operator supplied, not verified from weights.",
            ],
        )
        write_json_atomic(
            self.config.output_dir / "qualification.json",
            report.model_dump(mode="json"),
        )
        self._record_experience(report)
        return report

    def _record_experience(self, report: QualificationReport) -> None:
        if not self.config.library_dir or not self.patch or not self.trials:
            return
        from rpent.improvement.library import ExperienceLibrary

        case = self.config.protocol.probe_cases[0]
        # Store observations rather than converting a model's causal hypothesis
        # into a fact. Rejected and inconclusive attempts remain useful evidence.
        observations = [
            f"{trial.spec.trial_id}: native={trial.official_success}, outcome={trial.outcome}"
            for trial in self.trials
        ]
        lesson = ImprovementExperience(
            experience_id=f"{self.config.experiment_id}_observation",
            suite=case.suite,
            task_id=case.task_id,
            cue=self.patch.expected_effect,
            lesson=f"Edit {self.patch.edit.operation} {self.patch.edit.path}; decision={report.decision}. "
            + "; ".join(observations)
            + ". Causal explanation and transfer remain unverified.",
            evidence_refs=report.trial_ids,
            counterevidence=report.reasons,
            decision=report.decision,
            report_hash=fingerprint(report),
            patch=self.patch,
        )
        ExperienceLibrary(self.config.library_dir).append(lesson, report=report)

    def _propose(
        self, references: list[TrialResult], lessons: list[ImprovementExperience]
    ) -> ExperiencePatch | None:
        if self.config.patch_file:
            return read_patch(self.config.patch_file)
        if self.proposer is None:
            from rpent.improvement.proposer import APIProposer

            self.proposer = APIProposer(
                self.config.proposer_model, self.config.proposer_base_url
            )
        rejection = None
        evidence = {
            "trials": [trial.model_dump(mode="json") for trial in references],
            "editable_surface": "One global or task-family memory leaf; task-specific artifacts and execution code are frozen.",
        }
        for index in range(self.config.protocol.max_proposals):
            reply = self.proposer.propose(
                baseline=self.baseline,
                evidence=evidence,
                lessons=lessons,
                rejection=rejection,
            )
            details = reply.model_dump(mode="json")
            self.proposals.append(details)
            write_json_atomic(
                self.config.output_dir / f"proposal_{index + 1}.json", details
            )
            if reply.patch is None:
                rejection = reply.error or "no valid patch returned"
                continue
            try:
                self._check_patch(reply.patch)
            except (ValueError, OSError) as exc:
                rejection = str(exc)
                continue
            return reply.patch
        return None

    def _check_patch(self, patch: ExperiencePatch) -> None:
        if patch.base_snapshot_hash != self.baseline.content_hash:
            raise ValueError("patch base_snapshot_hash does not match frozen E0")
        known = {trial.spec.trial_id for trial in self.trials}
        for ref in patch.evidence_refs:
            trial_id = ref.removeprefix("trials/").split("/")[0]
            if trial_id not in known:
                raise ValueError(
                    f"patch evidence does not reference an observed trial: {ref}"
                )
        self.store.apply(self.baseline, patch.edit)

    def run(self) -> QualificationReport:
        """Run the finite protocol, falling back to E0 on unsupported evidence."""
        self.started = time.monotonic()
        self.store, self.baseline = prepare(self.config)
        manifest = self.config.output_dir / "experiment.json"
        if manifest.exists():
            raise ValueError(
                "experiment already started; use a fresh output directory (interrupted trials are retained)"
            )
        try:
            return self._run(manifest)
        except _TrialStopped as exc:
            return self._report("insufficient_evidence", [str(exc)])

    def _run(self, manifest: Path) -> QualificationReport:
        from rpent.improvement.library import ExperienceLibrary

        library = (
            ExperienceLibrary(self.config.library_dir)
            if self.config.library_dir
            else None
        )
        frozen = library.freeze() if library else None
        case = self.config.protocol.probe_cases[0]
        lessons = (
            frozen.retrieve(
                case,
                query=case.suite,
                char_budget=self.config.protocol.meta_char_budget,
            )
            if frozen
            else []
        )
        write_json_atomic(
            manifest,
            {
                "schema_version": 1,
                "config": self.config.model_dump(mode="json"),
                "config_hash": fingerprint(self.config),
                "scaffold_hash": self.scaffold_hash,
                "baseline_id": self.baseline.snapshot_id,
                "baseline_hash": self.baseline.content_hash,
                "meta_snapshot_hash": frozen.content_hash if frozen else None,
                "meta_input": [lesson.model_dump(mode="json") for lesson in lessons],
            },
        )
        # Reserve complete paired confirmation/regression episodes before any
        # model or environment calls, rather than spending their budget on probes.
        protocol = self.config.protocol
        required = 2 * (
            len(protocol.probe_cases)
            + len(protocol.confirmation_cases)
            + len(protocol.regression_cases)
        )
        if protocol.max_trials < required:
            return self._report(
                "insufficient_evidence",
                ["trial budget cannot cover the predeclared paired protocol"],
            )
        references = [
            self._trial("reference", case, self.baseline)
            for case in protocol.probe_cases
        ]
        if not all(_valid(trial) for trial in references):
            return self._report(
                "insufficient_evidence",
                ["reference execution did not produce complete native evidence"],
            )
        if len({trial.initial_state_id for trial in references}) != len(references):
            return self._report(
                "insufficient_evidence",
                ["probe seeds alias the same native initial state"],
            )
        try:
            self.patch = self._propose(references, lessons)
            if self.patch is None:
                return self._report(
                    "invalid",
                    ["bounded proposal attempts produced no valid memory patch"],
                )
            self._check_patch(self.patch)
            self.candidate = self.store.apply(self.baseline, self.patch.edit)
        except (ValueError, OSError) as exc:
            return self._report("invalid", [str(exc)])
        write_json_atomic(
            self.config.output_dir / "patch.json", self.patch.model_dump(mode="json")
        )
        probes = [
            self._trial("probe", case, self.candidate) for case in protocol.probe_cases
        ]
        problems = [
            self._pair_problem(base, candidate)
            for base, candidate in zip(references, probes)
        ]
        problems = [problem for problem in problems if problem]
        if problems:
            decision = (
                "rejected"
                if any("regressed" in problem for problem in problems)
                else "insufficient_evidence"
            )
            return self._report(decision, problems)
        if not any(
            candidate.official_success and not base.official_success
            for base, candidate in zip(references, probes)
        ):
            return self._report(
                "rejected", ["matched probes showed no native-success improvement"]
            )
        seen = {trial.initial_state_id for trial in references}
        for role, cases in (
            ("confirmation", protocol.confirmation_cases),
            ("regression", protocol.regression_cases),
        ):
            for case in cases:
                base = self._trial(f"{role}_base", case, self.baseline)
                candidate = self._trial(role, case, self.candidate)
                problem = self._pair_problem(base, candidate)
                if problem:
                    decision = (
                        "rejected"
                        if "regressed" in problem
                        else "insufficient_evidence"
                    )
                    return self._report(decision, [problem])
                if base.initial_state_id in seen:
                    return self._report(
                        "insufficient_evidence",
                        [
                            "requested seeds alias a previously used native initial state"
                        ],
                    )
                seen.add(base.initial_state_id)
                if not candidate.official_success:
                    return self._report(
                        "rejected", [f"candidate failed independent {role}"]
                    )
        return self._report(
            "accepted",
            [
                "native-success improvement in probes, independent confirmation and configured regressions passed"
            ],
        )
