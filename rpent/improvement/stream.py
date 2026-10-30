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

"""Construct memory on seed zero, then evaluate independently frozen corpora.

The paper split is over initial states of the same tasks, not unseen task IDs.
Evaluation never imports the editor or reads the construction meta library.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import yaml
from pydantic import Field, model_validator

from rpent.evaluation.result import write_json_atomic
from rpent.improvement.controller import scaffold_fingerprint
from rpent.improvement.models import (
    ExecutionConfig,
    Record,
    TaskCase,
    TrialResult,
    TrialSpec,
    fingerprint,
    initial_state_key,
)
from rpent.memory.snapshots import MemorySnapshot, SnapshotStore
from rpent.utils.logging import get_logger

if TYPE_CHECKING:
    from rpent.improvement.controller import TrialRunner
    from rpent.improvement.ppc import PPCController

logger = get_logger("improvement.stream")

STANDARD_SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")
PRO_SUITES = tuple(
    f"{suite}_{kind}" for suite in STANDARD_SUITES for kind in ("task", "swap")
)


class EditorConfig(Record):
    """Explicit editor request settings, separate from execution costs."""

    planner: Literal["api", "codex"] = "api"
    model: str = Field(min_length=1)
    base_url: str | None = None
    max_tokens: int = Field(default=8192, gt=0)
    timeout_s: int = Field(default=120, gt=0)
    reasoning_effort: Literal["none", "low", "medium", "high", "xhigh"] = "none"

    @model_validator(mode="after")
    def validate_codex_endpoint(self) -> EditorConfig:
        """Use the same local Codex provider configuration for all roles."""
        if self.planner == "codex" and self.base_url is not None:
            raise ValueError("Codex uses CODEX_BASE_URL; omit editor.base_url")
        return self


class MemorySource(Record):
    """Local corpus and an operator disclosure of its actual generation history."""

    path: Path
    provenance: str = Field(min_length=1)


class StreamConfig(Record):
    """Predeclared construction order and paired, frozen-memory evaluation."""

    schema_version: Literal[2] = 2
    experiment_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_-]*$")
    output_dir: Path
    benchmark: Literal["libero", "libero_pro"] = "libero"
    task_ids: list[int] = Field(default_factory=lambda: list(range(10)), min_length=1)
    evaluation_seeds: list[int] = Field(
        default_factory=lambda: list(range(1, 11)), min_length=1
    )
    initial_memory: MemorySource
    reference_memory: MemorySource
    execution: ExecutionConfig
    evaluation_execution: ExecutionConfig
    editor: EditorConfig
    use_meta_library: bool = True
    meta_char_budget: int = Field(default=12000, ge=0)
    k: Literal[1] = 1
    paper_version: Literal["2607.08448v5"] = "2607.08448v5"

    @model_validator(mode="after")
    def validate_protocol(self) -> StreamConfig:
        if len(set(self.task_ids)) != len(self.task_ids) or any(
            not 0 <= task <= 9 for task in self.task_ids
        ):
            raise ValueError("task_ids must be distinct LIBERO task IDs in [0, 9]")
        if len(set(self.evaluation_seeds)) != len(self.evaluation_seeds) or any(
            seed <= 0 for seed in self.evaluation_seeds
        ):
            raise ValueError(
                "evaluation seeds must be distinct and exclude construction seed 0"
            )
        if self.execution.model_dump(
            exclude={"budget"}
        ) != self.evaluation_execution.model_dump(exclude={"budget"}):
            raise ValueError(
                "construction and evaluation execution may differ only in episode budget"
            )
        return self

    @property
    def suites(self) -> tuple[str, ...]:
        """Return the fixed suite order for the chosen benchmark."""
        return STANDARD_SUITES if self.benchmark == "libero" else PRO_SUITES

    def cases(self, *, construction: bool) -> list[TaskCase]:
        """Materialize requested cases without loading simulator dependencies."""
        return [
            TaskCase(
                case_id=f"{suite}_t{task}_s{seed}", suite=suite, task_id=task, seed=seed
            )
            for suite in self.suites
            for task in self.task_ids
            for seed in ([0] if construction else self.evaluation_seeds)
        ]


class FrozenMemory(Record):
    """Complete input to evaluation, independent of all mutable construction state."""

    schema_version: Literal[2] = 2
    arm: Literal["reference", "ppc_meta", "ppc_no_meta"]
    experiment_id: str
    comparison_key: str
    protocol_hash: str
    snapshot_store: Path
    snapshot_id: str
    snapshot_hash: str
    construction_states: dict[str, str]
    evaluation_cases: list[TaskCase]
    execution: ExecutionConfig
    scaffold_hash: str
    initial_memory_hash: str
    source_provenance: str
    meta_hash: str | None = None
    limitations: list[str]

    @model_validator(mode="after")
    def validate_split(self) -> FrozenMemory:
        cases = self.evaluation_cases
        keys = [(case.suite, case.task_id, case.seed) for case in cases]
        if (
            not cases
            or len(set(keys)) != len(cases)
            or len({case.case_id for case in cases}) != len(cases)
        ):
            raise ValueError("frozen evaluation cases must be nonempty and distinct")
        if any(case.seed == 0 for case in cases) or not self.construction_states:
            raise ValueError(
                "frozen evaluation excludes seed zero and requires construction state identities"
            )
        return self


def load_stream_config(path: Path | str) -> StreamConfig:
    """Read strict YAML, with paths relative to the configuration file."""
    path = Path(path).resolve()
    config = StreamConfig.model_validate(yaml.safe_load(path.read_text("utf-8")))
    return config.model_copy(
        update={
            "output_dir": (path.parent / config.output_dir).resolve(),
            **{
                field: getattr(config, field).model_copy(
                    update={
                        "path": (path.parent / getattr(config, field).path).resolve()
                    }
                )
                for field in ("initial_memory", "reference_memory")
            },
        }
    )


def _write_sealed(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(path, {"content_hash": fingerprint(payload), "payload": payload})


def _read_sealed(path: Path | str) -> dict:
    record = json.loads(Path(path).read_text("utf-8"))
    if (
        set(record) != {"content_hash", "payload"}
        or fingerprint(record["payload"]) != record["content_hash"]
    ):
        raise ValueError(f"manifest integrity check failed: {path}")
    return record["payload"]


def _disjoint_paths(*paths: Path) -> None:
    resolved = [path.resolve() for path in paths]
    for index, path in enumerate(resolved):
        if any(
            path.is_relative_to(other) or other.is_relative_to(path)
            for other in resolved[:index]
        ):
            raise ValueError(
                "output and input memory must occupy separate directory trees"
            )


def plan(config: StreamConfig) -> dict[str, Any]:
    """Capture local inputs and write an offline manifest; never start a model."""
    _disjoint_paths(config.output_dir, config.initial_memory.path)
    _disjoint_paths(config.output_dir, config.reference_memory.path)
    store = SnapshotStore(config.output_dir / "snapshots")
    initial = store.capture(config.initial_memory.path)
    reference = store.capture(config.reference_memory.path)
    from robots.libero.improvement import _runtime_args

    for execution in (config.execution, config.evaluation_execution):
        _runtime_args(
            execution,
            TrialSpec(
                trial_id="static",
                experiment_id=config.experiment_id,
                case=config.cases(construction=True)[0],
                snapshot_id=initial.snapshot_id,
                snapshot_hash=initial.content_hash,
                scaffold_hash="static",
                budget=execution.budget,
                output_dir=config.output_dir / "static-unused",
            ),
            initial,
        )
    construction = config.cases(construction=True)
    evaluation = config.cases(construction=False)
    payload = {
        "config": config.model_dump(mode="json"),
        "protocol_hash": fingerprint(config),
        "initial_id": initial.snapshot_id,
        "initial_hash": initial.content_hash,
        "reference_id": reference.snapshot_id,
        "reference_hash": reference.content_hash,
        "construction_scaffold": scaffold_fingerprint(config.execution),
        "evaluation_scaffold": scaffold_fingerprint(config.evaluation_execution),
        "construction_cases": [case.model_dump(mode="json") for case in construction],
        "evaluation_cases": [case.model_dump(mode="json") for case in evaluation],
        "expected_rollouts": 4 * len(construction),
        "full_paper_case_table": config.task_ids == list(range(10))
        and config.evaluation_seeds == list(range(1, 11)),
        "behavior_verified": False,
        "unresolved": [
            "native initial-state identities require server validation",
            "operator-supplied VLA/checkpoint and memory provenance require verification",
        ],
    }
    destination = config.output_dir / "plan.json"
    if destination.exists() and _read_sealed(destination) != payload:
        raise ValueError(
            "plan differs from this output directory; choose a fresh output_dir"
        )
    _write_sealed(destination, payload)
    return payload


def _task_records(result: TrialResult) -> dict[str, bytes]:
    """Import only the selected episode's native audit and exported command pair."""
    if result.official_success is not True:
        return {}
    directory = result.spec.output_dir.resolve()
    case = result.spec.case
    tag = f"{case.suite.replace('libero_', '')}_t{case.task_id}_s{case.seed}"
    records = {}
    for key, name in (
        ("task_audit", f"{tag}.json"),
        ("task_recipe", f"{tag}_recipe.jsonl"),
    ):
        if key not in result.artifacts:
            raise ValueError(
                "successful selected trial did not export a native audit/recipe pair"
            )
        path = Path(result.artifacts[key]).resolve()
        if path != directory / name or path.is_symlink():
            raise ValueError(
                "task record must be the selected trial's own native artifact"
            )
        records[f"task-specific/{name}"] = path.read_bytes()
    audit = json.loads(records[f"task-specific/{tag}.json"])
    if any(
        audit.get(key) != value
        for key, value in {
            "suite": case.suite,
            "task_id": case.task_id,
            "seed": case.seed,
            "libero_terminated": True,
            "source_trial": result.spec.trial_id,
            "initial_state_id": result.initial_state_id,
        }.items()
    ):
        raise ValueError("task audit differs from the native selected trial")
    commands = [
        json.loads(line)
        for line in records[f"task-specific/{tag}_recipe.jsonl"]
        .decode("utf-8")
        .splitlines()
        if line.strip()
    ]
    if not commands or any(
        not isinstance(command, dict)
        or "action" not in command
        or command["action"] == "reset"
        for command in commands
    ):
        raise ValueError("recipe must contain recorded non-reset actions")
    return records


def _aggregate_costs(rows: list[dict]) -> dict[str, Any]:
    totals: dict[str, Any] = {}
    known = {}
    for key in sorted(
        {
            key
            for row in rows
            for key, value in row.items()
            if isinstance(value, (int, float)) or value is None
        }
    ):
        values = [row.get(key) for row in rows]
        subtotal = sum(value for value in values if value is not None)
        totals[key] = subtotal if all(value is not None for value in values) else None
        if totals[key] is None:
            known[key] = sum(
                row.get("known_subtotals", {}).get(key, row.get(key) or 0)
                for row in rows
            )
    if known:
        totals["known_subtotals"] = known
    return totals


def _require_concrete_sources(config: StreamConfig) -> None:
    for name, value in (
        ("vla_identity", config.execution.vla_identity),
        ("reference provenance", config.reference_memory.provenance),
    ):
        if (
            "运行前填写" in value
            or "REPLACE" in value.upper()
            or "TODO" in value.upper()
        ):
            raise ValueError(f"replace the template {name} before server execution")


def construct(config: StreamConfig, controller: PPCController) -> dict[str, Any]:
    """Construct memory, preserving an explicit failure record on interruption."""
    _require_concrete_sources(config)
    if (config.output_dir / "construction").exists():
        raise ValueError(
            "construction output already exists; automatic resume is unsupported"
        )
    try:
        return _construct(config, controller)
    except BaseException as exc:
        root = config.output_dir / "construction"
        if root.exists():
            completed = [
                _read_sealed(path) for path in sorted(root.glob("*/task_report.json"))
            ]
            _write_sealed(
                config.output_dir / "construction_report.json",
                {
                    "status": "interrupted"
                    if isinstance(exc, (KeyboardInterrupt, InterruptedError))
                    else "invalid",
                    "error": f"{type(exc).__name__}: {exc}",
                    "recorded_tasks": len(completed),
                    "cost": _aggregate_costs(
                        [report["total_cost"] for report in completed]
                    ),
                    "in_flight_cost": "unknown; inspect persisted trial events and stage requests",
                    "frozen": False,
                },
            )
        raise


def _construct(config: StreamConfig, controller: PPCController) -> dict[str, Any]:
    """Run the ordered seed-zero stream and seal candidate/reference manifests."""
    from rpent.improvement.ppc_models import PPCLesson

    inputs = plan(config)
    root = config.output_dir / "construction"
    if root.exists():
        raise ValueError(
            "construction output already exists; automatic resume is unsupported"
        )
    root.mkdir(parents=True)
    store = controller.snapshot_store
    if store.root.resolve() != (config.output_dir / "snapshots").resolve():
        raise ValueError("controller snapshot store differs from the planned store")
    current = store.load(inputs["initial_id"])
    lessons: list[dict] = []
    task_reports: list[dict] = []
    states: dict[str, str] = {}
    scaffold = inputs["construction_scaffold"]
    for case in config.cases(construction=True):
        if scaffold_fingerprint(config.execution) != scaffold:
            raise ValueError("construction scaffold changed after planning")
        # Freeze the prefix by serialization, not a reference to a mutable list.
        prefix = json.loads(json.dumps(lessons)) if config.use_meta_library else []
        selected_lessons = []
        remaining = config.meta_char_budget
        ranked = sorted(
            enumerate(prefix),
            key=lambda item: (item[1]["suite"] != case.suite, -item[0]),
        )
        for _index, entry in ranked:
            size = len(json.dumps(entry["lesson"], ensure_ascii=False))
            if size <= remaining:
                selected_lessons.append(PPCLesson.model_validate(entry["lesson"]))
                remaining -= size
        case_root = root / case.case_id
        case_root.mkdir()
        _write_sealed(
            case_root / "meta_prefix.json",
            {
                "entries": prefix,
                "selected": [
                    lesson.model_dump(mode="json") for lesson in selected_lessons
                ],
            },
        )
        result = controller.run(
            case,
            current,
            case_root / "ppc",
            config.execution.budget,
            meta_lessons=selected_lessons,
            scaffold_fingerprint=lambda: scaffold_fingerprint(config.execution),
        )
        report = result.model_dump(mode="json")
        _write_sealed(case_root / "task_report.json", report)
        task_reports.append(report)
        if result.status != "completed":
            raise ValueError(
                f"PPC task did not complete: {case.case_id}; see {case_root}"
            )
        identities = {trial.initial_state_id for trial in result.trials.values()}
        if len(identities) != 1 or None in identities:
            raise ValueError(
                "all four task rollouts must resolve the same initial state"
            )
        state = identities.pop()
        if initial_state_key(state) in {
            initial_state_key(value) for value in states.values()
        }:
            raise ValueError("construction cases alias the same native state")
        states[case.case_id] = state
        current = store.load(result.selected_snapshot_id)
        records = _task_records(result.trials["final"])
        if records:
            current = store.publish_task_records(current, records)
        # Reflection is persisted for both arms; only full recursion retrieves it.
        if (
            config.use_meta_library
            and result.reflection is not None
            and result.reflection.lesson is not None
        ):
            entry = {
                "suite": case.suite,
                "task_id": case.task_id,
                "task_report_hash": fingerprint(report),
                "selected_snapshot_id": current.snapshot_id,
                "lesson": result.reflection.lesson.model_dump(mode="json"),
            }
            lessons.append(entry)
            _write_sealed(root / "meta" / f"{case.case_id}.json", entry)
        _write_sealed(
            root / "prefix.json",
            {
                "last_case": case.case_id,
                "snapshot_id": current.snapshot_id,
                "entries": lessons,
                "construction_states": states,
            },
        )
        logger.info("Constructed %s; memory=%s", case.case_id, current.content_hash)
    if scaffold_fingerprint(config.execution) != scaffold:
        raise ValueError("construction scaffold changed before freeze")
    store.verify(current)
    reference = store.load(inputs["reference_id"])
    comparison_key = fingerprint(
        {
            "cases": inputs["evaluation_cases"],
            "execution": config.evaluation_execution.model_dump(mode="json"),
            "scaffold": inputs["evaluation_scaffold"],
            "initial": inputs["initial_hash"],
            "reference": inputs["reference_hash"],
        }
    )
    common = {
        "experiment_id": config.experiment_id,
        "comparison_key": comparison_key,
        "protocol_hash": inputs["protocol_hash"],
        "snapshot_store": store.root.resolve(),
        "construction_states": states,
        "evaluation_cases": config.cases(construction=False),
        "execution": config.evaluation_execution,
        "scaffold_hash": inputs["evaluation_scaffold"],
        "initial_memory_hash": inputs["initial_hash"],
        "limitations": [
            "The current SDK/tool executor differs from the historical file-mediated paper runtime; task/seed/permissions are aligned separately.",
            "Historical reference corpus may predate the scene-seed fix; provenance is disclosed, not retroactively validated.",
            "A composed memory is adopted at k=1 without a native-success selector; causal benefit requires paired frozen evaluation.",
            "Unspecified paper budgets are operator settings, not claimed paper constants.",
        ],
    }
    for arm, snapshot, provenance, meta_hash in (
        ("reference", reference, config.reference_memory.provenance, None),
        (
            "ppc_meta" if config.use_meta_library else "ppc_no_meta",
            current,
            config.initial_memory.provenance,
            fingerprint(lessons),
        ),
    ):
        manifest = FrozenMemory(
            arm=arm,
            snapshot_id=snapshot.snapshot_id,
            snapshot_hash=snapshot.content_hash,
            source_provenance=provenance,
            meta_hash=meta_hash,
            **common,
        )
        _write_sealed(
            config.output_dir / f"frozen-{arm}.json", manifest.model_dump(mode="json")
        )
    summary = {
        "status": "frozen",
        "tasks": len(task_reports),
        "rollouts": sum(len(report["trials"]) for report in task_reports),
        "snapshot_id": current.snapshot_id,
        "snapshot_hash": current.content_hash,
        "meta_hash": fingerprint(lessons),
        "construction_states": states,
        "protocol_hash": inputs["protocol_hash"],
        "cost": _aggregate_costs([report["total_cost"] for report in task_reports]),
    }
    _write_sealed(config.output_dir / "construction_report.json", summary)
    return summary


def load_frozen(path: Path | str) -> tuple[FrozenMemory, SnapshotStore, MemorySnapshot]:
    """Verify a sealed manifest and the entire selected execution corpus."""
    manifest = FrozenMemory.model_validate(_read_sealed(path))
    store = SnapshotStore(manifest.snapshot_store)
    snapshot = store.load(manifest.snapshot_id)
    if snapshot.content_hash != manifest.snapshot_hash:
        raise ValueError("frozen memory hash differs from its manifest")
    store.verify(snapshot)
    return manifest, store, snapshot


def evaluate(
    frozen_path: Path | str, output_dir: Path | str, runner: TrialRunner | None = None
) -> dict[str, Any]:
    """Evaluate a sealed corpus; no proposer, reflection, merge or meta retrieval."""
    manifest, store, snapshot = load_frozen(frozen_path)
    output = Path(output_dir).resolve()
    _disjoint_paths(output, store.root)
    if output.exists():
        raise ValueError("evaluation requires a fresh output directory")
    if scaffold_fingerprint(manifest.execution) != manifest.scaffold_hash:
        raise ValueError("execution scaffold differs from the frozen manifest")
    forbidden = {
        initial_state_key(value) for value in manifest.construction_states.values()
    }
    if runner is None:
        from robots.libero.improvement import build_trial_runner

        runner = build_trial_runner(
            manifest.execution, forbidden_initial_states=forbidden
        )
    output.mkdir(parents=True)
    _write_sealed(output / "frozen_input.json", manifest.model_dump(mode="json"))
    results: list[TrialResult] = []
    invalid: str | None = None
    try:
        for case in manifest.evaluation_cases:
            if scaffold_fingerprint(manifest.execution) != manifest.scaffold_hash:
                raise ValueError("evaluation scaffold changed")
            store.verify(snapshot)
            spec = TrialSpec(
                trial_id=case.case_id,
                experiment_id=manifest.experiment_id,
                case=case,
                snapshot_id=snapshot.snapshot_id,
                snapshot_hash=snapshot.content_hash,
                scaffold_hash=manifest.scaffold_hash,
                budget=manifest.execution.budget,
                output_dir=output / case.case_id,
            )
            spec.output_dir.mkdir()
            write_json_atomic(
                spec.output_dir / "trial_spec.json", spec.model_dump(mode="json")
            )
            result = runner.run(spec, snapshot)
            write_json_atomic(
                spec.output_dir / "trial_result.json", result.model_dump(mode="json")
            )
            if result.spec != spec:
                raise ValueError("runner returned a different trial specification")
            results.append(result)
            if (
                result.initial_state_id is None
                or initial_state_key(result.initial_state_id) in forbidden
            ):
                raise ValueError(
                    "evaluation initial state is missing or overlaps construction/another evaluation case"
                )
            forbidden.add(initial_state_key(result.initial_state_id))
            if (
                not result.cleanup_ok
                or result.outcome in {"infra_error", "cancelled"}
                or result.official_success is None
            ):
                raise ValueError(f"invalid evaluation episode: {case.case_id}")
            store.verify(snapshot)
        if scaffold_fingerprint(manifest.execution) != manifest.scaffold_hash:
            raise ValueError("evaluation scaffold changed before reporting")
        store.verify(snapshot)
    except (ValueError, OSError, RuntimeError) as exc:
        invalid = str(exc)
    suites = {}
    for case in manifest.evaluation_cases:
        suites.setdefault(case.suite, {"requested": 0, "completed": 0, "successes": 0})[
            "requested"
        ] += 1
    for result in results:
        row = suites[result.spec.case.suite]
        row["completed"] += 1
        row["successes"] += int(
            result.official_success is True
            and result.cleanup_ok
            and result.outcome == "completed"
        )
    for row in suites.values():
        row["success_rate"] = row["successes"] / row["requested"]
    summary = {
        "status": "invalid" if invalid else "completed",
        "error": invalid,
        "arm": manifest.arm,
        "comparison_key": manifest.comparison_key,
        "snapshot_hash": snapshot.content_hash,
        "protocol_hash": manifest.protocol_hash,
        "requested": len(manifest.evaluation_cases),
        "completed": len(results),
        "successes": sum(row["successes"] for row in suites.values()),
        "suites": suites,
        "cases": [
            {
                "case": result.spec.case.model_dump(mode="json"),
                "initial_state_id": result.initial_state_id,
                "success": result.official_success,
                "outcome": result.outcome,
                "cleanup_ok": result.cleanup_ok,
                "cost": result.cost.model_dump(mode="json"),
            }
            for result in results
        ],
        "limitations": manifest.limitations,
        "cost": _aggregate_costs(
            [
                result.cost.model_dump(mode="json", exclude={"unknown"})
                for result in results
            ]
        ),
    }
    summary["success_rate"] = summary["successes"] / summary["requested"]
    _write_sealed(output / "evaluation_report.json", summary)
    return summary


def compare(reports: list[Path | str], output: Path | str) -> dict[str, Any]:
    """Compare completed arms only when requested cases and native layouts match."""
    rows = [_read_sealed(path) for path in reports]
    if len(rows) < 2 or any(row["status"] != "completed" for row in rows):
        raise ValueError(
            "comparison needs at least two complete valid evaluation reports"
        )
    if len({row["comparison_key"] for row in rows}) != 1:
        raise ValueError("evaluation conditions differ across arms")
    identities = [
        {fingerprint(case["case"]): case["initial_state_id"] for case in row["cases"]}
        for row in rows
    ]
    if any(identities[0] != identity for identity in identities[1:]):
        raise ValueError("native initial states differ across arms")
    references = [row for row in rows if row["arm"] == "reference"]
    if len(references) != 1 or len({row["arm"] for row in rows}) != len(rows):
        raise ValueError("provide one reference report and distinct candidate arms")
    reference = references[0]
    result = {
        "comparison_key": reference["comparison_key"],
        "requested_per_arm": reference["requested"],
        "arms": [
            {
                "arm": row["arm"],
                "snapshot_hash": row["snapshot_hash"],
                "successes": row["successes"],
                "success_rate": row["success_rate"],
                "delta_percentage_points": 100
                * (row["success_rate"] - reference["success_rate"]),
                "suites": row["suites"],
            }
            for row in rows
        ],
    }
    _write_sealed(Path(output), result)
    return result
