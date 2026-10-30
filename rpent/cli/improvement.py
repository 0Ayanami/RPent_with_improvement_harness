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

"""Validate, run and deploy the memory-only LIBERO improvement protocol."""

from __future__ import annotations

import argparse
from pathlib import Path

from pydantic import ValidationError

from rpent.evaluation.result import write_json_atomic
from rpent.improvement.controller import ImprovementHarness, check, load_config
from rpent.improvement.deployment import select_snapshot
from rpent.improvement.models import QualificationReport, TaskCase, TrialSpec
from rpent.memory.snapshots import SnapshotStore
from rpent.utils.logging import get_logger

logger = get_logger("improvement")


def main(argv: list[str] | None = None) -> int:
    """Run the explicit protocol; static check requires no model credentials."""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("check", "run", "deploy"):
        command = commands.add_parser(name)
        command.add_argument("--config", required=True, type=Path)
        if name == "deploy":
            command.add_argument("--report", required=True, type=Path)
            command.add_argument("--suite", required=True)
            command.add_argument("--task", required=True, type=int)
            command.add_argument("--seed", required=True, type=int)
            command.add_argument("--output-dir", required=True, type=Path)
    for name in ("plan", "construct"):
        command = commands.add_parser(name)
        command.add_argument("--config", required=True, type=Path)
    command = commands.add_parser("evaluate")
    command.add_argument("--frozen", required=True, type=Path)
    command.add_argument("--output-dir", required=True, type=Path)
    command = commands.add_parser("compare")
    command.add_argument("--reports", nargs="+", required=True, type=Path)
    command.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        if args.command in {"plan", "construct", "evaluate", "compare"}:
            return _stream_command(args)
        config = load_config(args.config)
        if args.command == "check":
            result = check(config)
            logger.info(
                "Static inputs valid; baseline=%s. No behavioral qualification was performed.",
                result["baseline_hash"],
            )
            return 0
        from robots.libero.improvement import build_trial_runner

        runner = build_trial_runner(config.execution)
        if args.command == "run":
            report = ImprovementHarness(config, runner).run()
            logger.info(
                "Qualification: %s (%s)", report.decision, "; ".join(report.reasons)
            )
            return 0 if report.decision == "accepted" else 1
        report = QualificationReport.model_validate_json(args.report.read_text("utf-8"))
        case = TaskCase(
            case_id="deployment", suite=args.suite, task_id=args.task, seed=args.seed
        )
        store = SnapshotStore(config.output_dir / "snapshots")
        selection, snapshot = select_snapshot(config, report, case, store)
        output = args.output_dir.resolve()
        if output.exists():
            raise ValueError(
                "deployment output already exists; choose a fresh directory"
            )
        output.mkdir(parents=True)
        write_json_atomic(output / "selection.json", selection.model_dump(mode="json"))
        spec = TrialSpec(
            trial_id="deployment",
            experiment_id=config.experiment_id,
            case=case,
            snapshot_id=snapshot.snapshot_id,
            snapshot_hash=snapshot.content_hash,
            scaffold_hash=report.scaffold_hash,
            budget=config.execution.budget,
            output_dir=output,
        )
        write_json_atomic(output / "trial_spec.json", spec.model_dump(mode="json"))
        result = runner.run(spec, snapshot)
        store.verify(snapshot)
        write_json_atomic(output / "trial_result.json", result.model_dump(mode="json"))
        return (
            0
            if result.outcome == "completed"
            and result.official_success is True
            and result.cleanup_ok
            else 1
        )
    except (ValueError, OSError, RuntimeError, ValidationError) as exc:
        logger.error("Improvement run failed: %s", exc)
        return 2


def _stream_command(args: argparse.Namespace) -> int:
    """Dispatch the construction/evaluation workflow without legacy qualification."""
    from rpent.improvement.stream import (
        compare,
        construct,
        evaluate,
        load_stream_config,
        plan,
    )

    if args.command == "evaluate":
        report = evaluate(args.frozen, args.output_dir)
        logger.info(
            "Evaluation %s: %s/%s native successes",
            report["status"],
            report["successes"],
            report["requested"],
        )
        return 0 if report["status"] == "completed" else 2
    if args.command == "compare":
        compare(args.reports, args.output)
        logger.info("Paired memory comparison written to %s", args.output)
        return 0
    config = load_stream_config(args.config)
    if args.command == "plan":
        report = plan(config)
        logger.info(
            "Offline plan: %s construction rollouts; %s evaluation cases per arm. Native identities remain unverified.",
            report["expected_rollouts"],
            len(report["evaluation_cases"]),
        )
        return 0
    from robots.libero.improvement import build_trial_runner
    from rpent.improvement.ppc import PPCController
    from rpent.improvement.ppc_proposer import build_editor

    controller = PPCController(
        build_trial_runner(config.execution, export_task_records=True),
        build_editor(**config.editor.model_dump()),
        SnapshotStore(config.output_dir / "snapshots"),
    )
    report = construct(config, controller)
    logger.info(
        "Memory frozen after %s tasks: %s", report["tasks"], report["snapshot_hash"]
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
