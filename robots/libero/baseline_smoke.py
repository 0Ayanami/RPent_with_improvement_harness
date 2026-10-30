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

"""Run one ordinary Harness-VLA episode through a local Codex/remote GPU bridge.

The transport reuses the native trial lifecycle. It never invokes a memory
editor, PPC, construction stream, reflection, adoption or meta-library.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from robots.libero.remote import RemoteLiberoTrialRunner, source_identity
from rpent.evaluation.result import write_json_atomic
from rpent.improvement.models import (
    ExecutionConfig,
    RemoteExecutionConfig,
    TaskCase,
    TrialBudget,
    TrialResult,
    TrialSpec,
    fingerprint,
)
from rpent.memory.snapshots import SnapshotStore
from rpent.utils.logging import get_logger, init_output_dir

logger = get_logger("libero.baseline_smoke")


def smoke_report(result: TrialResult) -> dict[str, Any]:
    """Qualify integration from recorded native execution, independently of success."""
    output = result.spec.output_dir.resolve()

    def read(name: str) -> dict[str, Any]:
        path = output / name
        return json.loads(path.read_text("utf-8")) if path.is_file() else {}

    transcript = read("transcript.json")
    states = read("states.json")
    budget = read("budget.json")
    steps = states.get("steps", [])
    action_steps = [
        step
        for step in steps
        if str((step.get("command") or {}).get("action", "")).startswith("pi0_")
        and not (step.get("result") or {}).get("error")
        and (step.get("result") or {}).get("chunks_used", 0) > 0
    ]
    observations = [
        str(path.relative_to(output)) for path in sorted(output.rglob("*.png"))
    ]
    finish = transcript.get("finish")
    checks = {
        "native_identity": bool(result.initial_state_id),
        "native_outcome_recorded": result.official_success is not None,
        "valid_trial": result.outcome in {"completed", "budget_exceeded"},
        "cleanup": result.cleanup_ok,
        "finish": isinstance(finish, dict) and finish.get("_finish") is True,
        "observation_artifacts": bool(observations),
        "vla_predictions": (result.cost.vla_calls or 0) > 0,
        "vla_action_recorded": bool(action_steps),
        "environment_actions": budget.get("environment_steps_admitted", 0) > 0,
    }
    return {
        "status": "passed" if all(checks.values()) else "failed",
        "definition": "local Codex -> ordinary RPent toolkit -> real Pi0.5 -> LIBERO actions -> finish -> cleanup",
        "adaptive_memory_enabled": False,
        "checks": checks,
        "native_success": result.official_success,
        "native_success_source": result.success_source,
        "initial_state_id": result.initial_state_id,
        "outcome": result.outcome,
        "error": result.error,
        "finish_status": finish.get("status") if isinstance(finish, dict) else None,
        "vla_calls": result.cost.vla_calls,
        "admitted_env_steps": budget.get("environment_steps_admitted"),
        "actual_env_steps": result.cost.env_steps,
        "recorded_vla_commands": len(action_steps),
        "observations": observations,
        "cost": result.cost.model_dump(mode="json"),
        "artifacts": result.artifacts,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Execute one baseline episode with an explicitly selected fixed memory corpus."""
    output = args.output_dir.resolve()
    if output.exists():
        raise ValueError("baseline smoke requires a fresh output directory")
    if args.memory_dir is not None:
        memory = args.memory_dir.resolve()
        if memory.is_relative_to(output) or output.is_relative_to(memory):
            raise ValueError("memory and output require separate directory trees")
        if not (memory / "MEMORY.md").is_file():
            raise ValueError("baseline memory requires an existing MEMORY.md")
        provenance = args.memory_provenance
    else:
        memory = output / "empty-memory"
        provenance = (
            "Explicit empty-memory connectivity smoke; not the paper memory baseline"
        )
    output.mkdir(parents=True)
    init_output_dir(output)
    if args.empty_memory:
        memory.mkdir()
        (memory / "MEMORY.md").write_text("# Memory\n", encoding="utf-8")
    store = SnapshotStore(output / "snapshots")
    snapshot = store.capture(memory)
    budget = TrialBudget(
        max_turns=args.max_turns,
        timeout_s=args.timeout_s,
        max_env_steps=args.max_env_steps,
        max_vla_calls=args.max_vla_calls,
    )
    options = {
        "libero_type": args.libero_type,
        "max_episode_steps": args.max_env_steps,
        "vla_endpoint": args.vla_endpoint,
        "sam3_endpoint": args.sam3_endpoint,
    }
    if args.cuda_device is not None:
        options["cuda_device"] = args.cuda_device
    execution = ExecutionConfig(
        planner="codex",
        model=args.model,
        reasoning_effort=args.reasoning_effort,
        vla_identity=args.vla_identity,
        budget=budget,
        robot_options=options,
        remote=RemoteExecutionConfig(
            endpoint=args.worker_endpoint, request_timeout_s=args.request_timeout_s
        ),
    )
    spec = TrialSpec(
        trial_id="baseline",
        experiment_id="harness-vla-baseline-smoke",
        case=TaskCase(
            case_id="baseline", suite=args.suite, task_id=args.task, seed=args.seed
        ),
        snapshot_id=snapshot.snapshot_id,
        snapshot_hash=snapshot.content_hash,
        scaffold_hash=fingerprint(
            {
                "source": source_identity(),
                "execution": execution.model_dump(mode="json"),
            }
        ),
        budget=budget,
        output_dir=output / "episode",
    )
    write_json_atomic(
        output / "baseline-config.json",
        {
            "execution": execution.model_dump(mode="json"),
            "case": spec.case.model_dump(mode="json"),
            "memory_source": str(memory),
            "memory_provenance": provenance,
            "memory_hash": snapshot.content_hash,
            "adaptive_memory_enabled": False,
        },
    )
    result = RemoteLiberoTrialRunner(execution).run(spec, snapshot)
    report = smoke_report(result)
    write_json_atomic(output / "baseline-smoke.json", report)
    logger.info(
        "Baseline smoke %s: native_success=%s; VLA calls=%s; admitted env steps=%s; finish=%s",
        report["status"],
        report["native_success"],
        report["vla_calls"],
        report["admitted_env_steps"],
        report["finish_status"],
    )
    return report


def main(argv: list[str] | None = None) -> int:
    """Run an explicit baseline memory episode and report integration evidence."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker-endpoint", default="http://127.0.0.1:8870")
    memory = parser.add_mutually_exclusive_group(required=True)
    memory.add_argument("--memory-dir", type=Path)
    memory.add_argument("--empty-memory", action="store_true")
    parser.add_argument(
        "--memory-provenance", default="Operator-supplied original Harness-VLA corpus"
    )
    parser.add_argument("--suite", default="libero_object")
    parser.add_argument("--task", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--libero-type", choices=("standard", "pro", "plus"), default="standard"
    )
    parser.add_argument("--model", default="gpt-5.5")
    parser.add_argument(
        "--reasoning-effort",
        choices=("none", "low", "medium", "high", "xhigh"),
        default="xhigh",
    )
    parser.add_argument("--max-turns", type=int, default=30)
    parser.add_argument("--timeout-s", type=int, default=1200)
    parser.add_argument("--max-env-steps", type=int, default=1200)
    parser.add_argument("--max-vla-calls", type=int, default=240)
    parser.add_argument("--request-timeout-s", type=int, default=1200)
    parser.add_argument("--vla-identity", required=True)
    parser.add_argument("--vla-endpoint", default="http://127.0.0.1:8801")
    parser.add_argument("--sam3-endpoint", default="http://127.0.0.1:8802")
    parser.add_argument("--cuda-device", type=int)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        report = run(args)
        return 0 if report["status"] == "passed" else 2
    except (OSError, RuntimeError, ValueError) as exc:
        logger.error("Baseline smoke failed: %s", exc)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
