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

"""Independent LIBERO evaluation episodes for memory improvement trials."""

from __future__ import annotations

import argparse
import threading
import time
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

from robots.libero.robot_spec import get_robot_spec, get_toolkit
from rpent.dashboard.events import NullDashboardEventSink
from rpent.evaluation.result import write_json_atomic
from rpent.improvement.models import (
    ExecutionConfig,
    TrialCost,
    TrialResult,
    TrialSpec,
    initial_state_key,
)
from rpent.memory import MemoryManager
from rpent.memory.loading import prepare_run_memory
from rpent.memory.snapshots import SnapshotStore
from rpent.planner.base import build_planner
from rpent.utils.config import get_repo_root
from rpent.utils.logging import get_logger, init_output_dir
from rpent.utils.serialization import strip_inline_images

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

    from rpent.improvement.controller import TrialRunner
    from rpent.memory.snapshots import MemorySnapshot

logger = get_logger("improvement.libero")
_RUN_LOCK = threading.Lock()
_ROBOT_OPTIONS = {
    "max_episode_steps",
    "libero_type",
    "env_endpoint",
    "vla_endpoint",
    "sam3_endpoint",
    "cuda_device",
}


class _BudgetExceeded(RuntimeError):
    """An operation was refused before issuing its model or environment request."""


class _TrialOperations:
    """Measure actual execution entries and admit them against episode limits."""

    def __init__(self, spec: TrialSpec) -> None:
        self.spec = spec
        self.started = time.perf_counter()
        self.deadline = self.started + spec.budget.timeout_s
        self.env_steps = 0
        self.vla_calls = 0
        self.resets = 0
        self.env_steps_unknown = False
        self.reset_completed = False
        self.exhausted: str | None = None
        self.operation_error: str | None = None
        self.spans: list[dict[str, Any]] = []

    @contextmanager
    def span(self, category: str) -> Iterator[None]:
        start = time.perf_counter_ns()
        status = "completed"
        try:
            yield
        except BaseException:
            status = "error"
            raise
        finally:
            self.spans.append(
                {
                    "trial_id": self.spec.trial_id,
                    "call_id": f"span_{len(self.spans):06d}",
                    "category": category,
                    "start_ns": start,
                    "end_ns": time.perf_counter_ns(),
                    "status": status,
                }
            )

    def check(self, *, env_steps: int = 0, vla: bool = False) -> None:
        budget = self.spec.budget
        reason = self.exhausted
        if time.perf_counter() >= self.deadline:
            reason = "trial wall-clock budget exhausted"
        if (
            budget.max_env_steps is not None
            and self.env_steps + env_steps > budget.max_env_steps
        ):
            reason = "environment step budget cannot admit this action chunk"
        if vla and budget.max_vla_calls is not None:
            if self.vla_calls >= budget.max_vla_calls:
                reason = "VLA call budget exhausted"
        if reason:
            self.exhausted = reason
            raise _BudgetExceeded(reason)


class _BudgetEnv:
    """Keep hard admission at the actual step/chunk entry, below primitives."""

    def __init__(self, env: Any, operations: _TrialOperations) -> None:
        self._env = env
        self._operations = operations

    def __getattr__(self, name: str) -> Any:
        return getattr(self._env, name)

    def reset(self) -> Any:
        operations = self._operations
        operations.check()
        operations.resets += 1
        with operations.span("environment.reset"):
            result = self._env.reset()
        operations.reset_completed = True
        return result

    def step(self, action: Any) -> Any:
        return self._execute(1, self._env.step, action)

    def chunk_step(self, actions: Any, **kwargs: Any) -> Any:
        return self._execute(len(actions), self._env.chunk_step, actions, **kwargs)

    def _execute(self, length: int, handler: Any, *args: Any, **kwargs: Any) -> Any:
        operations = self._operations
        operations.check(env_steps=length)
        if length < 1:
            raise ValueError("an action chunk must contain at least one action")
        try:
            with operations.span(
                "environment.step" if length == 1 else "environment.chunk"
            ):
                result = handler(*args, **kwargs)
            operations.env_steps += length
            if length > 1:
                # RLinf returns a full chunk array, without an executed-step
                # count. Its post-termination padding cannot establish how
                # many actions advanced physics. Spend the full admission
                # reservation and keep the actual transition count unknown.
                operations.env_steps_unknown = True
            return result
        except Exception as exc:
            # A failed RPC may already have advanced the simulator. Keep the
            # reservation spent and report that the actual count is unknown.
            operations.env_steps += length
            operations.env_steps_unknown = True
            operations.operation_error = f"environment request failed: {exc}"
            raise


class _BudgetModel:
    """Count attempted predictions, including failed or unused VLA proposals."""

    def __init__(self, model: Any, operations: _TrialOperations) -> None:
        self._model = model
        self._operations = operations

    def predict(self, *args: Any, **kwargs: Any) -> Any:
        operations = self._operations
        operations.check(vla=True)
        operations.vla_calls += 1
        try:
            with operations.span("vla.predict"):
                return self._model.predict(*args, **kwargs)
        except Exception as exc:
            operations.operation_error = f"VLA request failed: {exc}"
            raise


def _runtime_args(
    execution: ExecutionConfig, spec: TrialSpec, snapshot: MemorySnapshot
):
    unknown = set(execution.robot_options) - _ROBOT_OPTIONS
    if unknown:
        raise ValueError(f"unsupported trial robot options: {sorted(unknown)}")
    robot = get_robot_spec()
    parser = argparse.ArgumentParser(add_help=False, exit_on_error=False)
    robot.add_cli_args(parser, False)
    argv = [
        "--suite",
        spec.case.suite,
        "--task",
        str(spec.case.task_id),
        "--seed",
        str(spec.case.seed),
    ]
    for name, value in execution.robot_options.items():
        if value is not None:
            argv.extend([f"--{name.replace('_', '-')}", str(value)])
    try:
        args = parser.parse_args(argv)
    except argparse.ArgumentError as exc:
        raise ValueError(str(exc)) from exc
    args.planner = execution.planner
    args.model = execution.model
    args.explore = False
    args.memory_profile = "local"
    args.memory_dir = str(snapshot.corpus_dir)
    args.output_dir = spec.output_dir
    # The toolkit owns the one trial reset, observed by _BudgetEnv.
    args.env_reset_on_connect = False
    if args.max_episode_steps <= 0:
        raise ValueError("max_episode_steps must be positive")
    if robot.validate_args is not None:
        robot.validate_args(args)
    config = robot.parse_config(args)
    config.prompt_vars["output_dir"] = str(spec.output_dir.resolve())
    prepare_run_memory(args, robot, config)
    return robot, args, config


class LiberoTrialRunner:
    """Run serial planner trials with isolated memory and native task outcomes.

    Borrowed endpoints remain owned by their operator. Each run creates a fresh
    toolkit and planner context and stops only the processes it started.
    """

    def __init__(
        self,
        execution: ExecutionConfig,
        *,
        export_task_records: bool = False,
        forbidden_initial_states: set[str] | None = None,
        planner_factory: Callable[..., Any] | None = None,
    ) -> None:
        self.execution = execution
        self.export_task_records = export_task_records
        self.forbidden_initial_states = forbidden_initial_states
        self.planner_factory = planner_factory or build_planner

    def run(self, spec: TrialSpec, snapshot: MemorySnapshot) -> TrialResult:
        """Execute one immutable memory snapshot and persist its trial evidence."""
        with _RUN_LOCK:
            return self._run(spec, snapshot)

    def _run(self, spec: TrialSpec, snapshot: MemorySnapshot) -> TrialResult:
        operations = _TrialOperations(spec)
        if (
            spec.snapshot_id != snapshot.snapshot_id
            or spec.snapshot_hash != snapshot.content_hash
        ):
            raise ValueError(
                "trial snapshot identity does not match its materialized corpus"
            )
        store = SnapshotStore(snapshot.corpus_dir.parent.parent)
        store.verify(snapshot)
        output = spec.output_dir.resolve()
        corpus = snapshot.corpus_dir.resolve()
        if output.is_relative_to(corpus) or corpus.is_relative_to(output):
            raise ValueError("trial output and memory corpus must have separate roots")
        if output.exists() and any(
            path.name not in {"trial_spec.json", "trial_events.jsonl", "selection.json"}
            for path in output.iterdir()
        ):
            raise ValueError(f"trial output must be fresh: {output}")
        output.mkdir(parents=True, exist_ok=True)
        init_output_dir(output)
        sink = NullDashboardEventSink()
        reads: list[dict[str, Any]] = []
        memory = MemoryManager(
            corpus,
            memory_access="read_only",
            read_audit=lambda event: reads.append(
                {**event, "trial_id": spec.trial_id, "snapshot_id": spec.snapshot_id}
            ),
            allowed_read_roots=[
                corpus,
                output,
                *(
                    get_repo_root() / "robots" / "libero" / "guides" / name
                    for name in (
                        "strict_hybrid_guide.md",
                        "pro_hybrid_guide.md",
                        "env_calibration.md",
                    )
                ),
            ],
            allowed_write_roots=[output],
        )
        daemons: list[Any] = []
        toolkit = None
        env = None
        initial_state_id = None
        planner_result = None
        error = None
        outcome = "completed"
        cleanup_ok = True
        official_success = None
        terminated = truncated = False
        task_records: dict[str, str] = {}
        try:
            robot, args, config = _runtime_args(self.execution, spec, snapshot)
            with operations.span("runtime.startup"):
                daemons, runtime = robot.init_runtime(args, output, sink, None)
            env = _BudgetEnv(runtime["env"], operations)
            initial_state_id = env.get_initial_state_id()
            if not isinstance(initial_state_id, str) or not initial_state_id:
                raise ValueError(
                    "environment does not expose its native initial-state identity"
                )
            if (
                self.forbidden_initial_states is not None
                and initial_state_key(initial_state_id) in self.forbidden_initial_states
            ):
                raise ValueError(
                    "initial state overlaps construction or an earlier evaluation case"
                )
            runtime = {
                **runtime,
                "env": env,
                "model": _BudgetModel(runtime["model"], operations),
            }
            with operations.span("toolkit.startup"):
                toolkit = get_toolkit(
                    runtime_kwargs=runtime,
                    dashboard_events=sink,
                    config=config,
                    mode="evaluation",
                    state_output_dir=output,
                    memory=memory,
                )
            operations.check()
            with operations.span("planner.startup"):
                planner = self.planner_factory(
                    self.execution.planner,
                    output_dir=output,
                    recipe_tag=config.recipe_tag,
                    robot_name="libero",
                    base_url=self.execution.base_url,
                    model=self.execution.model,
                    memory_dir=corpus,
                    max_tokens=self.execution.max_tokens,
                    planner_timeout_s=max(
                        0.001, operations.deadline - time.perf_counter()
                    ),
                    reasoning_effort=self.execution.reasoning_effort,
                    dashboard_events=sink,
                    **(
                        {"tools_only": True}
                        if self.execution.planner == "codex"
                        else {}
                    ),
                )
            with operations.span("planner.solve"):
                planner_result = planner.solve(
                    system_prompt=robot.prompts.render(
                        "system", variables=config.prompt_vars
                    ),
                    user_message=robot.prompts.render(
                        "user", variables=config.prompt_vars
                    ),
                    toolkit=toolkit,
                    max_turns=spec.budget.max_turns,
                )
            error = planner_result.error
            if error:
                outcome = (
                    "budget_exceeded" if "timed out" in error.lower() else "infra_error"
                )
            elif (
                planner_result.finish_result is None
                and planner_result.stats.get("turns_used", 0) >= spec.budget.max_turns
            ):
                outcome, error = "budget_exceeded", "planner request budget exhausted"
        except _BudgetExceeded as exc:
            outcome, error = "budget_exceeded", str(exc)
        except TimeoutError as exc:
            outcome, error = "budget_exceeded", str(exc) or "planner timed out"
        except (KeyboardInterrupt, InterruptedError) as exc:
            outcome, error = "cancelled", str(exc) or type(exc).__name__
        except Exception as exc:
            outcome, error = "infra_error", f"{type(exc).__name__}: {exc}"
        finally:
            with operations.span("cleanup"):
                if toolkit is not None:
                    try:
                        toolkit.close()
                    except Exception as exc:
                        cleanup_ok = False
                        error = f"{error + '; ' if error else ''}toolkit cleanup failed: {exc}"
                if env is not None and operations.reset_completed:
                    terminated = bool(env.terminated)
                    truncated = bool(env.truncated)
                    official_success = terminated
                    if self.export_task_records and terminated and toolkit is not None:
                        try:
                            recipe = toolkit.write_recipe(config.recipe_tag)
                            if not recipe:
                                raise ValueError(
                                    "successful episode has no recorded recipe"
                                )
                            audit_path = output / f"{config.recipe_tag}.json"
                            write_json_atomic(
                                audit_path,
                                {
                                    "suite": spec.case.suite,
                                    "task_id": spec.case.task_id,
                                    "seed": spec.case.seed,
                                    "regime": "strict_perception",
                                    "task_language": env.get_task_language(),
                                    "libero_terminated": True,
                                    "terminated": True,
                                    "initial_state_id": initial_state_id,
                                    "source_trial": spec.trial_id,
                                    "snapshot_hash": snapshot.content_hash,
                                    "record_source": "native environment and recorded command trace",
                                },
                            )
                            task_records = {
                                "task_audit": str(audit_path),
                                "task_recipe": str(output / recipe),
                            }
                        except Exception as exc:
                            cleanup_ok = False
                            error = f"{error + '; ' if error else ''}task record export failed: {exc}"
                for daemon in reversed(daemons):
                    try:
                        daemon.stop()
                    except Exception as exc:
                        cleanup_ok = False
                        error = f"{error + '; ' if error else ''}daemon cleanup failed: {exc}"
        if operations.exhausted:
            outcome, error = "budget_exceeded", error or operations.exhausted
        elif operations.operation_error:
            outcome, error = "infra_error", error or operations.operation_error
        if not cleanup_ok:
            outcome = "infra_error"
        if time.perf_counter() >= operations.deadline and outcome == "completed":
            outcome, error = "budget_exceeded", "trial wall-clock budget exhausted"
        try:
            store.verify(snapshot)
        except Exception as exc:
            outcome = "infra_error"
            error = (
                f"{error + '; ' if error else ''}snapshot integrity check failed: {exc}"
            )
        stats = planner_result.stats if planner_result is not None else {}
        artifacts = {
            "result": str(output / "trial.json"),
            "transcript": str(output / "transcript.json"),
            "spans": str(output / "spans.json"),
            "budget": str(output / "budget.json"),
            **task_records,
        }
        if (output / "states.json").exists():
            artifacts["states"] = str(output / "states.json")
        write_json_atomic(
            output / "transcript.json",
            {
                "finish": planner_result.finish_result if planner_result else None,
                "messages": strip_inline_images(planner_result.messages)
                if planner_result
                else [],
                "stats": stats,
                "error": error,
            },
        )
        write_json_atomic(output / "spans.json", {"spans": operations.spans})
        write_json_atomic(
            output / "budget.json",
            {
                "environment_steps_admitted": operations.env_steps,
                "actual_environment_steps_known": not operations.env_steps_unknown,
                "limit": spec.budget.model_dump(mode="json"),
                "exhausted": operations.exhausted,
                "accounting": "Full requested chunks consume admission; chunk transition counts are unavailable.",
            },
        )
        result = TrialResult(
            spec=spec,
            outcome=outcome,
            official_success=official_success,
            success_source="libero.env.terminated"
            if official_success is not None
            else "unavailable",
            initial_state_id=initial_state_id,
            terminated=terminated,
            truncated=truncated,
            cleanup_ok=cleanup_ok,
            error=error,
            cost=TrialCost(
                wall_time_s=time.perf_counter() - operations.started,
                planner_requests=stats.get("turns_used"),
                input_tokens=stats.get("total_input_tokens"),
                output_tokens=stats.get("total_output_tokens"),
                env_steps=None
                if operations.env_steps_unknown
                else operations.env_steps,
                vla_calls=operations.vla_calls,
                resets=operations.resets,
                unknown=["env_steps"] if operations.env_steps_unknown else [],
            ),
            memory_reads=reads,
            artifacts=artifacts,
        )
        write_json_atomic(output / "trial.json", result.model_dump(mode="json"))
        logger.info(
            "trial %s: %s, native success=%s", spec.trial_id, outcome, official_success
        )
        return result


def build_trial_runner(execution: ExecutionConfig, **kwargs: Any) -> TrialRunner:
    """Select local runtime ownership or a borrowed SSH-forwarded trial worker."""
    if execution.remote is not None:
        from robots.libero.remote import RemoteLiberoTrialRunner

        return RemoteLiberoTrialRunner(execution, **kwargs)
    return LiberoTrialRunner(execution, **kwargs)
