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

"""Prepare local Codex experiments and forward the remote worker over OpenSSH."""

from __future__ import annotations

import argparse
import copy
import shlex
import subprocess
from pathlib import Path

import httpx
import yaml
from pydantic import Field, field_validator

from rpent.evaluation.result import write_json_atomic
from rpent.improvement.models import Record, TaskCase, TrialSpec
from rpent.utils.config import get_repo_root
from rpent.utils.logging import get_logger, init_output_dir

logger = get_logger("improvement.ssh")


class SSHConnectionConfig(Record):
    """OpenSSH destination and one loopback-only local port forward."""

    host: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
    user: str | None = Field(default=None, pattern=r"^[A-Za-z0-9_][A-Za-z0-9_.-]*$")
    port: int = Field(default=22, ge=1, le=65535)
    local_port: int = Field(default=8870, ge=1, le=65535)
    remote_port: int = Field(default=8870, ge=1, le=65535)
    identity_file: Path | None = None
    server_project: str = "~/RPent"

    @field_validator("server_project")
    @classmethod
    def validate_project(cls, value: str) -> str:
        if not value or any(char in value for char in "\r\n\x00"):
            raise ValueError("server_project must be a single nonempty path")
        return value


def load_connection(path: Path) -> SSHConnectionConfig:
    """Read a connection file without reading credentials or changing SSH state."""
    config = SSHConnectionConfig.model_validate(yaml.safe_load(path.read_text("utf-8")))
    if config.identity_file is not None:
        identity = config.identity_file.expanduser()
        config = config.model_copy(
            update={"identity_file": (path.resolve().parent / identity).resolve()}
        )
    return config


def tunnel_command(config: SSHConnectionConfig) -> list[str]:
    """Build an argv vector; no local or remote shell command is evaluated."""
    command = [
        "ssh",
        "-N",
        "-T",
        "-p",
        str(config.port),
        "-o",
        "ExitOnForwardFailure=yes",
        "-o",
        "ServerAliveInterval=30",
        "-o",
        "ServerAliveCountMax=3",
        "-o",
        "StrictHostKeyChecking=yes",
        "-L",
        f"127.0.0.1:{config.local_port}:127.0.0.1:{config.remote_port}",
    ]
    if config.identity_file is not None:
        command.extend(["-i", str(config.identity_file)])
    destination = f"{config.user}@{config.host}" if config.user else config.host
    return [*command, destination]


def setup(
    directory: Path,
    connection: SSHConnectionConfig,
    *,
    reference_memory: Path | None = None,
    reference_provenance: str | None = None,
    vla_identity: str | None = None,
) -> None:
    """Create a fresh local profile; preserve every pre-existing directory."""
    directory = directory.resolve()
    if directory.exists():
        raise ValueError(
            "setup requires a new directory; existing profiles are preserved"
        )
    source = (
        get_repo_root() / "robots/libero/config/improvement/windows_codex_remote.yaml"
    )
    template = yaml.safe_load(source.read_text("utf-8"))
    if reference_memory is not None:
        reference_memory = reference_memory.resolve()
        if not (reference_memory / "MEMORY.md").is_file():
            raise ValueError("reference memory requires an existing MEMORY.md")
        template["reference_memory"]["path"] = str(reference_memory)
    if reference_provenance is not None:
        template["reference_memory"]["provenance"] = reference_provenance
    for name in ("execution", "evaluation_execution"):
        template[name]["remote"]["endpoint"] = (
            f"http://127.0.0.1:{connection.local_port}"
        )
        if vla_identity is not None:
            template[name]["vla_identity"] = vla_identity
    directory.mkdir(parents=True)
    for name in ("initial", "reference"):
        corpus = directory / "memory" / name
        corpus.mkdir(parents=True)
        (corpus / "MEMORY.md").write_text("# Memory\n", encoding="utf-8")
    (directory / "connection.yaml").write_text(
        yaml.safe_dump(connection.model_dump(mode="json"), allow_unicode=True),
        encoding="utf-8",
    )
    for meta, filename in (
        (True, "experiment.yaml"),
        (False, "experiment-no-meta.yaml"),
    ):
        config = copy.deepcopy(template)
        config["experiment_id"] = f"libero-codex-ppc-{'meta' if meta else 'no-meta'}"
        config["output_dir"] = f"runs/{'meta' if meta else 'no-meta'}"
        config["use_meta_library"] = meta
        (directory / filename).write_text(
            yaml.safe_dump(config, allow_unicode=True, sort_keys=False),
            encoding="utf-8",
        )
    logger.info(
        "Local profile written to %s; fill checkpoint and reference provenance before construction",
        directory,
    )


def check(connection: SSHConnectionConfig) -> dict:
    """Check worker protocol and matching source through an already-open tunnel."""
    from robots.libero.remote import source_identity

    endpoint = f"http://127.0.0.1:{connection.local_port}"
    with httpx.Client(timeout=10, trust_env=False) as client:
        response = client.get(f"{endpoint}/health")
        response.raise_for_status()
        result = response.json()
    if result.get("protocol") != 1 or result.get("source_hash") != source_identity():
        raise ValueError(
            "worker protocol or source differs; synchronize the current checkout and restart worker"
        )
    if result.get("active_trial") is not None:
        raise ValueError("worker is busy with another trial")
    logger.info("Remote worker ready through %s", endpoint)
    return result


def smoke(config_path: Path, output: Path) -> int:
    """Run one bounded integration episode, separate from construction/evaluation."""
    from robots.libero.improvement import build_trial_runner
    from rpent.improvement.controller import scaffold_fingerprint
    from rpent.improvement.stream import load_stream_config
    from rpent.memory.snapshots import SnapshotStore

    config = load_stream_config(config_path)
    if config.execution.remote is None:
        raise ValueError("remote smoke requires execution.remote")
    output = output.resolve()
    if output.exists():
        raise ValueError("smoke requires a new output directory")
    budget = config.execution.budget.model_copy(
        update={
            "max_turns": 3,
            "timeout_s": 300,
            "max_env_steps": 200,
            "max_vla_calls": 2,
        }
    )
    execution = config.execution.model_copy(update={"budget": budget})
    store = SnapshotStore(output / "snapshots")
    snapshot = store.capture(config.initial_memory.path)
    case = TaskCase(case_id="smoke", suite="libero_spatial", task_id=0, seed=0)
    spec = TrialSpec(
        trial_id="smoke",
        experiment_id="remote-integration",
        case=case,
        snapshot_id=snapshot.snapshot_id,
        snapshot_hash=snapshot.content_hash,
        scaffold_hash=scaffold_fingerprint(execution),
        budget=budget,
        output_dir=output / "trial",
    )
    result = build_trial_runner(execution).run(spec, snapshot)
    write_json_atomic(output / "smoke.json", result.model_dump(mode="json"))
    logger.info(
        "Smoke: outcome=%s; native_success=%s; cleanup=%s; this is not a benchmark score",
        result.outcome,
        result.official_success,
        result.cleanup_ok,
    )
    valid = (
        result.cleanup_ok
        and result.initial_state_id is not None
        and result.official_success is not None
        and result.outcome in {"completed", "budget_exceeded"}
    )
    return 0 if valid else 2


def main(argv: list[str] | None = None) -> int:
    """Create profiles, maintain a foreground tunnel, or run connection checks."""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    command = commands.add_parser("setup")
    command.add_argument("--directory", type=Path, required=True)
    command.add_argument("--host", required=True)
    command.add_argument("--user")
    command.add_argument("--port", type=int, default=22)
    command.add_argument("--local-port", type=int, default=8870)
    command.add_argument("--remote-port", type=int, default=8870)
    command.add_argument("--identity-file", type=Path)
    command.add_argument("--server-project", default="~/RPent")
    command.add_argument("--reference-memory", type=Path)
    command.add_argument("--reference-provenance")
    command.add_argument("--vla-identity")
    for name in ("tunnel", "check", "smoke"):
        command = commands.add_parser(name)
        command.add_argument("--config", type=Path, required=True)
        if name == "tunnel":
            command.add_argument("--dry-run", action="store_true")
        if name == "smoke":
            command.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    init_output_dir(get_repo_root() / "logs" / "remote-cli")
    try:
        if args.command == "setup":
            setup(
                args.directory,
                SSHConnectionConfig(
                    host=args.host,
                    user=args.user,
                    port=args.port,
                    local_port=args.local_port,
                    remote_port=args.remote_port,
                    identity_file=args.identity_file.resolve()
                    if args.identity_file
                    else None,
                    server_project=args.server_project,
                ),
                reference_memory=args.reference_memory,
                reference_provenance=args.reference_provenance,
                vla_identity=args.vla_identity,
            )
            return 0
        if args.command == "smoke":
            return smoke(args.config, args.output_dir)
        connection = load_connection(args.config)
        if args.command == "check":
            result = check(connection)
            write_json_atomic(
                args.config.resolve().parent / "connection-check.json", result
            )
            return 0
        command = tunnel_command(connection)
        logger.info("SSH tunnel command: %s", shlex.join(command))
        if args.dry_run:
            return 0
        return subprocess.call(command)
    except KeyboardInterrupt:
        logger.info("Stopped by operator")
        return 130
    except (OSError, ValueError, RuntimeError, httpx.HTTPError) as exc:
        logger.error("Remote setup failed: %s", exc)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
