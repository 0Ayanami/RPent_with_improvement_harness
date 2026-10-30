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

"""Local profile generation and OpenSSH boundaries without a remote service."""

import pytest
import yaml
from pydantic import ValidationError

from rpent.improvement.models import ExecutionConfig, RemoteExecutionConfig
from rpent.improvement.ssh import (
    SSHConnectionConfig,
    load_connection,
    main,
    setup,
    tunnel_command,
)
from rpent.improvement.stream import load_stream_config, plan


def test_profile_plans_paired_codex_arms_without_model_or_simulator(tmp_path):
    directory = tmp_path / "local"
    connection = SSHConnectionConfig(
        host="222.20.97.104", user="wanquan", local_port=18870
    )
    setup(directory, connection)
    meta = load_stream_config(directory / "experiment.yaml")
    no_meta = load_stream_config(directory / "experiment-no-meta.yaml")
    assert meta.execution.planner == meta.editor.planner == "codex"
    assert meta.execution.remote.endpoint == "http://127.0.0.1:18870"
    assert meta.execution == no_meta.execution
    assert meta.output_dir != no_meta.output_dir
    assert meta.use_meta_library and not no_meta.use_meta_library
    planned = plan(meta)
    assert planned["expected_rollouts"] == 160
    assert len(planned["evaluation_cases"]) == 400
    assert not planned["behavior_verified"]
    assert load_connection(directory / "connection.yaml") == connection
    before = (directory / "experiment.yaml").read_bytes()
    with pytest.raises(ValueError, match="new directory"):
        setup(directory, connection)
    assert (directory / "experiment.yaml").read_bytes() == before


def test_existing_reference_and_identity_are_preserved(tmp_path):
    reference = tmp_path / "corpus"
    reference.mkdir()
    (reference / "MEMORY.md").write_text("# Actual reference\n", encoding="utf-8")
    directory = tmp_path / "profile"
    setup(
        directory,
        SSHConnectionConfig(host="gpu6"),
        reference_memory=reference,
        reference_provenance="fixed revision with seed history",
        vla_identity="sha256:checkpoint",
    )
    config = load_stream_config(directory / "experiment.yaml")
    assert config.reference_memory.path == reference
    assert config.reference_memory.provenance == "fixed revision with seed history"
    assert (
        config.execution.vla_identity
        == config.evaluation_execution.vla_identity
        == "sha256:checkpoint"
    )


def test_tunnel_keeps_forward_local_and_does_not_evaluate_shell(tmp_path):
    key = tmp_path / "key with spaces"
    config = SSHConnectionConfig(
        host="gpu6", user="wanquan", port=2222, identity_file=key
    )
    command = tunnel_command(config)
    assert command[-1] == "wanquan@gpu6"
    assert command[command.index("-L") + 1] == "127.0.0.1:8870:127.0.0.1:8870"
    assert command[command.index("-i") + 1] == str(key)
    assert "StrictHostKeyChecking=yes" in command
    assert "ExitOnForwardFailure=yes" in command
    assert "-N" in command and "-T" in command
    with pytest.raises(ValidationError):
        SSHConnectionConfig(host="-oProxyCommand=bad")


@pytest.mark.parametrize(
    "endpoint",
    [
        "http://example.org:8870",
        "http://127.0.0.1",
        "https://localhost:8870",
        "http://localhost:8870/path",
        "http://user:secret@localhost:8870",
        "http://localhost:8870?token=secret",
        "http://localhost:99999",
    ],
)
def test_remote_endpoint_requires_explicit_loopback_forward(endpoint):
    with pytest.raises(ValueError):
        RemoteExecutionConfig(endpoint=endpoint)


def test_remote_rejects_api_before_trial_start():
    with pytest.raises(ValueError, match="require planner=codex"):
        ExecutionConfig(model="test:model", vla_identity="fixture", remote={})


def test_tunnel_dry_run_never_launches_process(tmp_path, monkeypatch):
    config = tmp_path / "connection.yaml"
    config.write_text(
        yaml.safe_dump({"host": "gpu6", "user": "wanquan"}), encoding="utf-8"
    )

    def fail(*args, **kwargs):
        raise AssertionError("dry-run must not launch SSH")

    monkeypatch.setattr("rpent.improvement.ssh.subprocess.call", fail)
    assert main(["tunnel", "--config", str(config), "--dry-run"]) == 0
