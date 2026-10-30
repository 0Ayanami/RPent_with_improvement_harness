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

"""Start, inspect or stop only this workspace's Pi0.5 and SAM3 processes."""

from __future__ import annotations

import argparse
import os
import signal
import socket
import subprocess
from pathlib import Path

from rpent.utils.logging import get_logger, init_output_dir

ROOT = Path(__file__).resolve().parents[3]
LOGS = ROOT / "logs/deploy-20261030"
SERVICES = {
    "pi05": (
        2,
        8801,
        "rpent.robots.components.pi05_vla_server",
        ["--embodiment", "libero"],
    ),
    "sam3": (5, 8802, "rpent.robots.components.sam3_server", []),
}
logger = get_logger("libero.deployment")


def environment() -> dict[str, str]:
    """Use the committed workspace environment, never an ignored historical copy."""
    script = Path(__file__).with_name("environment.sh")
    result = subprocess.run(
        [
            "bash",
            "--noprofile",
            "--norc",
            "-c",
            'source "$1" || exit; env -0',
            "--",
            str(script),
        ],
        cwd=ROOT,
        capture_output=True,
        check=True,
    )
    return {
        key.decode(): value.decode()
        for item in result.stdout.split(b"\0")
        if item
        for key, value in [item.split(b"=", 1)]
    }


def own_process(name: str) -> int | None:
    """Reject reused PIDs before any signal is sent."""
    pidfile = LOGS / f"{name}.pid"
    if not pidfile.is_file():
        return None
    pid = int(pidfile.read_text())
    proc = Path(f"/proc/{pid}")
    if not proc.exists():
        return None
    command = (proc / "cmdline").read_bytes().replace(b"\0", b" ").decode()
    if (
        SERVICES[name][2] not in command
        or str(ROOT / ".venv/bin/python") not in command
        or (proc / "cwd").resolve() != ROOT
    ):
        raise RuntimeError(f"PID {pid} no longer belongs to {name}; refusing to signal")
    return pid


def main(argv: list[str] | None = None) -> int:
    """Manage the pinned local services; worker lifetime remains operator owned."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("start", "status", "stop"))
    args = parser.parse_args(argv)
    if str(ROOT) != "/home2/wanquan/RPent":
        raise ValueError("this server profile requires /home2/wanquan/RPent")
    init_output_dir(LOGS)
    for name, (gpu, port, module, extra) in SERVICES.items():
        pid = own_process(name)
        if args.action == "start" and pid is None:
            with socket.socket() as sock:
                sock.bind(("127.0.0.1", port))
            env = environment()
            env["CUDA_VISIBLE_DEVICES"] = str(gpu)
            command = [
                str(ROOT / ".venv/bin/python"),
                "-m",
                module,
                *extra,
                "--cuda-device",
                str(gpu),
                "--transport",
                "http",
                "--host",
                "127.0.0.1",
                "--port",
                str(port),
            ]
            with (LOGS / f"{name}.log").open("ab", buffering=0) as output:
                process = subprocess.Popen(
                    command,
                    cwd=ROOT,
                    env=env,
                    stdin=subprocess.DEVNULL,
                    stdout=output,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
            pid = process.pid
            (LOGS / f"{name}.pid").write_text(str(pid) + "\n")
        elif args.action == "stop" and pid is not None:
            os.kill(pid, signal.SIGTERM)
        logger.info(
            "%s: action=%s pid=%s port=%s physical_gpu=%s",
            name,
            args.action,
            pid,
            port,
            gpu,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
