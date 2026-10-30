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

"""Fetch pinned baseline weights, LIBERO assets or the original memory corpus."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import subprocess
from pathlib import Path, PurePosixPath
from typing import Any

from rpent.evaluation.result import write_json_atomic
from rpent.utils.logging import get_logger, init_output_dir

ROOT = Path(__file__).resolve().parents[3]
CONFIG = Path(__file__).with_name("resources.json")
logger = get_logger("libero.deployment")


def workspace_path(value: str | Path) -> Path:
    """Resolve an operation's destination inside this checkout."""
    result = (ROOT / value).resolve()
    if not result.is_relative_to(ROOT):
        raise ValueError(f"destination leaves the RPent workspace: {value}")
    return result


def sha256(path: Path) -> str:
    """Hash a file without loading model weights into memory."""
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def _write_receipt(name: str, value: dict[str, Any]) -> None:
    destination = workspace_path(name)
    destination.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(destination, value)


def models(config: dict[str, Any], *, check_only: bool) -> None:
    """Resume the five public files and publish a receipt only after verification."""

    def fetch(item: dict[str, Any]) -> dict[str, Any]:
        destination = workspace_path(item["path"])
        if not destination.is_file():
            if check_only:
                raise ValueError(f"missing model file: {destination}")
            destination.parent.mkdir(parents=True, exist_ok=True)
            partial = workspace_path(
                destination.with_name(destination.name + ".partial")
            )
            if not partial.is_file() or sha256(partial) != item["sha256"]:
                logger.info("Downloading %s", item["path"])
                subprocess.run(
                    [
                        "curl",
                        "--silent",
                        "--show-error",
                        "-L",
                        "--fail",
                        "--retry",
                        "8",
                        "--retry-delay",
                        "5",
                        "--connect-timeout",
                        "20",
                        "--speed-time",
                        "90",
                        "--speed-limit",
                        "1024",
                        "--continue-at",
                        "-",
                        "--output",
                        str(partial),
                        item["url"],
                    ],
                    check=True,
                )
            if sha256(partial) != item["sha256"]:
                raise ValueError(f"download hash mismatch: {partial}")
            partial.replace(destination)
        actual = sha256(destination)
        if actual != item["sha256"]:
            raise ValueError(
                f"existing file hash mismatch; preserved for inspection: {destination}"
            )
        return {**item, "bytes": destination.stat().st_size}

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        verified = list(pool.map(fetch, config["files"]))
    _write_receipt(
        "logs/deploy-20261030/checkpoints.json",
        {
            "pi05_revision": config["pi05_revision"],
            "sam3_revision": config["sam3_revision"],
            "files": verified,
        },
    )
    logger.info("Verified all %s model/tokenizer/statistics files", len(verified))


def assets(config: dict[str, Any], *, check_only: bool) -> None:
    """Download the immutable revision; LIBERO's existing CLI owns package links."""
    item = config["assets"]
    destination = workspace_path(item["path"])
    if not check_only:
        from huggingface_hub import snapshot_download

        snapshot_download(
            item["repo"],
            repo_type="dataset",
            revision=item["revision"],
            local_dir=destination,
        )
    paths = [
        path
        for path in destination.rglob("*")
        if path.is_file() and ".cache" not in path.relative_to(destination).parts
    ]
    if len(paths) != item["file_count"] or any(
        not path.resolve().is_relative_to(ROOT) for path in paths
    ):
        raise ValueError(
            "LIBERO assets are incomplete or resolve outside this workspace"
        )
    _write_receipt(
        "logs/deploy-20261030/assets-provenance.json",
        {
            "root": str(destination),
            "revision": item["revision"],
            "file_count": len(paths),
            "all_assets_inside_workspace": True,
        },
    )
    logger.info(
        "LIBERO assets present: %s files at pinned revision %s",
        len(paths),
        item["revision"],
    )


def memory(config: dict[str, Any], directory: Path | None, *, check_only: bool) -> None:
    """Fetch the original corpus on Linux or Windows without POSIX lock dependencies."""
    item = config["memory"]
    manifest = json.loads(Path(__file__).with_name(item["manifest"]).read_text("utf-8"))
    destination = workspace_path(directory or item["path"])
    files = manifest["files"]
    for name in files:
        relative = PurePosixPath(name)
        if (
            relative.is_absolute()
            or ".." in relative.parts
            or "\\" in name
            or ":" in name
        ):
            raise ValueError(f"invalid original-memory manifest path: {name}")
        if not (destination / name).resolve().is_relative_to(destination):
            raise ValueError(f"memory path escapes its corpus: {name}")
    if (manifest["repo"], manifest["revision"], manifest["version"]) != (
        item["repo"],
        item["revision"],
        item["version"],
    ):
        raise ValueError("memory manifest identity differs from resource configuration")
    import httpx

    base = f"https://huggingface.co/datasets/{item['repo']}/resolve/{item['revision']}/libero/{item['version']}/"
    with httpx.Client(timeout=120, follow_redirects=True) as client:

        def fetch(entry: tuple[str, str]) -> None:
            name, expected = entry
            target = destination / name
            if target.is_file():
                if sha256(target) != expected:
                    raise ValueError(
                        f"modified original-memory file; preserved: {target}"
                    )
                return
            if check_only:
                raise ValueError(f"missing original-memory file: {target}")
            target.parent.mkdir(parents=True, exist_ok=True)
            data = client.get(base + name).raise_for_status().content
            if hashlib.sha256(data).hexdigest() != expected:
                raise ValueError(f"original-memory download hash mismatch: {name}")
            partial = workspace_path(target.with_name(target.name + ".partial"))
            if not partial.is_relative_to(destination):
                raise ValueError(f"memory partial file escapes its corpus: {name}")
            partial.write_bytes(data)
            partial.replace(target)

        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            list(pool.map(fetch, files.items()))
    actual = {
        path.relative_to(destination).as_posix()
        for path in destination.rglob("*")
        if path.is_file()
    }
    if actual != set(files):
        raise ValueError(
            "unexpected files in original memory; use a clean corpus directory"
        )
    _write_receipt(
        "logs/harness-vla-baseline/memory/baseline-provenance.json",
        manifest,
    )
    logger.info(
        "Verified %s unchanged memory files from %s", len(files), item["revision"]
    )


def main(argv: list[str] | None = None) -> int:
    """Prepare immutable inputs; this command never runs exploration or an editor."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--group", choices=("models", "assets", "memory"), required=True
    )
    parser.add_argument("--memory-dir", type=Path)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args(argv)
    if args.group != "memory" and str(ROOT) != "/home2/wanquan/RPent":
        raise ValueError("this GPU server profile requires /home2/wanquan/RPent")
    init_output_dir(workspace_path("logs/deploy-20261030"))
    config = json.loads(CONFIG.read_text("utf-8"))
    if args.group == "memory":
        memory(config, args.memory_dir, check_only=args.check_only)
    elif args.group == "models":
        models(config, check_only=args.check_only)
    else:
        assets(config, check_only=args.check_only)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
