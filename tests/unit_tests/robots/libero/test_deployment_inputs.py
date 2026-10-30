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

"""Deployment inputs remain immutable and confined to the selected checkout."""

import hashlib
import json
from contextlib import nullcontext
from types import SimpleNamespace

import pytest

from robots.libero.deployment import fetch_artifacts as fetch


def test_workspace_destination_cannot_escape_checkout(tmp_path, monkeypatch):
    monkeypatch.setattr(fetch, "ROOT", tmp_path)
    assert fetch.workspace_path("checkpoints/model") == tmp_path / "checkpoints/model"
    with pytest.raises(ValueError, match="leaves the RPent workspace"):
        fetch.workspace_path("../other/model")


@pytest.fixture
def original_memory(tmp_path, monkeypatch):
    corpus = tmp_path / "memory"
    corpus.mkdir()
    data = b"# Original memory\n"
    (corpus / "MEMORY.md").write_bytes(data)
    manifest = {
        "repo": "fixture",
        "revision": "fixed",
        "version": "original",
        "files": {"MEMORY.md": hashlib.sha256(data).hexdigest()},
    }
    (tmp_path / "memory-manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    monkeypatch.setattr(fetch, "ROOT", tmp_path)
    monkeypatch.setattr(fetch, "__file__", str(tmp_path / "fetch.py"))

    def unexpected_request(*args, **kwargs):
        raise AssertionError("verification must not make network requests")

    import httpx

    monkeypatch.setattr(
        httpx,
        "Client",
        lambda **kwargs: nullcontext(SimpleNamespace(get=unexpected_request)),
    )
    return corpus, {
        "memory": {**manifest, "path": "memory", "manifest": "memory-manifest.json"}
    }


def test_check_only_verifies_original_memory_and_preserves_modified_files(
    original_memory,
):
    corpus, config = original_memory
    fetch.memory(config, None, check_only=True)
    (corpus / "MEMORY.md").write_text("Changed memory", encoding="utf-8")
    with pytest.raises(ValueError, match="modified original-memory file"):
        fetch.memory(config, None, check_only=True)
    assert (corpus / "MEMORY.md").read_text("utf-8") == "Changed memory"


def test_extra_files_cannot_be_claimed_as_original_memory(original_memory):
    corpus, config = original_memory
    (corpus / "extra.md").write_text("Unrecorded experience", encoding="utf-8")
    with pytest.raises(ValueError, match="unexpected files"):
        fetch.memory(config, None, check_only=True)
    assert (corpus / "extra.md").is_file()
