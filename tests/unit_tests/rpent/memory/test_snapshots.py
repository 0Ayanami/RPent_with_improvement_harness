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

"""Offline contracts for immutable corpora, checked edits, and read evidence."""

from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import replace
from pathlib import Path

import pytest
import yaml

from rpent.memory import MemoryManager, snapshots
from rpent.memory.snapshots import MemoryEdit, SnapshotStore
from rpent.tools.common import CommonTools


def _hash(payload: bytes) -> str:
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _metadata(memory_id: str = "rule", *, scope: str = "global") -> dict:
    metadata = {
        "id": memory_id,
        "scope": scope,
        "title": "Check the grasp",
        "applies_when": "before transport",
        "confidence": "single-shot",
        "evidence": {"cells": ["libero10_task_t2_s0"], "attempts": 1},
        "related": [],
    }
    if scope == "global":
        metadata["kind"] = "strategy"
    else:
        metadata.update(
            suite="libero10", regime="task", task_id=2, task_language="Move the pan"
        )
    return metadata


def _leaf(
    path: Path, metadata: dict | None = None, body: str = "Check before moving.\n"
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "---\n" + yaml.safe_dump(metadata or _metadata()) + "---\n" + body
    path.write_text(text, encoding="utf-8", newline="\n")


@pytest.fixture
def corpus(tmp_path: Path) -> Path:
    root = tmp_path / "source"
    _leaf(root / "global" / "rule.md")
    (root / "MEMORY.md").write_text("# Hand-curated index\n", encoding="utf-8")
    (root / "README.md").write_text("# Corpus instructions\n", encoding="utf-8")
    (root / "task-specific").mkdir()
    (root / "task-specific" / "task.json").write_text(
        '{"success": true}\n', encoding="utf-8"
    )
    (root / "task-specific" / "task_recipe.jsonl").write_text(
        '{"tool":"move"}\n', encoding="utf-8"
    )
    (root / "_internal").mkdir()
    (root / "_internal" / "secret.md").write_text("hidden rating", encoding="utf-8")
    return root


def test_capture_preserves_published_corpus_and_keeps_manifest_outside(
    corpus, tmp_path
):
    store = SnapshotStore(tmp_path / "snapshots")
    baseline = store.capture(corpus)

    assert baseline.parent_id is None
    assert baseline.edit is None
    assert baseline.files["MEMORY.md"] == _hash((corpus / "MEMORY.md").read_bytes())
    for name in baseline.files:
        assert (baseline.corpus_dir / name).read_bytes() == (corpus / name).read_bytes()
        assert not (baseline.corpus_dir / name).samefile(corpus / name)
    assert "README.md" in baseline.files
    assert "task-specific/task_recipe.jsonl" in baseline.files
    assert not (baseline.corpus_dir / "_internal").exists()
    assert not (baseline.corpus_dir / "manifest.json").exists()
    assert (baseline.corpus_dir.parent / "manifest.json").is_file()
    assert store.capture(corpus) == baseline
    assert store.verify(baseline) == baseline
    with pytest.raises(TypeError):
        baseline.files["changed.md"] = "bad"


def test_revise_copies_baseline_and_rebuilds_hashed_index(corpus, tmp_path):
    store = SnapshotStore(tmp_path / "snapshots")
    baseline = store.capture(corpus)
    original = (baseline.corpus_dir / "global/rule.md").read_bytes()
    edit = MemoryEdit(
        operation="revise",
        path="global/rule.md",
        expected_old_hash=baseline.files["global/rule.md"],
        metadata={"title": "Observe before transport"},
        body="Check the wrist view.\n",
    )
    candidate = store.apply(baseline, edit)

    assert candidate.parent_id == baseline.snapshot_id
    assert candidate.edit == edit
    assert candidate.snapshot_id != baseline.snapshot_id
    assert candidate.content_hash != baseline.content_hash
    assert (baseline.corpus_dir / "global/rule.md").read_bytes() == original
    assert (corpus / "global/rule.md").read_bytes() == original
    assert (candidate.corpus_dir / "task-specific/task.json").read_bytes() == (
        corpus / "task-specific/task.json"
    ).read_bytes()
    assert "Observe before transport" in (candidate.corpus_dir / "MEMORY.md").read_text(
        encoding="utf-8"
    )
    assert (
        (candidate.corpus_dir / "MEMORY.md")
        .read_text("utf-8")
        .startswith("# Hand-curated index\n")
    )
    assert candidate.files["MEMORY.md"] == _hash(
        (candidate.corpus_dir / "MEMORY.md").read_bytes()
    )
    assert not (candidate.corpus_dir / "global/rule.md").samefile(
        baseline.corpus_dir / "global/rule.md"
    )
    assert store.apply(baseline, edit) == candidate
    with pytest.raises(ValueError, match="only one edit"):
        store.apply(candidate, edit)


def test_add_narrow_and_retire_are_independent_candidates(corpus, tmp_path):
    store = SnapshotStore(tmp_path / "snapshots")
    baseline = store.capture(corpus)
    added = store.apply(
        baseline,
        MemoryEdit(
            operation="add",
            path="task-family/task-family_libero10_task_t2.md",
            metadata=_metadata("task-family_libero10_task_t2", scope="task-family"),
            body="Observe the pan.\n",
        ),
    )
    assert "task-family/task-family_libero10_task_t2.md" in added.files
    narrow = store.apply(
        baseline,
        MemoryEdit(
            operation="narrow_scope",
            path="global/rule.md",
            expected_old_hash=baseline.files["global/rule.md"],
            metadata={"applies_when": "before transport and object visibility is low"},
        ),
    )
    assert "object visibility is low" in (narrow.corpus_dir / "MEMORY.md").read_text(
        encoding="utf-8"
    )
    retired = store.apply(
        baseline,
        MemoryEdit(
            operation="retire",
            path="global/rule.md",
            expected_old_hash=baseline.files["global/rule.md"],
        ),
    )
    assert "global/rule.md" not in retired.files
    assert "rule.md" not in (retired.corpus_dir / "MEMORY.md").read_text(
        encoding="utf-8"
    )
    assert "global/rule.md" in store.verify(baseline).files


def test_body_revision_preserves_handwritten_index_and_other_rules(corpus, tmp_path):
    original_index = b"# Operating constraints\r\nAlways inspect wrist view.\r\n"
    (corpus / "MEMORY.md").write_bytes(original_index)
    store = SnapshotStore(tmp_path / "snapshots")
    baseline = store.capture(corpus)
    candidate = store.apply(
        baseline,
        MemoryEdit(
            operation="revise",
            path="global/rule.md",
            expected_old_hash=baseline.files["global/rule.md"],
            body="A revised rule.\n",
        ),
    )
    assert (candidate.corpus_dir / "MEMORY.md").read_bytes() == original_index
    assert {
        path for path in baseline.files if baseline.files[path] != candidate.files[path]
    } == {"global/rule.md"}


@pytest.mark.parametrize(
    "path",
    [
        "../global/rule.md",
        "/global/rule.md",
        "global/../rule.md",
        "global/sub/rule.md",
        "global\\rule.md",
        "global/C:rule.md",
        "task-specific/task.json",
        "_internal/rule.md",
        "MEMORY.md",
        "global/CON.md",
        "global/rule..md",
        "global/.md",
    ],
)
def test_edit_rejects_unsafe_or_frozen_paths(path):
    with pytest.raises(ValueError):
        MemoryEdit(operation="add", path=path, metadata=_metadata(), body="rule")


@pytest.mark.parametrize(
    "changes",
    [
        {"unsupported": True},
        {"schema_version": 2},
        {"schema_version": True},
        {"metadata": {**_metadata(), "id": "different"}},
        {"metadata": {**_metadata(), "scope": "task-family"}},
        {"metadata": {**_metadata(), "title": 1}},
        {"metadata": {**_metadata(), "kind": ["strategy"]}},
        {"metadata": {**_metadata(), "confidence": "qualified"}},
        {"metadata": {**_metadata(), "extra_protocol": "hidden"}},
        {"metadata": {**_metadata(), "evidence": {"cells": [1]}}},
        {"metadata": {**_metadata(), "evidence": {"cells": []}}},
        {
            "metadata": {
                **_metadata(),
                "evidence": {"cells": ["cell"], "arbitrary": True},
            }
        },
        {"body": "   "},
        {"expected_old_hash": "sha256:" + "0" * 64},
    ],
)
def test_new_leaf_schema_is_strict(changes):
    data = {
        "operation": "add",
        "path": "global/rule.md",
        "metadata": _metadata(),
        "body": "rule",
    }
    with pytest.raises(ValueError):
        MemoryEdit.model_validate({**data, **changes})


@pytest.mark.parametrize(
    "changes",
    [
        {"expected_old_hash": None},
        {"expected_old_hash": "not-a-hash"},
        {"metadata": {"id": "replacement"}},
        {"metadata": {"confidence": "verified"}},
        {"operation": "narrow_scope", "metadata": {}},
        {"operation": "retire", "body": "replacement"},
    ],
)
def test_existing_edit_requires_hash_and_preserves_identity(changes):
    data = {
        "operation": "revise",
        "path": "global/rule.md",
        "expected_old_hash": "sha256:" + "0" * 64,
        "body": "rule",
    }
    with pytest.raises(ValueError):
        MemoryEdit.model_validate({**data, **changes})


def test_stale_hash_and_duplicate_id_do_not_publish_candidates(corpus, tmp_path):
    store = SnapshotStore(tmp_path / "snapshots")
    baseline = store.capture(corpus)
    with pytest.raises(ValueError, match="source hash"):
        store.apply(
            baseline,
            MemoryEdit(
                operation="revise",
                path="global/rule.md",
                expected_old_hash="sha256:" + "0" * 64,
                body="change",
            ),
        )
    with pytest.raises(ValueError, match="duplicate id"):
        store.apply(
            baseline,
            MemoryEdit(
                operation="add",
                path="task-family/rule.md",
                metadata=_metadata(scope="task-family"),
                body="change",
            ),
        )
    assert [path.name for path in store.root.iterdir()] == [baseline.snapshot_id]
    assert store.verify(baseline) == baseline


@pytest.mark.parametrize(
    "tamper_path", ["global/rule.md", "MEMORY.md", "task-specific/task.json"]
)
def test_load_detects_any_corpus_tampering(corpus, tmp_path, tamper_path):
    store = SnapshotStore(tmp_path / "snapshots")
    baseline = store.capture(corpus)
    (baseline.corpus_dir / tamper_path).write_text("tampered", encoding="utf-8")
    with pytest.raises(ValueError, match="hash mismatch"):
        store.load(baseline.snapshot_id)
    with pytest.raises(ValueError):
        store.capture(corpus)


def test_load_rejects_manifest_tampering_and_forged_snapshot(corpus, tmp_path):
    store = SnapshotStore(tmp_path / "snapshots")
    baseline = store.capture(corpus)
    with pytest.raises(ValueError, match="object does not match"):
        store.verify(replace(baseline, content_hash="sha256:" + "0" * 64))
    manifest_path = baseline.corpus_dir.parent / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["parent_id"] = "forged"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="identity mismatch"):
        store.load(baseline.snapshot_id)
    with pytest.raises(ValueError, match="invalid snapshot id"):
        store.load("../other")


def test_unpublished_added_files_and_symlink_sources_are_denied(corpus, tmp_path):
    store = SnapshotStore(tmp_path / "snapshots")
    baseline = store.capture(corpus)
    hidden = baseline.corpus_dir / "_internal"
    hidden.mkdir()
    with pytest.raises(ValueError, match="unpublished"):
        store.verify(baseline)
    outside = tmp_path / "outside.md"
    outside.write_text("secret", encoding="utf-8")
    link = corpus / "global" / "linked.md"
    try:
        link.symlink_to(outside)
    except OSError:
        pytest.skip("symlink creation is unavailable on this Windows account")
    with pytest.raises(ValueError, match="symlinks"):
        store.capture(corpus)


def test_capture_preserves_plain_notes_and_rejects_invalid_frontmatter(
    corpus, tmp_path
):
    (corpus / "global" / "notes.md").write_text("A hand-curated note", encoding="utf-8")
    store = SnapshotStore(tmp_path / "snapshots")
    assert "global/notes.md" in store.capture(corpus).files
    _leaf(corpus / "global" / "invalid.md", _metadata("wrong_id"))
    with pytest.raises(ValueError, match="does not match filename"):
        store.capture(corpus)


def test_publish_failure_cleans_staging(corpus, tmp_path, monkeypatch):
    def fail_write(*args, **kwargs):
        raise OSError("disk error")

    monkeypatch.setattr("rpent.memory.snapshots.write_json_atomic", fail_write)
    store = SnapshotStore(tmp_path / "snapshots")
    with pytest.raises(OSError, match="disk error"):
        store.capture(corpus)
    assert list(store.root.iterdir()) == []


def test_read_audit_records_actual_file_and_truncated_payload(tmp_path):
    root = tmp_path / "memory"
    root.mkdir()
    path = root / "MEMORY.md"
    original = "规则\r\n".encode("utf-8") * 20
    path.write_bytes(original)
    reads = []
    manager = MemoryManager(root, read_audit=reads.append)
    result = CommonTools(memory=manager).read_text_file(path=str(path), max_chars=8)
    event = reads.pop()

    assert event["path"] == str(path.resolve())
    assert event["relative_path"] == "MEMORY.md"
    assert event["source_hash"] == _hash(original)
    assert event["provided_hash"] == _hash(result.data["content"].encode("utf-8"))
    assert event["source_chars"] == result.data["size"] == 60
    assert event["provided_chars"] == len(result.data["content"])
    assert event["truncated"] is True
    assert event["access_method"] == "file_tool"
    CommonTools(memory=manager).read_text_file(path=str(path))
    assert reads[-1]["truncated"] is False


def test_audit_uses_supplied_source_even_if_file_later_changes(tmp_path):
    path = tmp_path / "file.txt"
    path.write_text("later contents", encoding="utf-8")
    reads = []
    manager = MemoryManager(tmp_path / "memory", read_audit=reads.append)
    manager.record_read(
        path, "original", source_content="original", source_bytes=b"original"
    )
    assert reads[0]["source_hash"] == _hash(b"original")
    assert reads[0]["relative_path"] is None
    manager.record_read(path, "later contents")
    assert reads[-1]["source_hash"] == _hash(path.read_bytes())
    MemoryManager(tmp_path).record_read(tmp_path / "missing", "no audit requested")


def test_trial_roots_limit_read_listing_and_writes(tmp_path):
    root, output, hidden = (tmp_path / name for name in ("memory", "output", "meta"))
    root.mkdir()
    output.mkdir()
    hidden.mkdir()
    manager = MemoryManager(
        root, allowed_read_roots=[root, output], allowed_write_roots=[output]
    )
    assert manager.authorize_read(output / "state.json") == output / "state.json"
    assert manager.authorize_write(output / "result.json") == output / "result.json"
    for method, path in (
        (manager.authorize_read, hidden / "secret.md"),
        (manager.list_directory, tmp_path),
        (manager.authorize_write, hidden / "secret.md"),
        (manager.authorize_write, root / "MEMORY.md"),
        (manager.authorize_read, root / "_internal/hidden.md"),
    ):
        with pytest.raises(PermissionError):
            method(path)
    assert (
        MemoryManager(root).authorize_read(hidden / "secret.md") == hidden / "secret.md"
    )


def test_merge_reports_missing_posix_locking_without_creating_internals(
    tmp_path, monkeypatch
):
    monkeypatch.setitem(sys.modules, "fcntl", None)
    manager = MemoryManager(tmp_path / "memory")
    with pytest.raises(RuntimeError, match="POSIX file locking"):
        manager.merge_memory(cell_tag="cell", run_state_dir=tmp_path, solved=False)
    assert not (manager.root / "_internal").exists()


@pytest.mark.parametrize("persistent", [False, True])
def test_windows_snapshot_publication_retries_are_bounded(
    corpus, tmp_path, monkeypatch, persistent
):
    original = snapshots.os.rename
    attempts = []

    def rename(source, destination):
        attempts.append(1)
        if persistent or len(attempts) == 1:
            error = PermissionError("Windows directory handle is in use")
            error.winerror = 5
            raise error
        return original(source, destination)

    monkeypatch.setattr(snapshots.os, "rename", rename)
    monkeypatch.setattr(snapshots.time, "sleep", lambda delay: None)
    store = SnapshotStore(tmp_path / "snapshots")
    if persistent:
        with pytest.raises(PermissionError):
            store.capture(corpus)
        assert len(attempts) == 5
        assert not list(store.root.glob("snapshot-*"))
        assert not list(store.root.glob(".staging-*"))
    else:
        snapshot = store.capture(corpus)
        assert store.verify(snapshot) == snapshot
        assert len(attempts) == 2


def test_batch_edits_are_atomic_and_can_build_on_previous_candidates(corpus, tmp_path):
    store = SnapshotStore(tmp_path / "snapshots")
    baseline = store.capture(corpus)
    revision = MemoryEdit(
        operation="revise",
        path="global/rule.md",
        expected_old_hash=baseline.files["global/rule.md"],
        metadata={"title": "Observe first"},
        body="Inspect the wrist view.\n",
    )
    addition = MemoryEdit(
        operation="add",
        path="global/recovery.md",
        metadata=_metadata("recovery"),
        body="Recover a missed grasp.\n",
    )
    candidate = store.apply_many(baseline, [revision, addition])
    manifest = json.loads((candidate.corpus_dir.parent / "manifest.json").read_text())

    assert candidate.parent_id == baseline.snapshot_id
    assert candidate.edit is None
    assert candidate.edits == (revision, addition)
    assert manifest["schema_version"] == 2
    assert manifest["edits"] == [
        edit.model_dump(mode="json") for edit in candidate.edits
    ]
    assert "Observe first" in (candidate.corpus_dir / "MEMORY.md").read_text()
    assert "global/recovery.md" in (candidate.corpus_dir / "MEMORY.md").read_text()
    assert store.apply_many(baseline, [revision, addition]) == candidate
    assert store.apply_many(candidate, []) == candidate
    assert store.apply_many(candidate.snapshot_id, ()) == candidate
    assert "global/recovery.md" not in store.verify(baseline).files
    for path in ("global/rule.md", "task-specific/task.json", "README.md"):
        assert (baseline.corpus_dir / path).read_bytes() == (corpus / path).read_bytes()
    with pytest.raises(ValueError, match="source hash"):
        store.apply_many(candidate, [revision])

    next_revision = MemoryEdit(
        operation="revise",
        path=revision.path,
        expected_old_hash=candidate.files[revision.path],
        body="Inspect and confirm the grasp.\n",
    )
    following = store.apply_many(candidate, (next_revision,))
    assert following.parent_id == candidate.snapshot_id
    assert following.files[addition.path] == candidate.files[addition.path]
    assert following.files[revision.path] != candidate.files[revision.path]

    # Existing single-edit candidates can also enter the construction stream.
    legacy = store.apply(baseline, revision)
    assert store.apply_many(legacy, [addition]).parent_id == legacy.snapshot_id
    assert (
        json.loads((legacy.corpus_dir.parent / "manifest.json").read_text())[
            "schema_version"
        ]
        == 1
    )


@pytest.mark.parametrize("duplicate_path", ["global/rule.md", "global/RULE.md"])
def test_batch_rejects_duplicate_targets_before_publication(
    corpus, tmp_path, duplicate_path
):
    store = SnapshotStore(tmp_path / "snapshots")
    baseline = store.capture(corpus)
    edit = MemoryEdit(
        operation="revise",
        path="global/rule.md",
        expected_old_hash=baseline.files["global/rule.md"],
        body="Inspect first.\n",
    )
    duplicate = edit.model_copy(update={"path": duplicate_path})
    with pytest.raises(ValueError, match="duplicate edit paths"):
        store.apply_many(baseline, [edit, duplicate])
    assert [path.name for path in store.root.iterdir()] == [baseline.snapshot_id]


def test_batch_source_checks_and_final_validation_never_publish_partial_edits(
    corpus, tmp_path
):
    store = SnapshotStore(tmp_path / "snapshots")
    baseline = store.capture(corpus)
    revision = MemoryEdit(
        operation="revise",
        path="global/rule.md",
        expected_old_hash=baseline.files["global/rule.md"],
        body="New rule.\n",
    )
    added = MemoryEdit(
        operation="add",
        path="global/recovery.md",
        metadata=_metadata("recovery"),
        body="Recover carefully.\n",
    )
    stale = MemoryEdit(
        operation="retire",
        path="global/recovery.md",
        expected_old_hash="sha256:" + "0" * 64,
    )
    with pytest.raises(ValueError, match="source hash"):
        store.apply_many(baseline, [revision, stale])
    conflicting = MemoryEdit(
        operation="add",
        path="task-family/rule.md",
        metadata=_metadata(scope="task-family"),
        body="A duplicate identity is invalid.\n",
    )
    with pytest.raises(ValueError, match="duplicate id"):
        store.apply_many(baseline, [revision, added, conflicting])
    assert [path.name for path in store.root.iterdir()] == [baseline.snapshot_id]
    assert store.verify(baseline) == baseline


def test_batch_preserves_custom_index_prose_and_updates_every_changed_row(
    corpus, tmp_path
):
    _leaf(corpus / "global/retired.md", _metadata("retired"))
    (corpus / "MEMORY.md").write_text(
        "# Operating constraints\nKeep the wrist camera visible.\n\n"
        "- [Original](global/rule.md)\n- [Old advice](global/retired.md)\n",
        encoding="utf-8",
    )
    store = SnapshotStore(tmp_path / "snapshots")
    baseline = store.capture(corpus)
    candidate = store.apply_many(
        baseline,
        [
            MemoryEdit(
                operation="revise",
                path="global/rule.md",
                expected_old_hash=baseline.files["global/rule.md"],
                metadata={"title": "Check the view"},
            ),
            MemoryEdit(
                operation="add",
                path="global/recovery.md",
                metadata=_metadata("recovery"),
                body="Recover carefully.\n",
            ),
            MemoryEdit(
                operation="retire",
                path="global/retired.md",
                expected_old_hash=baseline.files["global/retired.md"],
            ),
        ],
    )
    index = (candidate.corpus_dir / "MEMORY.md").read_text()
    assert index.startswith("# Operating constraints\nKeep the wrist camera visible.\n")
    assert "[Check the view](global/rule.md)" in index
    assert index.count("global/recovery.md") == 1
    assert "global/retired.md" not in index
    assert "global/retired.md" in store.verify(baseline).files


def test_batch_body_changes_preserve_index_bytes(corpus, tmp_path):
    _leaf(corpus / "global/recovery.md", _metadata("recovery"))
    original = b"# Custom rules\r\nAlways inspect the wrist view.\r\n"
    (corpus / "MEMORY.md").write_bytes(original)
    store = SnapshotStore(tmp_path / "snapshots")
    baseline = store.capture(corpus)
    edits = [
        MemoryEdit(
            operation="revise",
            path=path,
            expected_old_hash=baseline.files[path],
            body=f"Updated advice for {path}.\n",
        )
        for path in ("global/rule.md", "global/recovery.md")
    ]
    candidate = store.apply_many(baseline, edits)
    assert (candidate.corpus_dir / "MEMORY.md").read_bytes() == original
    assert {
        path
        for path, digest in baseline.files.items()
        if candidate.files[path] != digest
    } == {edit.path for edit in edits}


def _task_records(tag="spatial_task_t0_s0"):
    return {
        f"task-specific/{tag}.json": b'{"trial_id":"final_task0","terminated":true}\n',
        f"task-specific/{tag}_recipe.jsonl": b'{"action":"move","steps":4}\n',
    }


def test_trusted_task_records_preserve_native_bytes_and_allow_continuous_builds(
    corpus, tmp_path
):
    store = SnapshotStore(tmp_path / "snapshots")
    baseline = store.capture(corpus)
    records = _task_records()
    published = store.publish_task_records(baseline, records)
    manifest = json.loads((published.corpus_dir.parent / "manifest.json").read_text())

    assert published.parent_id == baseline.snapshot_id
    assert published.edit is None and published.edits == ()
    assert published.task_records == {
        path: _hash(data) for path, data in records.items()
    }
    assert manifest["schema_version"] == 2
    assert manifest["task_records"] == dict(published.task_records)
    for path, data in records.items():
        assert (published.corpus_dir / path).read_bytes() == data
        assert path not in baseline.files
    for path in baseline.files:
        assert published.files[path] == baseline.files[path]
    with pytest.raises(TypeError):
        published.task_records["changed"] = "bad"
    assert store.publish_task_records(published, records) == published
    assert store.publish_task_records(published, {}) == published
    next_snapshot = store.apply_many(
        published,
        [
            MemoryEdit(
                operation="revise",
                path="global/rule.md",
                expected_old_hash=published.files["global/rule.md"],
                body="Use the successful technique.\n",
            )
        ],
    )
    final = store.publish_task_records(
        next_snapshot, _task_records("spatial_task_t1_s0")
    )
    for path, data in records.items():
        assert (final.corpus_dir / path).read_bytes() == data
    assert len(final.files) == len(baseline.files) + 4
    assert store.verify(baseline) == baseline


@pytest.mark.parametrize(
    "path",
    [
        "../task-specific/spatial_t0_s0.json",
        "task-specific/../spatial_t0_s0.json",
        "task-specific/spatial/t0/spatial_t0_s0.json",
        "task-specific\\spatial_t0_s0.json",
        "task-specific/spatial_t0_s0.md",
        "task-specific/spatial_t00_s0.json",
        "task-specific/spatial_t0_s01.json",
        "task-specific/spatial_t0_s0.json:stream",
        "task-specific/spatial_t0_s0_recipe.json",
        "task-specific/CON.json",
        "global/spatial_t0_s0.json",
        "MEMORY.md",
    ],
)
def test_task_record_publication_rejects_non_native_or_unsafe_paths(
    corpus, tmp_path, path
):
    store = SnapshotStore(tmp_path / "snapshots")
    baseline = store.capture(corpus)
    with pytest.raises(ValueError, match="task record path"):
        store.publish_task_records(baseline, {path: b"{}"})
    assert [path.name for path in store.root.iterdir()] == [baseline.snapshot_id]


def test_task_record_pairs_reject_overwrites_and_case_aliases(corpus, tmp_path):
    store = SnapshotStore(tmp_path / "snapshots")
    baseline = store.capture(corpus)
    records = _task_records()
    with pytest.raises(ValueError, match="audit/recipe pair"):
        store.publish_task_records(baseline, {next(iter(records)): b"{}"})
    with pytest.raises(ValueError, match="must be bytes"):
        store.publish_task_records(baseline, dict.fromkeys(records, "text"))
    aliases = {
        path.replace("spatial", "SPATIAL"): data for path, data in records.items()
    }
    with pytest.raises(ValueError, match="duplicate task record paths"):
        store.publish_task_records(baseline, {**records, **aliases})

    published = store.publish_task_records(baseline, records)
    with pytest.raises(ValueError, match="aliases existing file"):
        store.publish_task_records(published, aliases)
    changed = {**records, next(iter(records)): b'{"trial_id":"a different trial"}'}
    with pytest.raises(ValueError, match="different bytes"):
        store.publish_task_records(published, changed)
    assert store.verify(published) == published
    assert len(list(store.root.iterdir())) == 2


def test_added_memory_leaf_cannot_alias_existing_case(corpus, tmp_path):
    store = SnapshotStore(tmp_path / "snapshots")
    baseline = store.capture(corpus)
    edit = MemoryEdit(
        operation="add",
        path="global/RULE.md",
        metadata=_metadata("RULE"),
        body="A case alias must not replace a different leaf.\n",
    )
    with pytest.raises(ValueError, match="aliases existing file"):
        store.apply_many(baseline, [edit])
    with pytest.raises(ValueError, match="aliases existing file"):
        store.apply(baseline, edit)
    assert store.verify(baseline) == baseline


def test_task_record_source_hash_and_payload_tampering_are_rejected(corpus, tmp_path):
    store = SnapshotStore(tmp_path / "snapshots")
    baseline = store.capture(corpus)
    snapshot = store.publish_task_records(baseline, _task_records())
    manifest_path = snapshot.corpus_dir.parent / "manifest.json"
    original = manifest_path.read_bytes()
    manifest = json.loads(original)
    path = next(iter(manifest["task_records"]))
    manifest["task_records"][path] = "sha256:" + "0" * 64
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="source hash"):
        store.load(snapshot.snapshot_id)
    manifest_path.write_bytes(original)
    (snapshot.corpus_dir / path).write_bytes(b"changed audit")
    with pytest.raises(ValueError, match="corpus hash mismatch"):
        store.verify(snapshot)
