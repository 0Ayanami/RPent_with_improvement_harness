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

"""Content-verified memory snapshots and atomic published-corpus updates."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import stat
import tempfile
import time
from collections.abc import Mapping
from dataclasses import dataclass
from dataclasses import field as dataclass_field
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from rpent.evaluation.result import write_json_atomic
from rpent.memory.manager import MemoryManager, _render, _split_frontmatter, _validate

_LAYERS = {"global", "task-family", "task-specific", "results"}
_METADATA_FIELDS = {
    "id",
    "scope",
    "kind",
    "title",
    "applies_when",
    "confidence",
    "evidence",
    "related",
    "suite",
    "regime",
    "task_id",
    "task_language",
}
_MUTABLE_FIELDS = {"title", "applies_when", "evidence", "related"}
_EVIDENCE_FIELDS = {
    "cells",
    "attempts",
    "solved_seeds",
    "failed_seeds",
    "contradicted_by",
}
_HASH = re.compile(r"sha256:[0-9a-f]{64}\Z")
_SNAPSHOT_ID = re.compile(r"snapshot-[0-9a-f]{64}\Z")
_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*\Z")
_RESERVED = {"CON", "PRN", "AUX", "NUL"} | {
    f"{prefix}{number}" for prefix in ("COM", "LPT") for number in range(1, 10)
}
_TASK_RECORD = re.compile(
    r"task-specific/([A-Za-z0-9][A-Za-z0-9_-]*_t(?:0|[1-9][0-9]*)"
    r"_s(?:0|[1-9][0-9]*))(_recipe\.jsonl|\.json)\Z"
)


def _sha256(content: bytes) -> str:
    return "sha256:" + hashlib.sha256(content).hexdigest()


def _json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def _leaf_path(value: str) -> str:
    parts = value.split("/")
    if (
        len(parts) != 2
        or parts[0] not in {"global", "task-family"}
        or not parts[1].endswith(".md")
    ):
        raise ValueError("edit path must be global/<id>.md or task-family/<id>.md")
    memory_id = parts[1][:-3]
    if (
        not _IDENTIFIER.fullmatch(memory_id)
        or memory_id.endswith(".")
        or memory_id.split(".", 1)[0].upper() in _RESERVED
    ):
        raise ValueError("memory id must be a portable filename identifier")
    return value


def _validate_leaf(
    path: str, metadata: dict[str, Any], body: str, *, new: bool
) -> None:
    """Apply the existing schema plus strict checks for a generated leaf."""
    for field in ("scope", "confidence"):
        if not isinstance(metadata.get(field), str):
            raise ValueError(f"{field} must be a string")
    if "kind" in metadata and not isinstance(metadata["kind"], str):
        raise ValueError("kind must be a string")
    _validate(metadata)
    scope, filename = path.split("/")
    if metadata.get("scope") != scope or metadata.get("id") != filename[:-3]:
        raise ValueError("leaf scope/id must match its directory and filename")
    if not body.strip():
        raise ValueError("memory body must not be empty")
    if new and set(metadata) - _METADATA_FIELDS:
        raise ValueError(
            f"unsupported leaf metadata: {sorted(set(metadata) - _METADATA_FIELDS)}"
        )
    for field in (
        "id",
        "scope",
        "confidence",
        "title",
        "applies_when",
        "suite",
        "regime",
        "task_language",
    ):
        if field in metadata and (
            not isinstance(metadata[field], str) or not metadata[field].strip()
        ):
            raise ValueError(f"{field} must be a non-empty string")
    if "task_id" in metadata and (
        type(metadata["task_id"]) not in (str, int) or str(metadata["task_id"]) == ""
    ):
        raise ValueError("task_id must be a non-empty string or integer")
    if "related" in metadata and (
        not isinstance(metadata["related"], list)
        or any(not isinstance(item, str) for item in metadata["related"])
    ):
        raise ValueError("related must be a list of strings")
    evidence = metadata["evidence"]
    if new and set(evidence) - _EVIDENCE_FIELDS:
        raise ValueError("unsupported evidence metadata")
    if any(not isinstance(cell, str) or not cell.strip() for cell in evidence["cells"]):
        raise ValueError("evidence.cells must contain non-empty strings")
    if "attempts" in evidence and (
        type(evidence["attempts"]) is not int or evidence["attempts"] < 0
    ):
        raise ValueError("evidence.attempts must be a non-negative integer")
    for field in ("solved_seeds", "failed_seeds", "contradicted_by"):
        if field in evidence and (
            not isinstance(evidence[field], list)
            or any(type(item) not in (str, int) for item in evidence[field])
        ):
            raise ValueError(f"evidence.{field} must be a list of strings or integers")


class MemoryEdit(BaseModel):
    """One checked storage edit; higher layers own hypotheses and qualification.

    Hashes address the complete source file bytes, including frontmatter.
    Revisions preserve identity and confidence metadata. Narrowing a scope
    requires explicit applicability text; its behavioral validity is evaluated
    by the caller, separately from storage validation.
    """

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    schema_version: Literal[1] = 1
    operation: Literal["add", "revise", "narrow_scope", "retire"]
    path: str
    expected_old_hash: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    body: str | None = None

    @field_validator("schema_version", mode="before")
    @classmethod
    def validate_version(cls, value: Any) -> int:
        """Reject booleans, coercions, and unsupported schema versions."""
        if type(value) is not int or value != 1:
            raise ValueError("unsupported memory edit schema version")
        return value

    @field_validator("path")
    @classmethod
    def validate_path(cls, value: str) -> str:
        """Accept only a portable published leaf path."""
        return _leaf_path(value)

    @field_validator("expected_old_hash")
    @classmethod
    def validate_hash(cls, value: str | None) -> str | None:
        """Require a full, algorithm-qualified SHA-256 digest."""
        if value is not None and not _HASH.fullmatch(value):
            raise ValueError(
                "expected_old_hash must be sha256:<64 lowercase hex digits>"
            )
        return value

    @model_validator(mode="after")
    def validate_operation(self) -> MemoryEdit:
        """Validate the metadata and payload owned by each edit operation."""
        if self.operation == "add":
            if self.expected_old_hash is not None:
                raise ValueError("add must not carry expected_old_hash")
            if self.body is None:
                raise ValueError("add requires a body")
            _validate_leaf(self.path, self.metadata, self.body, new=True)
        else:
            if self.expected_old_hash is None:
                raise ValueError("existing-leaf edits require expected_old_hash")
            if self.operation == "retire":
                if self.metadata or self.body is not None:
                    raise ValueError("retire cannot replace metadata or body")
            else:
                if set(self.metadata) - _MUTABLE_FIELDS:
                    raise ValueError(
                        "revision metadata may only change title, applies_when, evidence, related"
                    )
                if not self.metadata and self.body is None:
                    raise ValueError("revision requires metadata or a body")
                if self.body is not None and not self.body.strip():
                    raise ValueError("memory body must not be empty")
                if self.operation == "narrow_scope" and not self.metadata.get(
                    "applies_when"
                ):
                    raise ValueError("narrow_scope requires applies_when")
        return self


@dataclass(frozen=True)
class MemorySnapshot:
    """Published corpus identity; content is rechecked when loaded or applied."""

    snapshot_id: str
    corpus_dir: Path
    content_hash: str
    files: Mapping[str, str]
    parent_id: str | None = None
    edit: MemoryEdit | None = None
    edits: tuple[MemoryEdit, ...] = ()
    task_records: Mapping[str, str] = dataclass_field(
        default_factory=lambda: MappingProxyType({})
    )


class _Manifest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    schema_version: Literal[1] = 1
    snapshot_id: str
    content_hash: str
    files: dict[str, str]
    parent_id: str | None = None
    edit: MemoryEdit | None = None


class _BatchManifest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    schema_version: Literal[2] = 2
    snapshot_id: str
    content_hash: str
    files: dict[str, str]
    parent_id: str
    edits: list[MemoryEdit] = Field(default_factory=list)
    task_records: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_update(self) -> _BatchManifest:
        if not _SNAPSHOT_ID.fullmatch(self.parent_id):
            raise ValueError("invalid snapshot parent id")
        if bool(self.edits) == bool(self.task_records):
            raise ValueError("snapshot requires either edits or task records")
        paths = [edit.path for edit in self.edits]
        if len({path.casefold() for path in paths}) != len(paths):
            raise ValueError("duplicate edit paths, including case aliases")
        if self.task_records:
            _validate_task_record_paths(self.task_records)
            if any(
                not _HASH.fullmatch(digest) or self.files.get(path) != digest
                for path, digest in self.task_records.items()
            ):
                raise ValueError("task record source hash differs from snapshot files")
        return self


def _validate_task_record_paths(records: Mapping[str, Any]) -> None:
    """Require native audit/recipe pairs without admitting a model edit surface."""
    if any(not isinstance(path, str) for path in records):
        raise ValueError("task record paths must be strings")
    if len({path.casefold() for path in records}) != len(records):
        raise ValueError("duplicate task record paths, including case aliases")
    for path in records:
        match = _TASK_RECORD.fullmatch(path)
        if match is None:
            raise ValueError(
                "task record path must be task-specific/<suite>_t<task>_s<seed>.json "
                "or its _recipe.jsonl pair"
            )
        stem = match[1]
        if not all(
            f"task-specific/{stem}{suffix}" in records
            for suffix in (".json", "_recipe.jsonl")
        ):
            raise ValueError(f"task record requires its audit/recipe pair: {path}")


def _reject_link(path: Path) -> None:
    if path.is_symlink() or (
        path.exists()
        and getattr(path.stat(follow_symlinks=False), "st_file_attributes", 0)
        & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    ):
        raise ValueError(f"symlinks and directory junctions are not allowed: {path}")


def _checked_path(path: str | Path) -> Path:
    absolute = Path(path).absolute()
    for component in (*reversed(absolute.parents), absolute):
        _reject_link(component)
    return absolute.resolve()


def _scan(corpus: Path, *, importing: bool = False) -> dict[str, str]:
    files: dict[str, str] = {}
    names: set[str] = set()

    def visit(directory: Path) -> None:
        for path in sorted(directory.iterdir()):
            _reject_link(path)
            relative = path.relative_to(corpus)
            visible = relative.parts[0] in _LAYERS or (
                len(relative.parts) == 1
                and path.is_file()
                and path.name.endswith(".md")
            )
            if not visible:
                if importing:
                    continue
                raise ValueError(f"unexpected unpublished snapshot path: {relative}")
            if not path.resolve().is_relative_to(corpus):
                raise ValueError(f"snapshot path escapes corpus: {relative}")
            folded = relative.as_posix().casefold()
            if folded in names:
                raise ValueError(f"snapshot paths collide on Windows: {relative}")
            names.add(folded)
            if path.is_dir():
                visit(path)
            elif path.is_file():
                files[relative.as_posix()] = _sha256(path.read_bytes())
            else:
                raise ValueError(f"snapshot path is not a regular file: {relative}")

    _reject_link(corpus)
    visit(corpus)
    return dict(sorted(files.items()))


class SnapshotStore:
    """Publish immutable-by-contract corpora with manifests outside the corpus.

    Snapshots use independent copies, never shared writable links. Verification
    detects external mutation; filesystem permissions remain the caller's
    responsibility. The legacy single-edit API keeps its baseline restriction;
    batch edits and trusted task records can build on any verified snapshot.
    """

    def __init__(self, root: str | Path) -> None:
        self.root = _checked_path(root)

    def capture(self, source: str | Path) -> MemorySnapshot:
        """Import a baseline, preserving published artifacts and root Markdown."""
        source_dir = _checked_path(source)
        if not source_dir.is_dir():
            raise ValueError(f"memory corpus is not a directory: {source_dir}")
        if self.root.is_relative_to(source_dir):
            raise ValueError("snapshot store must be outside the source corpus")
        files = _scan(source_dir, importing=True)
        self._validate_corpus(source_dir)
        return self._publish(source_dir, files, parent_id=None, edit=None)

    def apply(self, baseline: MemorySnapshot | str, edit: MemoryEdit) -> MemorySnapshot:
        """Create a candidate without mutating its verified baseline."""
        parent = self.verify(baseline)
        if parent.parent_id is not None:
            raise ValueError(
                "a candidate must contain only one edit relative to a baseline"
            )
        # Revalidate even a caller-mutated model or a model created with construct().
        edit = MemoryEdit.model_validate(edit.model_dump())
        self._check_edits(parent, (edit,))
        return self._publish(
            parent.corpus_dir, parent.files, parent_id=parent.snapshot_id, edit=edit
        )

    def apply_many(
        self,
        parent: MemorySnapshot | str,
        edits: list[MemoryEdit] | tuple[MemoryEdit, ...],
    ) -> MemorySnapshot:
        """Atomically apply distinct leaf edits against one verified parent.

        Every source hash is checked against the same parent, even when the
        parent is a previous candidate. An empty batch returns that parent.
        Publication never exposes a partial batch or mutates its source.
        """
        baseline = self.verify(parent)
        if not isinstance(edits, (list, tuple)):
            raise ValueError("edits must be a list or tuple of MemoryEdit records")
        checked = tuple(MemoryEdit.model_validate(edit.model_dump()) for edit in edits)
        self._check_edits(baseline, checked)
        if not checked:
            return baseline
        return self._publish(
            baseline.corpus_dir,
            baseline.files,
            parent_id=baseline.snapshot_id,
            edit=None,
            edits=checked,
        )

    def publish_task_records(
        self, parent: MemorySnapshot | str, records: dict[str, bytes]
    ) -> MemorySnapshot:
        """Publish trusted native audit/recipe pairs without changing old records.

        The caller must export these bytes from a natively successful episode
        and include its trial/task/seed provenance in the audit. This storage
        boundary checks paired native filenames, bytes and hashes; the robot
        adapter owns native success and record schemas. Model output must use
        MemoryEdit instead and cannot reach this task-specific write surface.

        Keys are flat ``task-specific/<suite>_t<task>_s<seed>.json`` paths and
        their matching ``_recipe.jsonl`` paths. Existing identical bytes are
        idempotent; different bytes or case aliases are rejected. Empty input
        returns the verified parent.
        """
        baseline = self.verify(parent)
        if not isinstance(records, dict):
            raise ValueError("task records must be a path-to-bytes dictionary")
        records = dict(records)
        _validate_task_record_paths(records)
        if any(type(content) is not bytes for content in records.values()):
            raise ValueError("task record content must be bytes")
        aliases = {path.casefold(): path for path in baseline.files}
        changed = False
        for path, content in records.items():
            known = aliases.get(path.casefold())
            if known is not None and known != path:
                raise ValueError(f"task record path aliases existing file: {path}")
            previous = baseline.files.get(path)
            if previous is not None and previous != _sha256(content):
                raise ValueError(
                    f"task record already exists with different bytes: {path}"
                )
            changed |= previous is None
        if not changed:
            return baseline
        return self._publish(
            baseline.corpus_dir,
            baseline.files,
            parent_id=baseline.snapshot_id,
            edit=None,
            records=records,
        )

    @staticmethod
    def _check_edits(parent: MemorySnapshot, edits: tuple[MemoryEdit, ...]) -> None:
        paths = [edit.path for edit in edits]
        if len({path.casefold() for path in paths}) != len(paths):
            raise ValueError("duplicate edit paths, including case aliases")
        aliases = {path.casefold(): path for path in parent.files}
        for edit in edits:
            known = aliases.get(edit.path.casefold())
            if known is not None and known != edit.path:
                raise ValueError(f"edit path aliases existing file: {edit.path}")
            old_hash = parent.files.get(edit.path)
            if edit.operation == "add":
                if old_hash is not None:
                    raise ValueError(f"add target already exists: {edit.path}")
            elif old_hash != edit.expected_old_hash:
                raise ValueError(f"stale or missing source hash: {edit.path}")

    def load(self, snapshot_id: str) -> MemorySnapshot:
        """Load a published manifest and verify every file, including MEMORY.md."""
        if not _SNAPSHOT_ID.fullmatch(snapshot_id):
            raise ValueError("invalid snapshot id")
        directory = _checked_path(self.root / snapshot_id)
        manifest_path = directory / "manifest.json"
        _reject_link(manifest_path)
        data = manifest_path.read_bytes()
        parsed = json.loads(data)
        manifest_type = (
            _BatchManifest
            if isinstance(parsed, dict) and parsed.get("schema_version") == 2
            else _Manifest
        )
        manifest = manifest_type.model_validate_json(data)
        payload = manifest.model_dump(mode="json", exclude={"snapshot_id"})
        identity = "snapshot-" + hashlib.sha256(_json_bytes(payload)).hexdigest()
        if identity != snapshot_id or manifest.snapshot_id != snapshot_id:
            raise ValueError("snapshot manifest identity mismatch")
        corpus = _checked_path(directory / "corpus")
        actual = _scan(corpus)
        if (
            actual != manifest.files
            or _sha256(_json_bytes(actual)) != manifest.content_hash
        ):
            raise ValueError("snapshot corpus hash mismatch")
        if isinstance(manifest, _Manifest) and (
            (manifest.parent_id is None) != (manifest.edit is None)
        ):
            raise ValueError("snapshot parent and edit must be recorded together")
        self._validate_corpus(corpus)
        return MemorySnapshot(
            snapshot_id=snapshot_id,
            corpus_dir=corpus,
            content_hash=manifest.content_hash,
            files=MappingProxyType(actual),
            parent_id=manifest.parent_id,
            edit=manifest.edit if isinstance(manifest, _Manifest) else None,
            edits=tuple(manifest.edits) if isinstance(manifest, _BatchManifest) else (),
            task_records=MappingProxyType(
                manifest.task_records if isinstance(manifest, _BatchManifest) else {}
            ),
        )

    def verify(self, snapshot: MemorySnapshot | str) -> MemorySnapshot:
        """Recheck a snapshot and reject mismatched in-memory identity fields."""
        loaded = self.load(
            snapshot if isinstance(snapshot, str) else snapshot.snapshot_id
        )
        if not isinstance(snapshot, str) and loaded != snapshot:
            raise ValueError("snapshot object does not match the stored manifest")
        return loaded

    @staticmethod
    def _validate_corpus(corpus: Path) -> None:
        problems = MemoryManager(corpus).validate()
        if problems:
            raise ValueError("invalid memory corpus: " + "; ".join(problems))

    def _publish(
        self,
        source: Path,
        files: Mapping[str, str],
        *,
        parent_id: str | None,
        edit: MemoryEdit | None,
        edits: tuple[MemoryEdit, ...] = (),
        records: Mapping[str, bytes] | None = None,
    ) -> MemorySnapshot:
        _checked_path(self.root)
        self.root.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix=".staging-", dir=self.root)).resolve()
        try:
            corpus = staging / "corpus"
            corpus.mkdir()
            for relative, expected_hash in files.items():
                original = source / PurePosixPath(relative)
                _checked_path(original)
                payload = original.read_bytes()
                if _sha256(payload) != expected_hash:
                    raise ValueError(
                        f"source changed during snapshot capture: {relative}"
                    )
                destination = corpus / PurePosixPath(relative)
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(payload)
            if _scan(source, importing=parent_id is None) != files:
                raise ValueError("source changed during snapshot capture")
            changes = (edit,) if edit is not None else edits
            if changes:
                self._apply_edits(corpus, changes)
            for relative, content in (records or {}).items():
                destination = corpus / PurePosixPath(relative)
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(content)
            self._validate_corpus(corpus)
            actual = _scan(corpus)
            payload: dict[str, Any] = {
                "content_hash": _sha256(_json_bytes(actual)),
                "files": actual,
                "parent_id": parent_id,
            }
            if edits or records:
                payload.update(
                    schema_version=2,
                    edits=[change.model_dump(mode="json") for change in edits],
                    task_records={
                        path: _sha256(content)
                        for path, content in (records or {}).items()
                    },
                )
            else:
                payload.update(
                    schema_version=1,
                    edit=edit.model_dump(mode="json") if edit else None,
                )
            snapshot_id = "snapshot-" + hashlib.sha256(_json_bytes(payload)).hexdigest()
            write_json_atomic(
                staging / "manifest.json", {"snapshot_id": snapshot_id, **payload}
            )
            destination = self.root / snapshot_id
            _reject_link(destination)
            if destination.exists():
                return self.load(snapshot_id)
            for attempt in range(5):
                try:
                    os.rename(staging, destination)
                    break
                except FileExistsError:
                    return self.load(snapshot_id)
                except PermissionError as exc:
                    # Windows scanners can temporarily retain directory handles
                    # after the files close. Retry only those Windows errors;
                    # persistent access failures still abort publication.
                    if (
                        getattr(exc, "winerror", None) not in {5, 32, 33}
                        or attempt == 4
                    ):
                        raise
                    if destination.exists():
                        return self.load(snapshot_id)
                    time.sleep(0.025 * 2**attempt)
            return self.load(snapshot_id)
        finally:
            if staging.exists():
                if staging.parent != self.root or not staging.name.startswith(
                    ".staging-"
                ):
                    raise ValueError("refusing cleanup outside snapshot staging")
                shutil.rmtree(staging)

    @classmethod
    def _apply_edits(cls, corpus: Path, edits: tuple[MemoryEdit, ...]) -> None:
        index = corpus / "MEMORY.md"
        previous = index.read_bytes() if index.exists() else None
        manager = MemoryManager(corpus)
        generated_before = (
            index.read_text("utf-8") if manager.rebuild_index() is not None else None
        )
        for edit in edits:
            cls._apply_edit(corpus, edit)
        # Rebuild removes stale links even when retiring the last leaf.
        index.unlink(missing_ok=True)
        if manager.rebuild_index() is None:
            index.write_text("# Layered memory index\n\n_(none)_\n", encoding="utf-8")
        generated_after = index.read_bytes()
        if previous is None:
            return
        if all(
            edit.operation in {"revise", "narrow_scope"}
            and not ({"title", "applies_when"} & edit.metadata.keys())
            for edit in edits
        ):
            index.write_bytes(previous)
            return
        if previous.decode("utf-8").replace("\r\n", "\n") == generated_before:
            return
        # Merge changed rows into a hand-maintained index. Each lookup uses the
        # complete final generated index, while its custom prose accumulates.
        for edit in edits:
            index.write_bytes(generated_after)
            cls._preserve_index(corpus, edit, previous, generated_before)
            previous = index.read_bytes()

    @staticmethod
    def _preserve_index(
        corpus: Path,
        edit: MemoryEdit,
        previous: bytes | None,
        generated_before: str | None,
    ) -> None:
        """Update a derived index without discarding hand-maintained instructions."""
        if previous is None:
            return
        index = corpus / "MEMORY.md"
        if edit.operation in {"revise", "narrow_scope"} and not (
            {"title", "applies_when"} & edit.metadata.keys()
        ):
            index.write_bytes(previous)
            return
        old_text = previous.decode("utf-8").replace("\r\n", "\n")
        if old_text == generated_before:
            return
        new_text = index.read_text("utf-8")
        new_row = next(
            (
                line + "\n"
                for line in new_text.splitlines()
                if f"]({edit.path})" in line
            ),
            "",
        )
        # A simple index row has one link followed by optional applicability
        # prose. Other custom formats need explicit corpus preparation rather
        # than silently rewriting unrelated instructions or links.
        row = re.compile(
            r"(?m)^[-*] \[[^\]\n]+\]\(" + re.escape(edit.path) + r"\)[^\n\[]*(?:\n|$)"
        )
        remaining, count = row.subn(lambda match: new_row, old_text)
        if not count:
            if f"]({edit.path})" in old_text:
                raise ValueError(
                    "custom MEMORY.md link format requires rebuilding its index before applying this edit"
                )
            if new_row:
                remaining = old_text.rstrip("\n") + "\n\n" + new_row
        index.write_text(remaining, encoding="utf-8", newline="\n")

    @staticmethod
    def _apply_edit(corpus: Path, edit: MemoryEdit) -> None:
        destination = corpus / PurePosixPath(edit.path)
        if edit.operation == "retire":
            destination.unlink()
            return
        if edit.operation == "add":
            if destination.exists():
                raise ValueError(f"add target already exists: {edit.path}")
            metadata, body = edit.metadata, edit.body
        else:
            metadata, previous_body = _split_frontmatter(destination)
            metadata = {**metadata, **edit.metadata}
            body = edit.body if edit.body is not None else previous_body
        assert body is not None
        _validate_leaf(edit.path, metadata, body, new=edit.operation == "add")
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(_render(metadata, body), encoding="utf-8", newline="\n")


__all__ = ["MemoryEdit", "MemorySnapshot", "SnapshotStore"]
