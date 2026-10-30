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

"""Helpers for converting values into pickle-safe wire formats."""

from __future__ import annotations

import dataclasses
from typing import Any


def strip_inline_images(value: Any) -> Any:
    """Copy transcript values while omitting bulky inline image payloads."""
    if isinstance(value, list):
        return [strip_inline_images(item) for item in value]
    if isinstance(value, dict):
        if value.get("type") == "image":
            return {"type": "image", "source": {"_omitted_for_transcript": True}}
        if value.get("type") in {"image_url", "input_image"}:
            return {
                "type": value["type"],
                "image_url": {"_omitted_for_transcript": True},
            }
        return {key: strip_inline_images(item) for key, item in value.items()}
    if isinstance(value, str) and value.startswith("data:image/"):
        return "[Inline image omitted from transcript]"
    return value


def to_numpy_tree(value: Any) -> Any:
    """Recursively convert tensors and nested values into pickle-safe data."""
    if hasattr(value, "detach") and hasattr(value, "cpu") and hasattr(value, "numpy"):
        return value.detach().cpu().numpy()
    if dataclasses.is_dataclass(value):
        return to_numpy_tree(dataclasses.asdict(value))
    if isinstance(value, dict):
        return {key: to_numpy_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [to_numpy_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(to_numpy_tree(item) for item in value)
    return value
