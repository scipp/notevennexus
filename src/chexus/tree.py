# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Dataset:
    """Info about an HDF5 dataset"""

    name: str
    shape: tuple[int, ...]
    dtype: str
    parent: Group
    attrs: dict[str, Any] = field(default_factory=dict)
    value: Any | None = None


@dataclass
class Group:
    """Info about an HDF5 group"""

    name: str
    parent: Group | None = None
    attrs: dict[str, Any] = field(default_factory=dict)
    children: dict[str, Dataset | Group] = field(default_factory=dict)


def unroll_tree(tree: Group) -> dict[str, Dataset | Group]:
    """Unroll tree into a flat dictionary"""
    result = {}
    for child in tree.children.values():
        result[child.name] = child
        if isinstance(child, Group):
            result.update(unroll_tree(child))
    return result
