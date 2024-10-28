# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import h5py

_no_value_set = object()


@dataclass
class Dataset:
    """Info about an HDF5 dataset"""

    name: str
    shape: tuple[int, ...]
    dtype: str
    parent: Group
    attrs: dict[str, Any] = field(default_factory=dict)
    value: Any | None = None
    dataset: h5py.Dataset | None = None

    @property
    def value(self) -> Any | None:  # noqa: F811
        '''Returns the value of the dataset.
        If the value attribute has been set by the user
        that value is returned.
        Otherwise, if the dataset has access to a h5py dataset
        the value in that dataset will be returned.
        If no value was set and the object has no h5py dataset
        then None is returned.
        '''
        if self._value is not _no_value_set:
            return self._value
        if self.dataset is not None:
            # We don't want to cache the values
            # by saving them in the _value attribute.
            # The reason is that we don't want to
            # run out of memory if the file is large.
            try:
                return self.dataset.asstr()[()]
            except TypeError:
                return self.dataset[()]
        return None

    @value.setter
    def value(self, value: Any):
        # When the dataclass object is created
        # the setter will be called with
        # an instance of type property.
        # Set _value to uninitialized in that case.
        if isinstance(value, property):
            self._value = _no_value_set
        else:
            self._value = value


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
