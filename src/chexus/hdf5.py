# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
from typing import Any

import h5py

from .tree import Dataset, Group


def read_hdf5(path: str) -> Group:
    """Read HDF5 file and return tree of datasets and groups"""
    with h5py.File(path, "r") as f:
        return _read_group(f)


def _read_attrs(node: h5py.Dataset | h5py.Group) -> dict[str, Any]:
    """Read HDF5 attributes"""
    attrs = dict(node.attrs)
    # Convert bytes to strings
    for key, value in attrs.items():
        if isinstance(value, bytes):
            attrs[key] = value.decode(encoding='utf-8')
    return attrs


def _read_group(group: h5py.File, parent: Group | None = None) -> Group:
    """Read HDF5 group"""
    grp = Group(name=group.name, attrs=_read_attrs(group), children={}, parent=parent)
    for name, value in group.items():
        if isinstance(value, h5py.Dataset):
            grp.children[name] = _read_dataset(value, parent=grp)
        elif isinstance(value, h5py.Group):
            grp.children[name] = _read_group(value, parent=grp)
        else:
            raise ValueError(f"Unsupported type: {type(value)}")
    return grp


def _read_dataset(dataset: h5py.Dataset, parent: Group) -> Dataset:
    """Read HDF5 dataset"""
    ds = Dataset(
        name=dataset.name,
        shape=dataset.shape,
        dtype=dataset.dtype,
        attrs=_read_attrs(dataset),
        parent=parent,
    )
    try:
        ds.value = dataset.asstr()[()]
    except TypeError:
        pass
    return ds
