# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
import h5py

from .tree import Dataset, Group


def read_hdf5(path: str) -> Group:
    """Read HDF5 file and return tree of datasets and groups"""
    with h5py.File(path, "r") as f:
        return _read_group(f)


def _read_group(f: h5py.File) -> Group:
    """Read HDF5 group"""
    group = Group(name=f.name, attrs=dict(f.attrs), children={}, parent=None)
    for name, value in f.items():
        if isinstance(value, h5py.Dataset):
            group.children[name] = _read_dataset(value, group)
        elif isinstance(value, h5py.Group):
            group.children[name] = _read_group(value)
        else:
            raise ValueError(f"Unsupported type: {type(value)}")
    return group


def _read_dataset(f: h5py.Dataset, parent: Group) -> Dataset:
    """Read HDF5 dataset"""
    ds = Dataset(
        name=f.name,
        shape=f.shape,
        dtype=str(f.dtype),
        attrs=dict(f.attrs),
        parent=parent,
    )
    if ds.dtype == "object" and ds.shape == ():
        ds.value = f[()].decode(encoding='utf-8')
    return ds
