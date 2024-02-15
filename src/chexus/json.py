# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
import json
from typing import Any

import numpy as np

from .tree import Dataset, Group


def read_json(path: str) -> Group:
    """
    Read JSON NeXus file and return tree of datasets and groups.

    The JSON looks something like this:

    {
        "name": "delay",
        "type": "group",
        "attributes": [
            {
                "name": "NX_class",
                "dtype": "string",
                "values": "NXlog"
            }
        ],
        "children": [
            {
                "module": "f142",
                "config": {
                    "source": "source",
                    "topic": "topic",
                    "dtype": "double",
                    "value_units": "ns"
                },
                "attributes": [
                    {
                        "name": "units",
                        "dtype": "string",
                        "values": "ns"
                    }
                ]
            },
            {
                "module": "dataset",
                "config": {
                    "name": "slits",
                    "values": 1,
                    "type": "int64"
                }
            },
        ]
    },
    """
    with open(path, "r") as f:
        return _read_group(json.load(f))


def _read_group(group: dict[str, Any], parent: Group | None = None) -> Group:
    """Read JSON group"""
    name = group.get("name", '')
    if parent is not None:
        name = parent.name + '/' + name
    grp = Group(name=name, attrs={}, children={}, parent=parent)
    for child in group["children"]:
        if not isinstance(child, dict):
            continue
        module = child.get("module")
        if module is None:
            if child["type"] == "group":
                grp.children[child["name"]] = _read_group(child, parent=grp)
        elif module == "dataset":
            grp.children[child["config"]["name"]] = _read_dataset(child, parent=grp)
        elif module in ["f142", 'f144']:
            grp.children[child["config"]["source"]] = _read_source(child, parent=grp)
        elif module in ['tdct', 'ev42', 'ev44']:
            # No useful info in these?
            pass
        else:
            raise ValueError(f"Unsupported module: {module}")
    grp.attrs = _read_attrs(group)
    return grp


def _read_dataset(dataset: dict[str, Any], parent: Group) -> Dataset:
    """Read JSON dataset"""
    name = parent.name + '/' + dataset['config']["name"]
    if (values := dataset["config"].get("values")) is not None:
        type_from_values = type(values)
    return Dataset(
        name=name,
        shape=None,
        dtype=_translate_dtype(dataset["config"].get("type", type_from_values)),
        attrs=_read_attrs(dataset),
        parent=parent,
    )


def _read_source(source: dict[str, Any], parent: Group) -> Dataset:
    """Read JSON source"""
    name = parent.name + '/' + source['config']["source"]
    ds = Dataset(
        name=name,
        shape=None,
        dtype=_translate_dtype(source["config"]["dtype"]),
        attrs=_read_attrs(source),
        parent=parent,
    )
    if (units := source["config"].get("value_units")) is not None:
        ds.attrs['units'] = units
    return ds


def _translate_dtype(dtype: str) -> str:
    """Translate dtype from JSON to Python/NumPy"""
    if dtype == "double":
        return np.float64
    if dtype == "float":
        return np.float32
    if dtype == "string":
        return np.dtype('U')
    return np.dtype(dtype)


def _read_attrs(node: dict[str, Any]) -> dict[str, Any]:
    """Read JSON attributes"""
    attrs = {}
    for attr in node.get("attributes", {}):
        attrs[attr["name"]] = attr["values"]
    return attrs
