# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
import json
from typing import Any

from .tree import Dataset, Group


def read_json(path: str) -> Group:
    """
    Read JSON file and return tree of datasets and groups.

    The JSON looks like this:

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
        return _read_group(json.load(f), path=[])


def _read_group(group: dict[str, Any], path: list[str]) -> Group:
    """Read JSON group"""
    path = path + [group.get("name", '')]
    name = '/'.join(path)
    grp = Group(name=name, attrs={}, children={}, parent=None)
    for child in group["children"]:
        if not isinstance(child, dict):
            continue
        module = child.get("module")
        if module is None:
            if child["type"] == "group":
                grp.children[child["name"]] = _read_group(child, path=path)
        elif module == "dataset":
            grp.children[child["config"]["name"]] = _read_dataset(child, path=path)
        elif module in ["f142", 'f144']:
            grp.children[child["config"]["source"]] = _read_source(child, path=path)
        elif module in ['tdct', 'ev44']:
            # No useful info in these?
            pass
        else:
            raise ValueError(f"Unsupported module: {module}")
    grp.attrs = _read_attrs(group)
    return grp


def _read_dataset(dataset: dict[str, Any], path: list[str]) -> Dataset:
    """Read JSON dataset"""
    path = path + [dataset['config']["name"]]
    name = '/'.join(path)
    return Dataset(
        name=name,
        shape=None,
        dtype=dataset["config"].get("type"),
        attrs=_read_attrs(dataset),
        parent=None,
    )


def _read_source(source: dict[str, Any], path: list[str]) -> Dataset:
    """Read JSON source"""
    path = path + [source['config']["source"]]
    name = '/'.join(path)
    ds = Dataset(
        name=name,
        shape=None,
        dtype=source["config"]["dtype"],
        attrs=_read_attrs(source),
        parent=None,
    )
    if (units := source["config"].get("value_units")) is not None:
        ds.attrs['units'] = units
    return ds


def _read_attrs(node: dict[str, Any]) -> dict[str, Any]:
    """Read JSON attributes"""
    attrs = {}
    for attr in node.get("attributes", {}):
        attrs[attr["name"]] = attr["values"]
    return attrs
