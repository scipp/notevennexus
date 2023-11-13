# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
from .tree import Dataset, Group
from .validate import Validator, Violation


class has_NX_class_attr(Validator):
    def __init__(self) -> None:
        super().__init__("has_NX_class_attr", "NX_class attribute is present")

    def applies_to(self, node: Dataset | Group) -> bool:
        return isinstance(node, Group)

    def validate(
        self, tree: dict[str, Dataset | Group], node: Dataset | Group
    ) -> Violation | None:
        if "NX_class" not in node.attrs:
            return Violation(node.name, "NX_class attribute is missing")


class depends_on_exists(Validator):
    def __init__(self) -> None:
        super().__init__("depends_on_exists", "depends_on target is present")

    def applies_to(self, node: Dataset | Group) -> bool:
        return node.name.endswith('/depends_on') or "depends_on" in node.attrs

    def validate(
        self, tree: dict[str, Dataset | Group], node: Dataset | Group
    ) -> Violation | None:
        if node.name.endswith('/depends_on'):
            target = node.value
        else:
            target = node.attrs["depends_on"]
        if target == '.':
            return None
        if target not in tree:
            return Violation(node.name, f"depends_on target {target} is missing")


def base_validators():
    return [has_NX_class_attr(), depends_on_exists()]
