# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
from .tree import Dataset, Group
from .validate import Validator, Violation


class NX_class_attr_missing(Validator):
    def __init__(self) -> None:
        super().__init__("NX_class_attr_missing", "NX_class attribute is missing")

    def applies_to(self, node: Dataset | Group) -> bool:
        return isinstance(node, Group)

    def validate(
        self, tree: dict[str, Dataset | Group], node: Dataset | Group
    ) -> Violation | None:
        if "NX_class" not in node.attrs:
            return Violation(node.name)


class depends_on_target_missing(Validator):
    def __init__(self) -> None:
        super().__init__("depends_on_target_missing", "depends_on target is missing")

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


class legacy_nexus_class(Validator):
    def __init__(self) -> None:
        super().__init__("legacy_NX_class", "Check if NX_class is deprecated")

    def applies_to(self, node: Dataset | Group) -> bool:
        return isinstance(node, Group)

    def validate(
        self, tree: dict[str, Dataset | Group], node: Dataset | Group
    ) -> Violation | None:
        if (nx_class := node.attrs.get("NX_class")) is not None:
            if nx_class in ['NXgeometry', 'NXshape']:
                return Violation(node.name, f"NX_class {nx_class} is deprecated")


class group_has_units(Validator):
    def __init__(self) -> None:
        super().__init__("group_has_units", "Group should not have units attribute")

    def applies_to(self, node: Dataset | Group) -> bool:
        return isinstance(node, Group)

    def validate(
        self, tree: dict[str, Dataset | Group], node: Dataset | Group
    ) -> Violation | None:
        if 'units' in node.attrs:
            return Violation(node.name)


class invalid_units(Validator):
    def __init__(self) -> None:
        super().__init__("invalid_units", "Invalid units")

    def applies_to(self, node: Dataset | Group) -> bool:
        return isinstance(node, Dataset) and 'units' in node.attrs

    def validate(
        self, tree: dict[str, Dataset | Group], node: Dataset | Group
    ) -> Violation | None:
        units = node.attrs['units']
        if units.startswith('NX_'):
            return Violation(node.name, f"Invalid units {units}")
        invalid = ['hz']
        if units in invalid:
            return Violation(node.name, f"Invalid units {units}")


class index_has_units(Validator):
    def __init__(self) -> None:
        super().__init__("index_has_units", "Index should not have units attribute")

    def applies_to(self, node: Dataset | Group) -> bool:
        names = [
            'detector_number',
            'detector_id',
            'detector_index',
            'event_id',
            'event_index',
            'winding_order',
            'faces',
            'detector_faces',
            'cylinders',
            'cue_index',
        ]
        name = node.name.split('/')[-1]
        return isinstance(node, Dataset) and name in names

    def validate(
        self, tree: dict[str, Dataset | Group], node: Dataset | Group
    ) -> Violation | None:
        if 'units' in node.attrs:
            return Violation(node.name)


class float_dataset_has_no_units(Validator):
    def __init__(self) -> None:
        super().__init__(
            "float_dataset_has_no_units", "Float dataset should have units attribute"
        )

    def applies_to(self, node: Dataset | Group) -> bool:
        return isinstance(node, Dataset)

    def validate(
        self, tree: dict[str, Dataset | Group], node: Dataset | Group
    ) -> Violation | None:
        if node.dtype in ['float32', 'float64'] and 'units' not in node.attrs:
            return Violation(node.name)


def is_transformation(node: Dataset | Group) -> bool:
    return 'transformation_type' in node.attrs


class offset_units_missing(Validator):
    def __init__(self) -> None:
        super().__init__(
            "offset_units_missing",
            "Transformation with offset attr should also have offset_units attr.",
        )

    def applies_to(self, node: Dataset | Group) -> bool:
        return isinstance(node, Dataset) and is_transformation(node)

    def validate(
        self, tree: dict[str, Dataset | Group], node: Dataset | Group
    ) -> Violation | None:
        if 'offset' in node.attrs:
            if 'offset_units' not in node.attrs:
                return Violation(node.name)


class transformation_depends_on_missing(Validator):
    def __init__(self) -> None:
        super().__init__(
            "transformation_missing_depends_on",
            "Transformation should have depends_on attribute",
        )

    def applies_to(self, node: Dataset | Group) -> bool:
        return isinstance(node, Dataset) and is_transformation(node)

    def validate(
        self, tree: dict[str, Dataset | Group], node: Dataset | Group
    ) -> Violation | None:
        if 'depends_on' not in node.attrs:
            return Violation(node.name)


def base_validators():
    return [
        NX_class_attr_missing(),
        depends_on_target_missing(),
        legacy_nexus_class(),
        group_has_units(),
        invalid_units(),
        index_has_units(),
        float_dataset_has_no_units(),
        offset_units_missing(),
        transformation_depends_on_missing(),
    ]
