# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
from .tree import Dataset, Group
from .validate import Validator, Violation


class NX_class_attr_missing(Validator):
    def __init__(self) -> None:
        super().__init__("NX_class_attr_missing", "NX_class attribute is missing")

    def applies_to(self, node: Dataset | Group) -> bool:
        return isinstance(node, Group)

    def validate(self, node: Dataset | Group) -> Violation | None:
        if "NX_class" not in node.attrs:
            return Violation(node.name)


class depends_on_target_missing(Validator):
    def __init__(self) -> None:
        super().__init__("depends_on_target_missing", "depends_on target is missing")

    def applies_to(self, node: Dataset | Group) -> bool:
        return node.name.endswith('/depends_on') or "depends_on" in node.attrs

    def validate(self, node: Dataset | Group) -> Violation | None:
        if node.name.endswith('/depends_on'):
            target = node.value
        else:
            target = node.attrs["depends_on"]
        if target == '.':
            return None
        if not isinstance(target, str):
            return Violation(node.name, f"depends_on target {target} is not a string")
        path = target.split('/')
        if path[0] == '':
            start = self._find_root(node)
            path = path[1:]
        else:
            start = node.parent
        if path[0] == '.':
            path = path[1:]
        for name in path:
            if name not in start.children:
                return Violation(node.name, f"depends_on target {target} is missing")
            start = start.children[name]

    def _find_root(self, node: Dataset | Group) -> Dataset | Group:
        while node.parent is not None:
            node = node.parent
        return node


class legacy_nexus_class(Validator):
    def __init__(self) -> None:
        super().__init__("legacy_NX_class", "Check if NX_class is deprecated")

    def applies_to(self, node: Dataset | Group) -> bool:
        return isinstance(node, Group) and "NX_class" in node.attrs

    def validate(self, node: Dataset | Group) -> Violation | None:
        nx_class = node.attrs.get("NX_class")
        if nx_class in ['NXgeometry', 'NXorientation', 'NXshape', 'NXtranslation']:
            return Violation(node.name, f"NX_class {nx_class} is deprecated")


class group_has_units(Validator):
    def __init__(self) -> None:
        super().__init__("group_has_units", "Group should not have units attribute")

    def applies_to(self, node: Dataset | Group) -> bool:
        return isinstance(node, Group)

    def validate(self, node: Dataset | Group) -> Violation | None:
        if 'units' in node.attrs:
            return Violation(node.name)


class invalid_units(Validator):
    def __init__(self) -> None:
        super().__init__("invalid_units", "Invalid units")

    def applies_to(self, node: Dataset | Group) -> bool:
        return isinstance(node, Dataset) and 'units' in node.attrs

    def validate(self, node: Dataset | Group) -> Violation | None:
        units = node.attrs['units']
        if units.startswith('NX_'):  # Placeholder from NeXus standard
            return Violation(node.name, f"Invalid units {units}")
        invalid = ['hz']
        if units in invalid:
            return Violation(node.name, f"Invalid units {units}")


class index_has_units(Validator):
    def __init__(self) -> None:
        super().__init__(
            "index_has_units", "Index or mask should not have units attribute"
        )

    def applies_to(self, node: Dataset | Group) -> bool:
        names = [
            'cue_index',
            'cylinders',
            'detector_faces',
            'detector_number',
            'event_id',
            'event_index',
            'faces',
            'image_key',
            'winding_order',
        ]
        name = node.name.split('/')[-1]
        return (
            isinstance(node, Dataset) and name in names or name.startswith('pixel_mask')
        )

    def validate(self, node: Dataset | Group) -> Violation | None:
        if 'units' in node.attrs:
            return Violation(node.name)


class float_dataset_has_no_units(Validator):
    def __init__(self) -> None:
        super().__init__(
            "float_dataset_has_no_units", "Float dataset should have units attribute"
        )

    def applies_to(self, node: Dataset | Group) -> bool:
        return isinstance(node, Dataset) and node.dtype in ['float32', 'float64']

    def validate(self, node: Dataset | Group) -> Violation | None:
        if 'units' not in node.attrs:
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
        return (
            isinstance(node, Dataset)
            and is_transformation(node)
            and 'offset' in node.attrs
        )

    def validate(self, node: Dataset | Group) -> Violation | None:
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

    def validate(self, node: Dataset | Group) -> Violation | None:
        if 'depends_on' not in node.attrs:
            return Violation(node.name)


physical_components = [
    'NXaperture',
    'NXattenuator',
    'NXbeam',
    'NXbeam_stop',
    'NXbending_magnet',
    'NXcapillary',
    'NXcollimator',
    'NXcrystal',
    'NXdetector',
    'NXdetector_module',
    'NXdisk_chopper',
    'NXfermi_chopper',
    'NXfilter',
    'NXflipper',
    'NXfresnel_zone_plate',
    'NXgrating',
    'NXguide',
    'NXinsertion_device',
    'NXmirror',
    'NXmoderator',
    'NXmonitor',
    'NXmonochromator',
    'NXpinhole',
    'NXpolarizer',
    'NXpositioner',
    'NXsample',
    'NXsensor',
    'NXslit',
    'NXsource',
    'NXvelocity_selector',
    'NXxraylens',
]


class depends_on_missing(Validator):
    def __init__(self) -> None:
        super().__init__(
            'depends_on_missing',
            'Group describes a physical component but has no depends_on',
        )

    def applies_to(self, node: Dataset | Group) -> bool:
        if isinstance(node, Group):
            if node.attrs.get('NX_class') in physical_components:
                return True
        return False

    def validate(self, node: Dataset | Group) -> Violation | None:
        if 'depends_on' not in node.children:
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
        depends_on_missing(),
    ]
