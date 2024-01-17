# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
import numpy as np

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
        super().__init__(
            "depends_on_target_missing",
            "depends_on target is missing or is not a transformation.",
        )

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
        if not is_transformation(start):
            return Violation(
                node.name, f"depends_on target {target} is not a transformation"
            )

    def _find_root(self, node: Dataset | Group) -> Dataset | Group:
        while node.parent is not None:
            node = node.parent
        return node


class NX_class_is_legacy(Validator):
    def __init__(self) -> None:
        super().__init__("NX_class_is_legacy", "Check if NX_class is deprecated")

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


class units_invalid(Validator):
    def __init__(self) -> None:
        super().__init__("units_invalid", "Invalid units attribute")

    def applies_to(self, node: Dataset | Group) -> bool:
        return isinstance(node, Dataset) and 'units' in node.attrs

    def validate(self, node: Dataset | Group) -> Violation | None:
        units = node.attrs['units']
        invalid = ['hz']
        if not isinstance(units, str):
            return Violation(node.name, f'Invalid units type {type(units)}')
        # Units starting with NX_ are likely placeholders from the NeXus standard
        if units.startswith('NX_') or units in invalid:
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
        return isinstance(node, Dataset) and name in names

    def validate(self, node: Dataset | Group) -> Violation | None:
        if 'units' in node.attrs:
            return Violation(node.name)


class mask_has_units(Validator):
    def __init__(self) -> None:
        super().__init__("mask_has_units", "Mask should not have units attribute")

    def applies_to(self, node: Dataset | Group) -> bool:
        return isinstance(node, Dataset) and node.name.startswith('pixel_mask')

    def validate(self, node: Dataset | Group) -> Violation | None:
        if 'units' in node.attrs:
            return Violation(node.name)


class float_dataset_units_missing(Validator):
    def __init__(self) -> None:
        super().__init__(
            "float_dataset_units_missing", "Float dataset should have units attribute"
        )

    def applies_to(self, node: Dataset | Group) -> bool:
        return isinstance(node, Dataset) and node.dtype in [np.float32, np.float64]

    def validate(self, node: Dataset | Group) -> Violation | None:
        if 'units' not in node.attrs:
            return Violation(node.name)


class non_numeric_dataset_has_units(Validator):
    def __init__(self) -> None:
        super().__init__(
            "non_numeric_dataset_has_units",
            "Non-numeric dataset should not have units attribute",
        )

    def applies_to(self, node: Dataset | Group) -> bool:
        return isinstance(node, Dataset) and not np.issubdtype(node.dtype, np.number)

    def validate(self, node: Dataset | Group) -> Violation | None:
        if 'units' in node.attrs:
            return Violation(node.name)


def is_transformation(node: Dataset | Group) -> bool:
    return 'transformation_type' in node.attrs and 'vector' in node.attrs


class transformation_offset_units_missing(Validator):
    def __init__(self) -> None:
        super().__init__(
            "transformation_offset_units_missing",
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
            "transformation_depends_on_missing",
            "Transformation should have depends_on attribute",
        )

    def applies_to(self, node: Dataset | Group) -> bool:
        return isinstance(node, Dataset) and is_transformation(node)

    def validate(self, node: Dataset | Group) -> Violation | None:
        if 'depends_on' not in node.attrs:
            return Violation(node.name)


class chopper_frequency_units_invalid(Validator):
    def __init__(self) -> None:
        super().__init__(
            "chopper_frequency_unit_invalid",
            "The unit of NXdisk_chopper.rotation_speed should have dimension 1/Time",
        )

    def applies_to(self, node: Dataset | Group) -> bool:
        return (
            isinstance(node, Group)
            and node.attrs.get('NX_class') == 'NXdisk_chopper'
            and 'rotation_speed' in node.children
        )

    def validate(self, node: Dataset | Group) -> Violation | None:
        import scipp as sc

        if 'units' in node.children.get('rotation_speed').attrs:
            unit = node.children.get('rotation_speed').attrs.get('units')
            try:
                sc.scalar(1, unit=unit).to(unit='Hz')
            except sc.UnitError:
                pass
            else:
                return
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


def base_validators(*, has_scipp=True):
    validators = [
        depends_on_missing(),
        depends_on_target_missing(),
        float_dataset_units_missing(),
        group_has_units(),
        index_has_units(),
        mask_has_units(),
        non_numeric_dataset_has_units(),
        NX_class_attr_missing(),
        NX_class_is_legacy(),
        transformation_depends_on_missing(),
        transformation_offset_units_missing(),
        units_invalid(),
    ]
    if has_scipp:
        validators += [
            chopper_frequency_units_invalid(),
        ]
    return validators
