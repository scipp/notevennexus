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
        return node.name.endswith("/depends_on") or "depends_on" in node.attrs

    def validate(self, node: Dataset | Group) -> Violation | None:
        if node.name.endswith("/depends_on"):
            target = node.value
        else:
            target = node.attrs["depends_on"]
        if target == ".":
            return None
        if not isinstance(target, str):
            return Violation(node.name, f"depends_on target {target} is not a string")
        path = target.split("/")
        if path[0] == "":
            start = self._find_root(node)
            path = path[1:]
        else:
            start = node.parent
        if path[0] == ".":
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
        if nx_class in ["NXgeometry", "NXorientation", "NXshape", "NXtranslation"]:
            return Violation(node.name, f"NX_class {nx_class} is deprecated")


class group_has_units(Validator):
    def __init__(self) -> None:
        super().__init__("group_has_units", "Group should not have units attribute")

    def applies_to(self, node: Dataset | Group) -> bool:
        return isinstance(node, Group)

    def validate(self, node: Dataset | Group) -> Violation | None:
        if "units" in node.attrs:
            return Violation(node.name)


class units_invalid(Validator):
    def __init__(self) -> None:
        super().__init__("units_invalid", "Invalid units attribute")

    def applies_to(self, node: Dataset | Group) -> bool:
        return isinstance(node, Dataset) and "units" in node.attrs

    def validate(self, node: Dataset | Group) -> Violation | None:
        units = node.attrs["units"]
        invalid = ["hz"]
        if not isinstance(units, str):
            return Violation(node.name, f"Invalid units type {type(units)}")
        # Units starting with NX_ are likely placeholders from the NeXus standard
        if units.startswith("NX_") or units in invalid:
            return Violation(node.name, f"Invalid units {units}")


class index_has_units(Validator):
    def __init__(self) -> None:
        super().__init__(
            "index_has_units", "Index or mask should not have units attribute"
        )

    def applies_to(self, node: Dataset | Group) -> bool:
        names = [
            "cue_index",
            "cylinders",
            "detector_faces",
            "detector_number",
            "event_id",
            "event_index",
            "faces",
            "image_key",
            "winding_order",
        ]
        name = node.name.split("/")[-1]
        return isinstance(node, Dataset) and name in names

    def validate(self, node: Dataset | Group) -> Violation | None:
        if "units" in node.attrs:
            return Violation(node.name)


class mask_has_units(Validator):
    def __init__(self) -> None:
        super().__init__("mask_has_units", "Mask should not have units attribute")

    def applies_to(self, node: Dataset | Group) -> bool:
        return isinstance(node, Dataset) and node.name.startswith("pixel_mask")

    def validate(self, node: Dataset | Group) -> Violation | None:
        if "units" in node.attrs:
            return Violation(node.name)


class float_dataset_units_missing(Validator):
    def __init__(self) -> None:
        super().__init__(
            "float_dataset_units_missing", "Float dataset should have units attribute"
        )

    def applies_to(self, node: Dataset | Group) -> bool:
        return isinstance(node, Dataset) and node.dtype in [np.float32, np.float64]

    def validate(self, node: Dataset | Group) -> Violation | None:
        if "units" not in node.attrs:
            return Violation(node.name)


class dataset_units_check(Validator):
    def __init__(self) -> None:
        super().__init__(
            "dataset_units_check",
            "Dataset should have units parasable by scipp",
        )

    def applies_to(self, node: Dataset | Group) -> bool:
        return isinstance(node, Dataset) and "units" in node.attrs

    def validate(self, node: Dataset | Group) -> Violation | None:
        import scipp as sc

        try:
            sc.Unit(node.attrs["units"])
        except sc.UnitError:
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
        if "units" in node.attrs:
            return Violation(node.name)


def is_transformation(node: Dataset | Group) -> bool:
    return "transformation_type" in node.attrs and "vector" in node.attrs


class transformation_offset_units_missing(Validator):
    def __init__(self) -> None:
        super().__init__(
            "transformation_offset_units_missing",
            "Transformation with offset attr should also have offset_units attr.",
        )

    def applies_to(self, node: Dataset | Group) -> bool:
        return is_transformation(node) and "offset" in node.attrs

    def validate(self, node: Dataset | Group) -> Violation | None:
        if "offset_units" not in node.attrs:
            return Violation(node.name)


class transformation_offset_units_invalid(Validator):
    def __init__(self) -> None:
        super().__init__(
            "transformation_offset_units_invalid",
            "Transformation offset_units attr. should be a length unit",
        )

    def applies_to(self, node: Dataset | Group) -> bool:
        return is_transformation(node) and "offset_units" in node.attrs

    def validate(self, node: Dataset | Group) -> Violation | None:
        import scipp as sc

        try:
            sc.scalar(1, unit=node.attrs["offset_units"]).to(unit="m")
        except sc.UnitError:
            return Violation(node.name)


class transformation_units_invalid(Validator):
    def __init__(self) -> None:
        super().__init__(
            "transformation_value_units_invalid",
            "Transformation value units should be a length unit "
            "if transformation type is translation and "
            "a rotation unit if transformation type is rotation",
        )

    def applies_to(self, node: Dataset | Group) -> bool:
        return is_transformation(node) and (
            isinstance(node, Dataset)
            or (isinstance(node, Group) and 'value' in node.children)
        )

    def validate(self, node: Dataset | Group) -> Violation | None:
        import scipp as sc

        unit = (node.children['value'] if isinstance(node, Group) else node).attrs.get(
            'units'
        )
        expected_unit = (
            "m" if node.attrs["transformation_type"] == "translation" else "rad"
        )
        try:
            sc.scalar(1, unit=unit).to(unit=expected_unit)
        except sc.UnitError:
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
        if "depends_on" not in node.attrs:
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
            and node.attrs.get("NX_class") == "NXdisk_chopper"
            and "rotation_speed" in node.children
        )

    def validate(self, node: Dataset | Group) -> Violation | None:
        import scipp as sc

        rotation_speed = node.children.get("rotation_speed")
        if (
            "NXlog" == rotation_speed.attrs.get("NX_class")
            and "value" in rotation_speed.children
        ):
            unit = rotation_speed.children["value"].attrs.get("units")
        elif "NXlog" != rotation_speed.attrs.get("NX_class"):
            unit = rotation_speed.attrs.get("units")
        else:
            return
        try:
            sc.scalar(1, unit=unit).to(unit="Hz")
        except sc.UnitError:
            pass
        else:
            return
        return Violation(node.name)


class detector_numbers_unique_in_detector(Validator):
    def __init__(self) -> None:
        super().__init__(
            "detector_numbers are not unique",
            "The values in all detector_numbers fields in all "
            "detectors should be unique.",
        )

    def applies_to(self, node: Dataset | Group) -> bool:
        return (
            isinstance(node, Group)
            and node.attrs.get('NX_class') == 'NXdetector'
            and 'detector_number' in node.children
        )

    def validate(self, node: Dataset | Group) -> Violation | None:
        detector_numbers = np.asarray(node.children['detector_number'].value)
        if not hasattr(detector_numbers, '__len__'):
            return
        if not len(detector_numbers.ravel()) == len(np.unique(detector_numbers)):
            return Violation(node.name)


class event_id_subset_of_detector_number(Validator):
    def __init__(self) -> None:
        super().__init__(
            "event_id is not subset of associated detector_numbers",
            "The values in the event_id field in NXevent_data should be "
            "a subset of the values in the detector_number dataset on the "
            "associated NXdetector.",
        )

    def applies_to(self, node: Dataset | Group) -> bool:
        return (
            isinstance(node, Group)
            and node.attrs.get('NX_class') == 'NXevent_data'
            and 'event_id' in node.children
            and 'detector_number' in node.parent.children
        )

    def validate(self, node: Dataset | Group) -> Violation | None:
        if not np.isin(
            node.children['event_id'].value,
            node.parent.children['detector_number'].value,
        ).all():
            return Violation(node.name)


class NXdetector_pixel_offsets_are_unambiguous(Validator):
    def __init__(self) -> None:
        super().__init__(
            "Shape of pixel offsets does not correspond to detector_number",
            "If detector_number is multi-dimensional, so should the pixel_offsets, "
            "else group attributes AXISNAME_indices (see NXdata) should indicate the "
            "correspondence between detector_number and pixel_offsets.",
        )

    def applies_to(self, node: Dataset | Group) -> bool:
        return (
            isinstance(node, Group)
            and node.attrs.get('NX_class') == 'NXdetector'
            and 'detector_number' in node.children
            and 'x_pixel_offset' in node.children
        )

    def validate(self, node: Dataset | Group) -> Violation | None:
        shape = node.children['detector_number'].shape
        for dim in 'xyz':
            key = f'{dim}_pixel_offset'
            if key not in node.children:
                continue
            pixel_shape = node.children[key].shape
            if shape != pixel_shape and f'{key}_indices' not in node.attrs:
                return Violation(
                    node.name,
                    f'{key} shape does not match detector_number and no {key}_indices '
                    'attribute found',
                )


physical_components = [
    "NXaperture",
    "NXattenuator",
    "NXbeam",
    "NXbeam_stop",
    "NXbending_magnet",
    "NXcapillary",
    "NXcollimator",
    "NXcrystal",
    "NXdetector",
    "NXdetector_module",
    "NXdisk_chopper",
    "NXfermi_chopper",
    "NXfilter",
    "NXflipper",
    "NXfresnel_zone_plate",
    "NXgrating",
    "NXguide",
    "NXinsertion_device",
    "NXmirror",
    "NXmoderator",
    "NXmonitor",
    "NXmonochromator",
    "NXpinhole",
    "NXpolarizer",
    "NXpositioner",
    "NXsample",
    "NXsensor",
    "NXslit",
    "NXsource",
    "NXvelocity_selector",
    "NXxraylens",
]


class depends_on_missing(Validator):
    def __init__(self) -> None:
        super().__init__(
            "depends_on_missing",
            "Group describes a physical component but has no depends_on",
        )

    def applies_to(self, node: Dataset | Group) -> bool:
        if isinstance(node, Group):
            if node.attrs.get("NX_class") in physical_components:
                return True
        return False

    def validate(self, node: Dataset | Group) -> Violation | None:
        if "depends_on" not in node.children:
            return Violation(node.name)


class NXlog_has_value(Validator):
    def __init__(self) -> None:
        super().__init__(
            "NXlog_has_value",
            "NXlogs must have a value field except for top_dead_center",
        )

    def applies_to(self, node: Dataset | Group) -> bool:
        return isinstance(node, Group) and node.attrs.get("NX_class") == "NXlog"

    def validate(self, node: Dataset | Group) -> Violation | None:
        if node.name.rsplit("/", 1)[-1] == "top_dead_center":
            if "value" in node.children:
                return Violation(
                    node.name, "top_dead_center logs must not have a value"
                )
        else:
            if "value" not in node.children:
                return Violation(node.name, "NXlog must have a value")


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
        NXlog_has_value(),
        detector_numbers_unique_in_detector(),
        event_id_subset_of_detector_number(),
        NXdetector_pixel_offsets_are_unambiguous(),
    ]
    if has_scipp:
        validators += [
            chopper_frequency_units_invalid(),
            dataset_units_check(),
            transformation_offset_units_invalid(),
            transformation_units_invalid(),
        ]
    return validators
