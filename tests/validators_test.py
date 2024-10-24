# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
import numpy as np
import pytest

import chexus


def test_depends_on_missing():
    # class with, e.g., attr NX_class=NXdetector should have child depends_on.
    good = chexus.Group(name="x", attrs={"NX_class": "NXdetector"})
    depends_on = chexus.Dataset(
        name="depends_on", value=".", shape=None, dtype=str, parent=good
    )
    good.children["depends_on"] = depends_on
    assert chexus.validators.depends_on_missing().applies_to(good)
    assert chexus.validators.depends_on_missing().validate(good) is None
    bad = chexus.Group(name="x", attrs={"NX_class": "NXdetector"}, children={})
    assert chexus.validators.depends_on_missing().applies_to(bad)
    result = chexus.validators.depends_on_missing().validate(bad)
    assert isinstance(result, chexus.Violation)
    assert result.name == "x"


def test_depends_on_target_missing():
    group = chexus.Group(name="x", attrs={"NX_class": "NXdetector"})
    depends_on = chexus.Dataset(
        name="x/depends_on", value=".", shape=None, dtype=str, parent=group
    )
    group.children["depends_on"] = depends_on
    # '.' means no transform, this should pass
    assert chexus.validators.depends_on_target_missing().applies_to(depends_on)
    assert chexus.validators.depends_on_target_missing().validate(depends_on) is None
    depends_on.value = "transform"
    result = chexus.validators.depends_on_target_missing().validate(depends_on)
    assert isinstance(result, chexus.Violation)
    assert result.name == "x/depends_on"
    transform = chexus.Dataset(
        name="x/transform",
        value=None,
        shape=None,
        dtype=float,
        parent=group,
        attrs={"depends_on": "missing"},
    )
    group.children["transform"] = transform
    result = chexus.validators.depends_on_target_missing().validate(depends_on)
    # Still fails because 'transform' is not a transform
    assert isinstance(result, chexus.Violation)
    assert result.name == "x/depends_on"
    transform.attrs["transformation_type"] = "translation"
    transform.attrs["vector"] = [1.0, 0.0, 0.0]
    assert chexus.validators.depends_on_target_missing().validate(depends_on) is None
    assert chexus.validators.depends_on_target_missing().applies_to(transform)
    result = chexus.validators.depends_on_target_missing().validate(transform)
    assert isinstance(result, chexus.Violation)
    assert result.name == "x/transform"


def test_float_dataset_units_missing():
    good = chexus.Dataset(
        name="x",
        value=1.0,
        shape=None,
        dtype=float,
        parent=None,
        attrs={"units": "m"},
    )
    assert chexus.validators.float_dataset_units_missing().validate(good) is None
    bad = chexus.Dataset(
        name="x",
        value=1.0,
        shape=None,
        dtype=float,
        parent=None,
        attrs={},
    )
    result = chexus.validators.float_dataset_units_missing().validate(bad)
    assert isinstance(result, chexus.Violation)
    assert result.name == "x"


def test_group_has_units():
    good = chexus.Group(name="x", attrs={})
    assert chexus.validators.group_has_units().applies_to(good)
    assert chexus.validators.group_has_units().validate(good) is None
    bad = chexus.Group(name="x", attrs={"units": "m"})
    assert chexus.validators.group_has_units().applies_to(bad)
    result = chexus.validators.group_has_units().validate(bad)
    assert isinstance(result, chexus.Violation)
    assert result.name == "x"


@pytest.mark.parametrize("name", ["detector_number", "event_index", "event_id"])
def test_index_has_units(name: str):
    good = chexus.Dataset(
        name=name,
        value=1,
        shape=None,
        dtype=int,
        parent=None,
        attrs={},
    )
    assert chexus.validators.index_has_units().applies_to(good)
    assert chexus.validators.index_has_units().validate(good) is None
    bad = chexus.Dataset(
        name=name,
        value=1,
        shape=None,
        dtype=int,
        parent=None,
        attrs={"units": ""},
    )
    assert chexus.validators.index_has_units().applies_to(bad)
    result = chexus.validators.index_has_units().validate(bad)
    assert isinstance(result, chexus.Violation)
    assert result.name == name


@pytest.mark.parametrize("name", ["pixel_mask", "pixel_mask_0", "pixel_mask_1"])
def test_mask_has_units(name: str):
    good = chexus.Dataset(
        name=name,
        value=1,
        shape=None,
        dtype=int,
        parent=None,
        attrs={},
    )
    assert chexus.validators.mask_has_units().applies_to(good)
    assert chexus.validators.mask_has_units().validate(good) is None
    bad = chexus.Dataset(
        name=name,
        value=1,
        shape=None,
        dtype=int,
        parent=None,
        attrs={"units": ""},
    )
    assert chexus.validators.mask_has_units().applies_to(bad)
    result = chexus.validators.mask_has_units().validate(bad)
    assert isinstance(result, chexus.Violation)
    assert result.name == name


@pytest.mark.parametrize("dtype", [str, bool, np.bool_])
def test_non_numeric_dataset_has_units(dtype):
    good = chexus.Dataset(
        name="x",
        value=1,
        shape=None,
        dtype=dtype,
        parent=None,
        attrs={},
    )
    assert chexus.validators.non_numeric_dataset_has_units().applies_to(good)
    assert chexus.validators.non_numeric_dataset_has_units().validate(good) is None
    bad = chexus.Dataset(
        name="x",
        value=1,
        shape=None,
        dtype=dtype,
        parent=None,
        attrs={"units": ""},
    )
    assert chexus.validators.non_numeric_dataset_has_units().applies_to(bad)
    result = chexus.validators.non_numeric_dataset_has_units().validate(bad)
    assert isinstance(result, chexus.Violation)
    assert result.name == "x"


def test_NX_class_attr_missing():
    good = chexus.Group(name="x", attrs={"NX_class": "NXtransformations"})
    assert chexus.validators.NX_class_attr_missing().validate(good) is None
    bad = chexus.Group(name="x", attrs={})
    result = chexus.validators.NX_class_attr_missing().validate(bad)
    assert isinstance(result, chexus.Violation)
    assert result.name == "x"


def test_NX_class_is_legacy():
    good = chexus.Group(name="x", attrs={"NX_class": "NXtransformations"})
    assert chexus.validators.NX_class_is_legacy().validate(good) is None
    bad = chexus.Group(name="x", attrs={"NX_class": "NXgeometry"})
    result = chexus.validators.NX_class_is_legacy().validate(bad)
    assert isinstance(result, chexus.Violation)
    assert result.name == "x"


def test_transformation_depends_on_missing():
    good = chexus.Dataset(
        name="x",
        value=1,
        shape=None,
        dtype=float,
        parent=None,
        attrs={
            "transformation_type": "translation",
            "vector": [1.0, 0.0, 0.0],
            "depends_on": ".",
        },
    )
    assert chexus.validators.transformation_depends_on_missing().applies_to(good)
    assert chexus.validators.transformation_depends_on_missing().validate(good) is None
    bad = chexus.Dataset(
        name="x",
        value=1,
        shape=None,
        dtype=float,
        parent=None,
        attrs={
            "transformation_type": "translation",
            "vector": [1.0, 0.0, 0.0],
        },
    )
    assert chexus.validators.transformation_depends_on_missing().applies_to(bad)
    result = chexus.validators.transformation_depends_on_missing().validate(bad)
    assert isinstance(result, chexus.Violation)
    assert result.name == "x"


def test_transformation_offset_units_missing():
    good = chexus.Dataset(
        name="x",
        value=1,
        shape=None,
        dtype=float,
        parent=None,
        attrs={
            "transformation_type": "translation",
            "vector": [1.0, 0.0, 0.0],
            "offset": 1.0,
            "offset_units": "m",
        },
    )
    assert chexus.validators.transformation_offset_units_missing().applies_to(good)
    assert (
        chexus.validators.transformation_offset_units_missing().validate(good) is None
    )
    bad = chexus.Dataset(
        name="x",
        value=1,
        shape=None,
        dtype=float,
        parent=None,
        attrs={
            "transformation_type": "translation",
            "vector": [1.0, 0.0, 0.0],
            "offset": 1.0,
        },
    )
    assert chexus.validators.transformation_offset_units_missing().applies_to(bad)
    result = chexus.validators.transformation_offset_units_missing().validate(bad)
    assert isinstance(result, chexus.Violation)
    assert result.name == "x"


@pytest.mark.parametrize("units", ["NX_LENGTH", "NX_DIMENSIONLESS", "hz", ["m"]])
def test_units_invalid(units: str):
    good = chexus.Dataset(
        name="x", shape=None, dtype=float, parent=None, attrs={"units": ""}
    )
    assert chexus.validators.units_invalid().applies_to(good)
    assert chexus.validators.units_invalid().validate(good) is None
    bad = chexus.Dataset(
        name="x", shape=None, dtype=float, parent=None, attrs={"units": units}
    )
    assert chexus.validators.units_invalid().applies_to(bad)
    result = chexus.validators.units_invalid().validate(bad)
    assert isinstance(result, chexus.Violation)
    assert result.name == "x"


@pytest.mark.parametrize(
    "units",
    ["m", "", "1/year", "Hz/ms"],
)
def test_dataset_units_check_valid(units: str):
    dataset = chexus.Dataset(
        name="x",
        value=1.0,
        shape=None,
        dtype=np.float64,
        parent=None,
        attrs={"units": units},
    )
    assert chexus.validators.dataset_units_check().applies_to(dataset)
    result = chexus.validators.dataset_units_check().validate(dataset)
    assert result is None


@pytest.mark.parametrize(
    "units",
    ["test", "seco"],
)
def test_dataset_units_check_not_valid(units: str):
    dataset = chexus.Dataset(
        name="x",
        value=1.0,
        shape=None,
        dtype=np.float64,
        parent=None,
        attrs={"units": units},
    )
    assert chexus.validators.dataset_units_check().applies_to(dataset)
    result = chexus.validators.dataset_units_check().validate(dataset)
    assert isinstance(result, chexus.Violation)
    assert result.name == "x"


@pytest.mark.parametrize(
    ("units", "good"),
    [
        ("hz", False),
        ("", False),
        ("m/s", False),
        ("1/year", True),
        ("Hz", True),
        ("MHz", True),
        ("1/ms", True),
    ],
)
def test_NXdisk_chopper_units(units: str, good: bool):
    group = chexus.Group(
        name="x",
        attrs={"NX_class": "NXdisk_chopper"},
    )
    group.children["rotation_speed"] = chexus.Dataset(
        name="x/rotation_speed",
        value=1.0,
        shape=None,
        dtype=float,
        parent=group,
        attrs={"units": units},
    )
    assert chexus.validators.chopper_frequency_units_invalid().applies_to(group)
    result = chexus.validators.chopper_frequency_units_invalid().validate(group)
    if good:
        assert result is None
    else:
        assert isinstance(result, chexus.Violation)
        assert result.name == "x"


@pytest.mark.parametrize(
    ("units", "good"),
    [
        ("hz", False),
        ("", False),
        ("m/s", False),
        ("1/year", True),
        ("Hz", True),
        ("MHz", True),
        ("1/ms", True),
    ],
)
def test_NXdisk_chopper_units_log(units: str, good: bool):
    group = chexus.Group(
        name="x",
        attrs={"NX_class": "NXdisk_chopper"},
    )
    group.children["rotation_speed"] = chexus.Group(
        name="x/rotation_speed",
        attrs={"NX_class": "NXlog"},
    )
    group.children["rotation_speed"].children["value"] = chexus.Dataset(
        name="x/rotation_speed/value",
        value=1.0,
        shape=None,
        dtype=float,
        parent=group.children["rotation_speed"],
        attrs={"units": units},
    )
    assert chexus.validators.chopper_frequency_units_invalid().applies_to(group)
    result = chexus.validators.chopper_frequency_units_invalid().validate(group)
    if good:
        assert result is None
    else:
        assert isinstance(result, chexus.Violation)
        assert result.name == "x"


def test_NXlog_has_value():
    good = chexus.Group(
        name="x",
        parent=None,
        attrs={"NX_class": "NXlog"},
        children={},
    )
    good.children["time"] = chexus.Dataset(
        name="x/time", value=3.0, shape=None, dtype=int, parent=good
    )
    good.children["value"] = chexus.Dataset(
        name="x/value", value=1.0, shape=None, dtype=float, parent=good
    )
    assert chexus.validators.NXlog_has_value().validate(good) is None

    bad = chexus.Group(
        name="x",
        parent=None,
        attrs={"NX_class": "NXlog"},
        children={},
    )
    bad.children["time"] = chexus.Dataset(
        name="x/time", value=3.0, shape=None, dtype=int, parent=good
    )
    result = chexus.validators.float_dataset_units_missing().validate(bad)
    assert isinstance(result, chexus.Violation)
    assert result.name == "x"


def test_NXlog_top_dead_center_has_no_value():
    good = chexus.Group(
        name="top_dead_center",
        parent=None,
        attrs={"NX_class": "NXlog"},
        children={},
    )
    good.children["time"] = chexus.Dataset(
        name="top_dead_center/time", value=3.0, shape=None, dtype=int, parent=good
    )
    assert chexus.validators.NXlog_has_value().validate(good) is None

    bad = chexus.Group(
        name="top_dead_center",
        parent=None,
        attrs={"NX_class": "NXlog"},
        children={},
    )
    bad.children["time"] = chexus.Dataset(
        name="top_dead_center/time", value=3.0, shape=None, dtype=int, parent=good
    )
    bad.children["value"] = chexus.Dataset(
        name="top_dead_center/value", value=1.0, shape=None, dtype=float, parent=bad
    )
    result = chexus.validators.float_dataset_units_missing().validate(bad)
    assert isinstance(result, chexus.Violation)
    assert result.name == "top_dead_center"


def test_NXlog_nested_top_dead_center_has_no_value():
    good_parent = chexus.Group(
        name="chopper",
        parent=None,
        attrs={"NX_class": "NXdisk_chopper"},
        children={},
    )
    good = chexus.Group(
        name="chopper/top_dead_center",
        parent=good_parent,
        attrs={"NX_class": "NXlog"},
        children={},
    )
    good_parent.children["top_dead_center"] = good
    good.children["time"] = chexus.Dataset(
        name="top_dead_center/time", value=3.0, shape=None, dtype=int, parent=good
    )
    assert chexus.validators.NXlog_has_value().validate(good) is None

    bad_parent = chexus.Group(
        name="chopper",
        parent=None,
        attrs={"NX_class": "NXdisk_chopper"},
        children={},
    )
    bad = chexus.Group(
        name="chopper/top_dead_center",
        parent=bad_parent,
        attrs={"NX_class": "NXlog"},
        children={},
    )
    bad_parent.children["top_dead_center"] = bad
    bad.children["time"] = chexus.Dataset(
        name="top_dead_center/time", value=3.0, shape=None, dtype=int, parent=good
    )
    bad.children["value"] = chexus.Dataset(
        name="top_dead_center/value", value=1.0, shape=None, dtype=float, parent=bad
    )
    result = chexus.validators.float_dataset_units_missing().validate(bad)
    assert isinstance(result, chexus.Violation)
    assert result.name == "chopper/top_dead_center"


@pytest.mark.parametrize('value', [(1, 2, 3), np.array([1, 2, 3]), [[0, 1], [2, 3]]])
def test_duplicate_detector_number(value):
    det = chexus.Group(name="detector1", attrs={"NX_class": "NXdetector"})
    det.children = {
        'detector_number': chexus.Dataset(
            name="detector_number",
            value=value,
            shape=np.array(value).shape,
            dtype=int,
            parent=det,
        )
    }
    det2 = chexus.Group(name="detector2", attrs={"NX_class": "NXdetector"})
    det2.children = {
        'detector_number': chexus.Dataset(
            name="detector_number", value=[4, 5, 6], shape=(3,), dtype=int, parent=det2
        )
    }
    assert chexus.validators.detector_numbers_unique_in_all_detectors().applies_to(det)
    validator = chexus.validators.detector_numbers_unique_in_all_detectors()
    assert validator.validate(det) is None
    assert validator.validate(det2) is None
    # Second time the same detector numbers are seen, we expect a violation
    assert validator.validate(det) is not None


def test_event_id_not_in_detector_number():
    det = chexus.Group(name="detector1", attrs={"NX_class": "NXdetector"})
    det.children = {
        'data_good': chexus.Group(
            name="data_good", attrs={"NX_class": "NXevent_data"}, parent=det
        ),
        'data_bad': chexus.Group(
            name="data_bad", attrs={"NX_class": "NXevent_data"}, parent=det
        ),
        'detector_number': chexus.Dataset(
            name="detector_number", value=[1, 2, 3], shape=(3,), dtype=int, parent=det
        ),
    }
    det.children['data_good'].children = {
        'event_id': chexus.Dataset(
            name='event_id',
            value=[1, 2, 3],
            shape=(3,),
            dtype=int,
            parent=det.children['data_good'],
        )
    }
    det.children['data_bad'].children = {
        'event_id': chexus.Dataset(
            name='event_id',
            value=[1, 4],
            shape=(2,),
            dtype=int,
            parent=det.children['data_bad'],
        )
    }
    assert chexus.validators.event_id_subset_of_detector_number().applies_to(
        det.children['data_good']
    )
    assert chexus.validators.event_id_subset_of_detector_number().applies_to(
        det.children['data_bad']
    )
    assert (
        chexus.validators.event_id_subset_of_detector_number().validate(
            det.children['data_good']
        )
        is None
    )
    assert (
        chexus.validators.event_id_subset_of_detector_number().validate(
            det.children['data_bad']
        )
        is not None
    )
