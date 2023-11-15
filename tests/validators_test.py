# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
import numpy as np
import pytest

import notevennexus as nen


def test_depends_on_missing():
    # class with, e.g., attr NX_class=NXdetector should have child depends_on.
    good = nen.Group(name='x', attrs={'NX_class': 'NXdetector'})
    depends_on = nen.Dataset(
        name='depends_on', value='.', shape=None, dtype=str, parent=good
    )
    good.children['depends_on'] = depends_on
    assert nen.validators.depends_on_missing().applies_to(good)
    assert nen.validators.depends_on_missing().validate(good) is None
    bad = nen.Group(name='x', attrs={'NX_class': 'NXdetector'}, children={})
    assert nen.validators.depends_on_missing().applies_to(bad)
    result = nen.validators.depends_on_missing().validate(bad)
    assert isinstance(result, nen.Violation)
    assert result.name == 'x'


def test_depends_on_target_missing():
    group = nen.Group(name='x', attrs={'NX_class': 'NXdetector'})
    depends_on = nen.Dataset(
        name='x/depends_on', value='.', shape=None, dtype=str, parent=group
    )
    group.children['depends_on'] = depends_on
    # '.' means no transform, this should pass
    assert nen.validators.depends_on_target_missing().applies_to(depends_on)
    assert nen.validators.depends_on_target_missing().validate(depends_on) is None
    depends_on.value = 'transform'
    result = nen.validators.depends_on_target_missing().validate(depends_on)
    assert isinstance(result, nen.Violation)
    assert result.name == 'x/depends_on'
    transform = nen.Dataset(
        name='x/transform',
        value=None,
        shape=None,
        dtype=float,
        parent=group,
        attrs={'depends_on': 'missing'},
    )
    group.children['transform'] = transform
    result = nen.validators.depends_on_target_missing().validate(depends_on)
    # Still fails because 'transform' is not a transform
    assert isinstance(result, nen.Violation)
    assert result.name == 'x/depends_on'
    transform.attrs['transformation_type'] = 'translation'
    transform.attrs['vector'] = [1.0, 0.0, 0.0]
    assert nen.validators.depends_on_target_missing().validate(depends_on) is None
    assert nen.validators.depends_on_target_missing().applies_to(transform)
    result = nen.validators.depends_on_target_missing().validate(transform)
    assert isinstance(result, nen.Violation)
    assert result.name == 'x/transform'


def test_float_dataset_units_missing():
    good = nen.Dataset(
        name='x',
        value=1.0,
        shape=None,
        dtype=float,
        parent=None,
        attrs={'units': 'm'},
    )
    assert nen.validators.float_dataset_units_missing().validate(good) is None
    bad = nen.Dataset(
        name='x',
        value=1.0,
        shape=None,
        dtype=float,
        parent=None,
        attrs={},
    )
    result = nen.validators.float_dataset_units_missing().validate(bad)
    assert isinstance(result, nen.Violation)
    assert result.name == 'x'


def test_group_has_units():
    good = nen.Group(name='x', attrs={})
    assert nen.validators.group_has_units().applies_to(good)
    assert nen.validators.group_has_units().validate(good) is None
    bad = nen.Group(name='x', attrs={'units': 'm'})
    assert nen.validators.group_has_units().applies_to(bad)
    result = nen.validators.group_has_units().validate(bad)
    assert isinstance(result, nen.Violation)
    assert result.name == 'x'


@pytest.mark.parametrize('name', ['detector_number', 'event_index', 'event_id'])
def test_index_has_units(name: str):
    good = nen.Dataset(
        name=name,
        value=1,
        shape=None,
        dtype=int,
        parent=None,
        attrs={},
    )
    assert nen.validators.index_has_units().applies_to(good)
    assert nen.validators.index_has_units().validate(good) is None
    bad = nen.Dataset(
        name=name,
        value=1,
        shape=None,
        dtype=int,
        parent=None,
        attrs={'units': ''},
    )
    assert nen.validators.index_has_units().applies_to(bad)
    result = nen.validators.index_has_units().validate(bad)
    assert isinstance(result, nen.Violation)
    assert result.name == name


@pytest.mark.parametrize('name', ['pixel_mask', 'pixel_mask_0', 'pixel_mask_1'])
def test_mask_has_units(name: str):
    good = nen.Dataset(
        name=name,
        value=1,
        shape=None,
        dtype=int,
        parent=None,
        attrs={},
    )
    assert nen.validators.mask_has_units().applies_to(good)
    assert nen.validators.mask_has_units().validate(good) is None
    bad = nen.Dataset(
        name=name,
        value=1,
        shape=None,
        dtype=int,
        parent=None,
        attrs={'units': ''},
    )
    assert nen.validators.mask_has_units().applies_to(bad)
    result = nen.validators.mask_has_units().validate(bad)
    assert isinstance(result, nen.Violation)
    assert result.name == name


@pytest.mark.parametrize('dtype', [str, bool, np.bool_])
def test_non_numeric_dataset_has_units(dtype):
    good = nen.Dataset(
        name='x',
        value=1,
        shape=None,
        dtype=dtype,
        parent=None,
        attrs={},
    )
    assert nen.validators.non_numeric_dataset_has_units().applies_to(good)
    assert nen.validators.non_numeric_dataset_has_units().validate(good) is None
    bad = nen.Dataset(
        name='x',
        value=1,
        shape=None,
        dtype=dtype,
        parent=None,
        attrs={'units': ''},
    )
    assert nen.validators.non_numeric_dataset_has_units().applies_to(bad)
    result = nen.validators.non_numeric_dataset_has_units().validate(bad)
    assert isinstance(result, nen.Violation)
    assert result.name == 'x'


def test_NX_class_attr_missing():
    good = nen.Group(name='x', attrs={'NX_class': 'NXtransformations'})
    assert nen.validators.NX_class_attr_missing().validate(good) is None
    bad = nen.Group(name='x', attrs={})
    result = nen.validators.NX_class_attr_missing().validate(bad)
    assert isinstance(result, nen.Violation)
    assert result.name == 'x'


def test_NX_class_is_legacy():
    good = nen.Group(name='x', attrs={'NX_class': 'NXtransformations'})
    assert nen.validators.NX_class_is_legacy().validate(good) is None
    bad = nen.Group(name='x', attrs={'NX_class': 'NXgeometry'})
    result = nen.validators.NX_class_is_legacy().validate(bad)
    assert isinstance(result, nen.Violation)
    assert result.name == 'x'


def test_transformation_depends_on_missing():
    good = nen.Dataset(
        name='x',
        value=1,
        shape=None,
        dtype=float,
        parent=None,
        attrs={
            'transformation_type': 'translation',
            'vector': [1.0, 0.0, 0.0],
            'depends_on': '.',
        },
    )
    assert nen.validators.transformation_depends_on_missing().applies_to(good)
    assert nen.validators.transformation_depends_on_missing().validate(good) is None
    bad = nen.Dataset(
        name='x',
        value=1,
        shape=None,
        dtype=float,
        parent=None,
        attrs={
            'transformation_type': 'translation',
            'vector': [1.0, 0.0, 0.0],
        },
    )
    assert nen.validators.transformation_depends_on_missing().applies_to(bad)
    result = nen.validators.transformation_depends_on_missing().validate(bad)
    assert isinstance(result, nen.Violation)
    assert result.name == 'x'


def test_transformation_offset_units_missing():
    good = nen.Dataset(
        name='x',
        value=1,
        shape=None,
        dtype=float,
        parent=None,
        attrs={
            'transformation_type': 'translation',
            'vector': [1.0, 0.0, 0.0],
            'offset': 1.0,
            'offset_units': 'm',
        },
    )
    assert nen.validators.transformation_offset_units_missing().applies_to(good)
    assert nen.validators.transformation_offset_units_missing().validate(good) is None
    bad = nen.Dataset(
        name='x',
        value=1,
        shape=None,
        dtype=float,
        parent=None,
        attrs={
            'transformation_type': 'translation',
            'vector': [1.0, 0.0, 0.0],
            'offset': 1.0,
        },
    )
    assert nen.validators.transformation_offset_units_missing().applies_to(bad)
    result = nen.validators.transformation_offset_units_missing().validate(bad)
    assert isinstance(result, nen.Violation)
    assert result.name == 'x'


@pytest.mark.parametrize('units', ['NX_LENGTH', 'NX_DIMENSIONLESS', 'hz'])
def test_units_invalid(units: str):
    good = nen.Dataset(
        name='x', shape=None, dtype=float, parent=None, attrs={'units': ''}
    )
    assert nen.validators.units_invalid().applies_to(good)
    assert nen.validators.units_invalid().validate(good) is None
    bad = nen.Dataset(
        name='x', shape=None, dtype=float, parent=None, attrs={'units': units}
    )
    assert nen.validators.units_invalid().applies_to(bad)
    result = nen.validators.units_invalid().validate(bad)
    assert isinstance(result, nen.Violation)
    assert result.name == 'x'
