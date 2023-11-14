# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
import pytest

import notevennexus as nen


@pytest.fixture
def tree_with_detector() -> nen.Group:
    detector1 = nen.Group(
        name='/entry/instrument/detector1', attrs={'NX_class': 'NXdetector'}
    )
    detector2 = nen.Group(
        name='/entry/instrument/detector2', attrs={'NX_class': 'NXdetector'}
    )
    instrument = nen.Group(
        name='/entry/instrument',
        attrs={'NX_class': 'NXinstrument'},
        children={'detector1': detector1, 'detector2': detector2},
    )
    entry = nen.Group(
        name='/entry',
        attrs={'NX_class': 'NXentry'},
        children={'instrument': instrument},
    )
    root = nen.Group(name='/', children={'entry': entry})
    detector1.parent = instrument
    detector2.parent = instrument
    instrument.parent = entry
    return root


def test_validate_detects_some_nested_problems(tree_with_detector: nen.Group):
    root = tree_with_detector
    detector1 = root.children['entry'].children['instrument'].children['detector1']
    detector1.children['depends_on'] = nen.Dataset(
        name='/entry/instrument/detector1/depends_on',
        shape=None,
        dtype=str,
        attrs={},
        parent=detector1,
        value='transform',
    )
    detector1.children['detector_number'] = nen.Dataset(
        name='/entry/instrument/detector1/detector_number',
        shape=(1,),
        dtype='int32',
        attrs={'units': ''},
        parent=detector1,
        value=0,
    )

    validators = nen.validators.base_validators()
    results = nen.validate(root, validators=validators)
    assert results[nen.validators.depends_on_missing].fails == 1
    assert results[nen.validators.depends_on_target_missing].fails == 1
    assert results[nen.validators.index_has_units].fails == 1
    assert results[nen.validators.float_dataset_units_missing].fails == 0
