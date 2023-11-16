# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
import pytest

import chexus


@pytest.fixture
def tree_with_detector() -> chexus.Group:
    detector1 = chexus.Group(
        name='/entry/instrument/detector1', attrs={'NX_class': 'NXdetector'}
    )
    detector2 = chexus.Group(
        name='/entry/instrument/detector2', attrs={'NX_class': 'NXdetector'}
    )
    instrument = chexus.Group(
        name='/entry/instrument',
        attrs={'NX_class': 'NXinstrument'},
        children={'detector1': detector1, 'detector2': detector2},
    )
    entry = chexus.Group(
        name='/entry',
        attrs={'NX_class': 'NXentry'},
        children={'instrument': instrument},
    )
    root = chexus.Group(name='/', children={'entry': entry})
    detector1.parent = instrument
    detector2.parent = instrument
    instrument.parent = entry
    return root


def test_validate_detects_some_nested_problems(tree_with_detector: chexus.Group):
    root = tree_with_detector
    detector1 = root.children['entry'].children['instrument'].children['detector1']
    detector1.children['depends_on'] = chexus.Dataset(
        name='/entry/instrument/detector1/depends_on',
        shape=None,
        dtype=str,
        attrs={},
        parent=detector1,
        value='transform',
    )
    detector1.children['detector_number'] = chexus.Dataset(
        name='/entry/instrument/detector1/detector_number',
        shape=(1,),
        dtype='int32',
        attrs={'units': ''},
        parent=detector1,
        value=0,
    )

    validators = chexus.validators.base_validators()
    results = chexus.validate(root, validators=validators)
    assert results[chexus.validators.depends_on_missing].fails == 1
    assert results[chexus.validators.depends_on_target_missing].fails == 1
    assert results[chexus.validators.index_has_units].fails == 1
    assert results[chexus.validators.float_dataset_units_missing].fails == 0
