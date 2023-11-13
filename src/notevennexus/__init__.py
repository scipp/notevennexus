# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)

# flake8: noqa
import importlib.metadata

try:
    __version__ = importlib.metadata.version(__package__ or __name__)
except importlib.metadata.PackageNotFoundError:
    __version__ = "0.0.0"

del importlib

from . import validators
from .hdf5 import read_hdf5
from .io import make_fileinfo
from .json import read_json
from .tree import Dataset, Group, unroll_tree
from .validate import Validator, report, validate
