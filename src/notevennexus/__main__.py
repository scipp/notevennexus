# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
import argparse

import notevennexus as nen


def _is_text_file(path: str) -> bool:
    """Check if file is text file"""
    try:
        with open(path, 'r') as f:
            f.read()
        return True
    except UnicodeDecodeError:
        return False


def main():
    parser = argparse.ArgumentParser(description='Validate NeXus files.')
    parser.add_argument('path', help='Input file')
    args = parser.parse_args()
    path = args.path
    group = nen.read_json(path) if _is_text_file(path) else nen.read_hdf5(path)

    validators = nen.validators.base_validators()
    nen.validate(group, validators=validators)
    print(nen.report(validators=validators))
    print(nen.make_fileinfo(path))


if __name__ == '__main__':
    main()
