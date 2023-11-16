# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
import argparse

import chexus


def _is_text_file(path: str) -> bool:
    """Check if file is text file"""
    try:
        with open(path, 'r') as f:
            f.readline()
        return True
    except UnicodeDecodeError:
        return False


def main():
    parser = argparse.ArgumentParser(description='Validate NeXus files.')
    parser.add_argument(
        '--checksums', action='store_true', help='Compute and print checksums'
    )
    parser.add_argument('path', help='Input file')
    args = parser.parse_args()
    path = args.path
    group = chexus.read_json(path) if _is_text_file(path) else chexus.read_hdf5(path)

    validators = chexus.validators.base_validators()
    results = chexus.validate(group, validators=validators)
    print(chexus.report(results=results))
    print(chexus.make_fileinfo(path))
    if args.checksums:
        print(chexus.compute_checksum(path))


if __name__ == '__main__':
    main()
