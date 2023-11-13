# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
import argparse

import notevennexus as nen


def main():
    parser = argparse.ArgumentParser(description='Validate NeXus files.')
    # parser.add_argument(
    #    '-i', '--input', help='Description for input argument', required=True
    # )
    # parser.add_argument(
    #    '-o', '--output', help='Description for output argument', required=True
    # )
    # Last unnamed argument is the input file
    parser.add_argument('path', help='Input file')
    args = parser.parse_args()
    path = args.path
    group = nen.read_hdf5(path)

    validators = nen.validators.base_validators()
    nen.validate(group, validators=validators)
    print(nen.report(validators=validators))
    print(nen.make_fileinfo(path))


if __name__ == '__main__':
    main()
