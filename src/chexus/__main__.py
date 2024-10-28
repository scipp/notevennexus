# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
# ruff: noqa: T201
import argparse
import sys

import chexus


def _is_text_file(path: str) -> bool:
    """Check if file is text file"""
    try:
        with open(path) as f:
            f.readline()
        return True
    except UnicodeDecodeError:
        return False


def main():
    parser = argparse.ArgumentParser(description="Validate NeXus files.")
    parser.add_argument(
        "--checksums", action="store_true", help="Compute and print checksums"
    )
    parser.add_argument(
        "--ignore-missing",
        action="store_true",
        help="Skip the validators that have missing dependencies",
    )
    # Add argument to return bad exit code if validation fails
    parser.add_argument(
        "--exit-on-fail",
        action="store_true",
        help="Return a non-zero exit code if validation fails",
    )
    parser.add_argument(
        "-r",
        "--root-path",
        type=lambda s: s if s.startswith("/") else "/" + s,
        help="Path to the top-level group to validate",
        default="",
    )
    parser.add_argument("path", help="Input file")
    args = parser.parse_args()
    path = args.path
    ignore_missing = args.ignore_missing

    has_scipp = False
    try:
        import scipp  # noqa: F401
    except ModuleNotFoundError:
        if not ignore_missing:
            print(
                "Error: Scipp was not found. The Nexus file validation was not run.\n"
                "To run the full test suite you need to install scipp"
                " using `pip install scipp` or `conda install -c scipp scipp`."
                " This is recommended.\n"
                "To run only the tests that don't require scipp and ignore this"
                " error add the flag `--ignore-missing`."
            )
            sys.exit(1)
    else:
        has_scipp = True

    if _is_text_file(path):
        group = chexus.read_json(path)
    else:
        # File is closed when 'reader' goes out of scope.
        # We need to keep it open for lazily loading values.
        reader = chexus.read_hdf5(path)
        group = next(reader)

    validators = chexus.validators.base_validators(has_scipp=has_scipp)

    def skip_condition(node):
        return not node.name.startswith(args.root_path)

    results = chexus.validate(
        group, validators=validators, skip_condition=skip_condition
    )
    print(chexus.report(results=results))
    print(chexus.make_fileinfo(path))
    if args.checksums:
        print(chexus.compute_checksum(path))
    if args.exit_on_fail and chexus.has_violations(results):
        print("Validation has failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
