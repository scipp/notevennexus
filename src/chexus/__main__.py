# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
import argparse
import sys

import chexus


def _is_text_file(path: str) -> bool:
    """Check if file is text file"""
    try:
        with open(path, 'r') as f:
            f.readline()
        return True
    except UnicodeDecodeError:
        return False


def _path_to_keys(path: str) -> list[str]:
    """
    Convert path to list of keys.

    The path is a string with indices separated by slashes, e.g., "/entry1/group1".

    >>> _path_to_keys("/entry1/group1")
    ['entry1', 'group1']
    >>> _path_to_keys("/")
    []
    >>> _path_to_keys("")
    []
    >>> _path_to_keys("entry1")
    ['entry1']
    """
    if path in ['', '/']:
        return []

    return path.removeprefix('/').removesuffix('/').split('/')


def _retrieve_item(
    nested_obj: chexus.Group | chexus.Dataset, *keys: str, _visited: str = ''
) -> chexus.Group | chexus.Dataset:
    """Retrieve an item from a nested group using a list of keys."""
    if len(keys) == 0:
        return nested_obj

    cur_key, next_keys = keys[0], keys[1:]
    if not isinstance(nested_obj, chexus.Group) or not (cur_key in nested_obj.children):
        raise KeyError(f"{_visited}/{cur_key} not found.")

    return _retrieve_item(
        nested_obj.children[cur_key], *next_keys, _visited='/'.join((_visited, cur_key))
    )


def _find_root_group(group: chexus.Group, root_path: str) -> chexus.Group:
    """Find the root group of the file and returns its parent without other siblings.

    If the root group is the top-level group, it is returned as is.
    """
    keys = _path_to_keys(root_path)

    if not isinstance((candidate := _retrieve_item(group, *keys)), chexus.Group):
        raise ValueError(f"{root_path} is not a group.")

    return candidate


def _prune_siblings(root_child: chexus.Group) -> chexus.Group | None:
    """Return a new group with ``root_child`` as an only child."""

    if (parent := root_child.parent) is None:
        return

    return chexus.Group(
        name=parent.name,
        attrs=parent.attrs,
        children={
            name: child
            for name, child in parent.children.items()
            if child is root_child
        },
        parent=parent.parent,
    )


def main():
    parser = argparse.ArgumentParser(description='Validate NeXus files.')
    parser.add_argument(
        '--checksums', action='store_true', help='Compute and print checksums'
    )
    parser.add_argument(
        '--ignore-missing',
        action='store_true',
        help='Skip the validators that have missing dependecies',
    )
    parser.add_argument(
        '-r',
        '--root-groups',
        help='Path to the top-level group to validate',
        default='/',
    )
    parser.add_argument('path', help='Input file')

    args = parser.parse_args()
    path = args.path
    ignore_missing = args.ignore_missing

    has_scipp = False
    try:
        import scipp  # noqa: F401
    except ModuleNotFoundError:
        if not ignore_missing:
            print(
                'Error: Scipp was not found. The Nexus file validation was not run.\n'
                'To run the full test suite you need to install scipp'
                ' using `pip install scipp` or `conda install -c scipp scipp`.'
                ' This is recommended.\n'
                'To run only the tests that don\'t require scipp and ignore this'
                ' error add the flag `--ignore-missing`.'
            )
            sys.exit(1)
    else:
        has_scipp = True

    group = chexus.read_json(path) if _is_text_file(path) else chexus.read_hdf5(path)
    root_group = _find_root_group(group, args.root_groups)
    parent = _prune_siblings(root_group)
    target_group = root_group if parent is None else parent

    validators = chexus.validators.base_validators(has_scipp=has_scipp)
    results = chexus.validate(target_group, validators=validators)
    print(chexus.report(results=results))
    print(chexus.make_fileinfo(path))
    if args.checksums:
        print(chexus.compute_checksum(path))


if __name__ == '__main__':
    main()
