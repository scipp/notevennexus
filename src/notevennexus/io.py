# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
import hashlib


def compute_checksum(file_path) -> str:
    checksums = {
        'md5': hashlib.md5(usedforsecurity=False),
        'sha256': hashlib.sha256(usedforsecurity=False),
    }
    result = {}
    for hasher in checksums.values():
        with open(file_path, 'rb') as file:
            for chunk in iter(lambda: file.read(4096), b''):
                hasher.update(chunk)
        result[hasher.name] = hasher.hexdigest()
    info = f"md5: {result['md5']}\n"
    info += f"sha256: {result['sha256']}\n"
    return info


def make_fileinfo(path: str) -> str:
    """Get basic file info"""
    info = f"File: {path}\n"
    info += compute_checksum(path)
    return info
