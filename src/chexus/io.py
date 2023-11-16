# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
import hashlib
import os


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
    info = "\n"
    info = f"File: {path}\n"
    from datetime import datetime

    info += f"Created: {datetime.fromtimestamp(os.path.getctime(path))}\n"
    info += f"Modified: {datetime.fromtimestamp(os.path.getmtime(path))}\n"
    size = os.path.getsize(path)
    units = ["byte", "kByte", "MByte", "GByte", "TByte"]
    for unit in units:
        if size < 1024:
            info += f"Size: {size:.2f} {unit}\n"
            break
        size /= 1024
    else:
        info += f"Size: {size:.2f} {units[-1]}\n"
    return info
