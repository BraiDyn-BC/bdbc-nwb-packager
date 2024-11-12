# MIT License
#
# Copyright (c) 2024 Keisuke Sehara and Ryo Aoki
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""the common components that may be frequently used when handling NWB-metadata."""

from typing import Any

import numpy as _np
import h5py as _h5

from .. import (
    types as _types,
)


PathLike = _types.PathLike
JSONLike = dict[str, Any]


class MetadataParseError(ValueError):
    def __init__(self, msg):
        super().__init__(msg)


def read_metadata_as_dict(h5file: PathLike) -> JSONLike:
    def pythonify_(entry: _h5.Dataset) -> Any:
        content = _np.array(entry).ravel()
        if _np.issubdtype(content.dtype, _np.integer):
            if content.size == 1:
                return int(content[0])
            else:
                return tuple(int(v) for v in content)
        elif _np.issubdtype(content.dtype, _np.floating):
            if content.size == 1:
                return float(content[0])
            else:
                return tuple(float(v) for v in content)
        else:
            # assumes byte string
            return content[0].decode('utf-8')

    def as_string_items_(entry: _h5.Dataset) -> tuple[str]:
        content = _np.array(entry).ravel()
        return tuple(item.decode('utf-8') for item in content)

    with _h5.File(h5file, 'r') as src:
        group = src['metadata']
        metadata = dict((key.lower(), pythonify_(group[key])) for key in group.keys())
        metadata['bhv_raw_labels'] = as_string_items_(src['behavior_raw/label'])
        metadata['bhv_ds_labels']  = as_string_items_(src['behavior_ds/label'])
    return metadata
