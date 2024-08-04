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
from typing import Union
from pathlib import Path

import numpy as _np
import numpy.typing as _npt
import pandas as _pd
import h5py as _h5


def load_raw_daq(rawfile: Union[str, Path]) -> _pd.DataFrame:
    with _h5.File(rawfile, 'r') as src:
        labels = tuple(item.decode('utf-8').replace('.', '').replace(' ', '-').replace('-', '_') \
                       for item in _np.array(src['behavior_raw/label']).ravel())
        data = _np.array(src['behavior_raw/data'])
    return _pd.DataFrame(data=dict((labels[i], data[i]) for i in range(len(labels))))


def extract_blocks(flags: _npt.NDArray) -> _pd.DataFrame:
    """extracts ranges of flags appearing in blocks, in terms of sample indices.
    application of this method to floating-points number arrays is not recommended.
    """
    stepidxx = _np.where(_np.concatenate([(True,), (_np.diff(flags) != 0)]))[0]
    data = {
        'start': [],
        'stop': [],
        'value': [],
    }
    def _add(start, stop):
        vals = tuple(set(v for v in flags[start:stop]))
        assert len(vals) == 1
        data['start'].append(start)
        data['stop'].append(stop)
        data['value'].append(vals[0])

    for start, stop in zip(stepidxx[:-1], stepidxx[1:]):
        _add(int(start), int(stop))
    _add(int(stop), int(flags.size))
    return _pd.DataFrame(data=data)

