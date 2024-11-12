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

from typing import Iterator
from time import time as _now

import numpy as _np
import h5py as _h5
import pynwb as _nwb

from .types import PathLike
from . import (
    stdio as _stdio,
    file_metadata as _file_metadata,
    timebases as _timebases,
)


def iterate_raw_daq_recordings(
    metadata: _file_metadata.Metadata,
    rawfile: PathLike,
    timebases: _timebases.Timebases,
    verbose: bool = True,
) -> Iterator[_nwb.TimeSeries]:
    _stdio.message("loading raw DAQ data...", end=' ', verbose=verbose)
    start = _now()
    with _h5.File(rawfile, 'r') as src:
        raw = _np.array(src['behavior_raw/data']).T  # --> shape (T, N)
    assert raw.shape[1] == len(metadata.task.raw_labels)

    t = timebases.raw
    for i, lab in enumerate(metadata.task.raw_labels):
        yield _nwb.TimeSeries(
            name=lab.replace('_raw', ''),  # FIXME; need description
            data=raw[:, i],
            unit="a.u.",  # FIXME: check units
            timestamps=t,
        )
    stop = _now()
    _stdio.message(f"done (took {(stop - start):.1f} sec).", verbose=verbose)


def iterate_downsampled_daq_recordings(
    metadata: _file_metadata.Metadata,
    rawfile: PathLike,
    timebases: _timebases.Timebases,
    verbose: bool = True,
) -> Iterator[_nwb.TimeSeries]:
    _stdio.message("loading down-sampled DAQ data...", end=' ', verbose=verbose)
    start = _now()
    with _h5.File(rawfile, 'r') as src:
        ds = _np.array(src['behavior_ds/data']).T  # --> shape (T, N)
    assert ds.shape[1] == len(metadata.task.downsampled_labels)
    assert ds.shape[0] >= timebases.B.size

    t    = timebases.B
    clip = slice(0, t.size)
    for i, lab in enumerate(metadata.task.downsampled_labels):
        yield _nwb.TimeSeries(
            name=lab.replace('_ds', ''),  # FIXME; need description
            data=ds[clip, i],
            unit="a.u.",  # FIXME: check units
            timestamps=t,
        )
    stop = _now()
    _stdio.message(f"done (took {(stop - start):.1f} sec).", verbose=verbose)
