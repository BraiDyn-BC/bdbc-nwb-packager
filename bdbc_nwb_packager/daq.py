# MIT License
#
# Copyright (c) 2024-2025 Keisuke Sehara, Ryo Aoki, and Shoya Sugimoto
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
    logging as _logging,
    file_metadata as _file_metadata,
    timebases as _timebases,
)


def iterate_raw_daq_recordings(
    metadata: _file_metadata.Metadata,
    rawfile: PathLike,
    timebases: _timebases.Timebases,
    verbose: bool = True,
) -> Iterator[_nwb.TimeSeries]:
    _logging.info("start retrieving raw DAQ data")
    start = _now()
    with _h5.File(rawfile, 'r') as src:
        raw = _np.array(src['behavior_raw/data']).T  # --> shape (T, N)
    assert raw.shape[1] == len(metadata.task.raw_labels)

    t = timebases.raw
    num = 0
    for i, lab in enumerate(metadata.task.raw_labels):
        lab = lab.replace('_raw', '').strip()  # FIXME; need description
        if len(lab) == 0:
            continue
        _logging.debug(f"found record: {lab}")
        yield _nwb.TimeSeries(
            name=lab,
            data=raw[:, i],
            unit="a.u.",  # FIXME: check units
            timestamps=t,
        )
        num += 1
    stop = _now()
    _logging.info(f"done retrieving {num} time series (took {(stop - start):.1f} sec).")


def iterate_downsampled_daq_recordings(
    metadata: _file_metadata.Metadata,
    rawfile: PathLike,
    timebases: _timebases.Timebases,
    verbose: bool = True,
) -> Iterator[_nwb.TimeSeries]:
    _logging.info("start retrieving down-sampled DAQ data")
    start = _now()
    with _h5.File(rawfile, 'r') as src:
        ds = _np.array(src['behavior_ds/data']).T  # --> shape (T, N)
    assert ds.shape[1] == len(metadata.task.downsampled_labels)
    assert ds.shape[0] >= timebases.B.size

    t    = timebases.B
    num  = 0
    clip = slice(0, t.size)
    for i, lab in enumerate(metadata.task.downsampled_labels):
        lab = lab.replace('_ds', '').strip()  # FIXME; need description
        if len(lab) == 0:
            continue
        _logging.debug(f"found record: {lab}")
        yield _nwb.TimeSeries(
            name=lab,
            data=ds[clip, i],
            unit="a.u.",  # FIXME: check units
            timestamps=t,
        )
        num += 1
    stop = _now()
    _logging.info(f"done retrieving {num} time series (took {(stop - start):.1f} sec).")
