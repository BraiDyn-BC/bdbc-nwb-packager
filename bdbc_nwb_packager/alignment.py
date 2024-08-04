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

from typing import Callable, Dict
import warnings as _warnings

import numpy as _np
import numpy.typing as _npt
import pandas as _pd


def timebase_to_daqindices(
    daq_t: _npt.NDArray[_np.floating],
    pulse_t: _npt.NDArray[_np.floating],
    tol: float = 1e-6,
) -> _npt.NDArray[_np.int32]:
    """better directly using the PulseTriggers instances"""
    out = _np.empty((pulse_t.size,), dtype=_np.int32)
    out.fill(-1)
    daqsize = daq_t.size
    offset = 0
    for i, pulsetrig in enumerate(pulse_t):
        while (offset < daqsize) and (abs(pulsetrig - daq_t[offset]) > tol):
            offset += 1
        if offset == daqsize:
            break
        out[i] = offset
    return out


def daqindex_to_pulseindex(
    daqidxx: _npt.NDArray[_np.integer],
    daq_t: _npt.NDArray[_np.floating],
    pulse_t: _npt.NDArray[_np.floating],
) -> _npt.NDArray[_np.int32]:
    """`daqidxx assumed to be sorted in the ascending order,
    except for entries with -1, which indicates that the value
    is missing there.

    the returning array will contain -2, which indicates
    "out of the period where pulses are defined".
    """
    out = _np.empty(daqidxx.size, dtype=_np.int32)
    out.fill(-2)
    pulsesiz = pulse_t.size
    offset = 0
    for i, daqidx in enumerate(daqidxx):
        if daqidx == -1:
            out[i] = -1
            continue
        
        daqtrig = daq_t[daqidx]
        while ((offset + 1) < pulsesiz) and (pulse_t[offset + 1] < daqtrig):
            offset += 1
        if offset == 0:
            # before the beginning of the pulses
            continue
        elif offset == (pulsesiz - 1):
            # at the end of the pulses
            break
        out[i] = offset
    return out


def align_trials_to_pulses(
    trials: _pd.DataFrame,
    daq_t: _npt.NDArray[_np.floating],
    pulse_t: _npt.NDArray[_np.floating],
    columnsettings: Dict[str, str]
) -> _pd.DataFrame:
    for required in ('start', 'end'):
        if required not in columnsettings.keys():
            raise ValueError(f"trials do not contain the '{required}' column")
    aligned = dict()
    for col, typ in columnsettings.items():
        if typ == 'time':
            aligned[col] = daqindex_to_pulseindex(trials[col].values, daq_t, pulse_t)
        elif typ == 'value':
            aligned[col] = trials[col].values
        else:
            raise ValueError(f"unexpected column type: {typ}")
    aligned = _pd.DataFrame(data=aligned)
    return aligned.loc[~_np.logical_or(aligned.start == -2, aligned.end == -2)]


def upsample(
    values: _npt.NDArray[_np.floating],
    size: int,
    pulseidxx: _npt.NDArray[_np.integer],
    max_skips: int = 1,
) -> _npt.NDArray[_np.float32]:
    out = _np.empty((size,), dtype=_np.float32)
    out.fill(_np.nan)
    offsetceil = pulseidxx.size - 1  # exclude the last one
    offset = 0
    
    def _linear(start, stop, vstart, vend):
        siz = stop - start
        vals = _np.arange(0, siz) / (siz - 1)  # [0, 1]
        return vals * (vend - vstart) + vstart
        
    offsetceil = pulseidxx.size - 1  # exclude the last one
    offset = 0
    while offset < offsetceil:
        if _np.isnan(values[offset]):
            offset += 1
            continue
        elif ~_np.isnan(values[offset + 1]):
            start = pulseidxx[offset]
            stop  = pulseidxx[offset + 1] + 1
            out[start:stop] = _linear(start, stop, values[offset], values[offset + 1])
            offset += 1
        else:
            for skip in range(1, max_skips + 1):
                if ((offset + skip) < offsetceil) and ~_np.isnan(values[offset + skip + 1]):
                    start = pulseidxx[offset]
                    stop  = pulseidxx[offset + skip + 1] + 1
                    out[start:stop] = _linear(
                        start, stop,
                        values[offset], values[offset + skip + 1]
                    )
                    offset += skip  # further incremented below
                    break
            # finally (not interpolated other than the block above)
            offset += 1
    return out


def downsample(
    values: _npt.NDArray[_np.floating],
    pulseidxx: _npt.NDArray[_np.integer],
    reduce: Callable[[_npt.NDArray[_np.floating]], float] = _np.nanmean,
) -> _npt.NDArray[_np.float32]:
    out = _np.empty((pulseidxx.size,), dtype=_np.float32)
    out.fill(_np.nan)
    interval = round(_np.diff(pulseidxx).mean())
    with _warnings.catch_warnings():
        _warnings.filterwarnings(
            'ignore',
            message='mean of empty slice',
            category=RuntimeWarning
        )
        for i, start, stop in zip(
            range(pulseidxx.size - 1),
            pulseidxx[:-1],
            pulseidxx[1:]):
            out[i] = reduce(values[start:stop])
        start = pulseidxx[-1]
        stop  = min(values.size, start + interval)
        out[-1] = reduce(values[start:stop])
    return out

