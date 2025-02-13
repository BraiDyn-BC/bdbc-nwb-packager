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

from typing import Callable
import warnings as _warnings

import numpy as _np
import numpy.typing as _npt


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
            pulseidxx[1:]
        ):
            out[i] = reduce(values[start:stop])
        start = pulseidxx[-1]
        stop  = min(values.size, start + interval)
        out[-1] = reduce(values[start:stop])
    return out
