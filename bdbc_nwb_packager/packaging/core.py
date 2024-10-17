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

from typing import Tuple
from collections import namedtuple as _namedtuple
import sys as _sys

import numpy as _np
import numpy.typing as _npt
import h5py as _h5

from .. import (
    metadata as _metadata,
    paths as _paths,
)

PathLike = _paths.PathLike


class Timebases(_namedtuple('Timebases', (
    'raw',
    'videos',
    'B',
    'V',
))):
    @property
    def dFF(self) -> _npt.NDArray[_np.float32]:
        """the timebase for hemodynamics-corrected signals"""
        return self.B


class PulseTriggers(_namedtuple('PulseTriggers', (
    'videos',
    'B',
    'V',
))):
    @property
    def dFF(self) -> _npt.NDArray[_np.integer]:
        """the pulse triggers for hemodynamics-corrected signals"""
        return self.B

    def as_timebases(
        self,
        ref: _npt.NDArray[_np.floating]
    ) -> _npt.NDArray[_np.floating]:
        return Timebases(
            raw=ref,
            videos=ref[self.videos] if self.videos is not None else None,
            B=ref[self.B],
            V=ref[self.V],
        )


def print_message(msg: str, end: str = '\n', verbose: bool = True):
    if verbose:
        print(msg, end=end, file=_sys.stderr, flush=True)


def load_timebases(
    metadata: _metadata.Metadata,
    rawfile: PathLike,
    verbose: bool = True,
) -> Tuple[PulseTriggers, Timebases]:
    with _h5.File(rawfile, 'r') as src:
        # NOTE: indexing is in the MATLAB format:
        # need to subtract 1 to convert to the Python format indices
        imgPulse = _np.array(src["sync_pulse/img_acquisition_start"], dtype=_np.uint32).ravel() - 1
        videoPulse = _np.array(src["sync_pulse/vid_acquisition_start"], dtype=_np.uint32).ravel() - 1

        # TODO: check other entries
        trigs = PulseTriggers(
            videos=videoPulse,
            B=imgPulse[::2],
            V=imgPulse[1::2],
        )
        timebase = Timebases(
            raw=_np.array(src["tick_in_second/raw"], dtype=_np.float32).ravel(),
            videos=_np.array(src["tick_in_second/vid"], dtype=_np.float32).ravel(),
            B=_np.array(src["tick_in_second/img"], dtype=_np.float32).ravel(), 
            V=_np.array(src["tick_in_second/img"], dtype=_np.float32).ravel(),
        )
        return trigs, timebase
