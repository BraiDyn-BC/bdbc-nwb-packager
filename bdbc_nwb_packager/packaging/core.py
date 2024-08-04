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
from pathlib import Path
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
        sizB = src['image/Ib'].shape[0]
        sizV = src['image/Iv'].shape[0]

        # NOTE: indexing is in the MATLAB format:
        # need to subtract 1 to convert to the Python format indices
        if 'vid_acq' in src['sync_pulse'].keys():
            videos = _np.array(src["sync_pulse/vid_acq"], dtype=_np.uint32).ravel() - 1 
        else:
            videos = None
        
        # imaging pulses
        img_raw = _np.array(src["sync_pulse/img_acq"], dtype=_np.uint32).ravel() - 1
        if img_raw.size % 2 != 0:
            img_raw = _np.concatenate([raw, (0,)])
        img_raw = img_raw.reshape((-1, 2))
        assert sizB <= img_raw.shape[0]
        assert sizV <= img_raw.shape[0]
        
        pulseB = img_raw[:sizB, 0] - 1
        pulseV = img_raw[:sizV, 0] - 1
    
        # task acquisition
        _, acqsiz = src['behavior_raw/data'].shape  # (N, T)
        raw_t = _np.arange(0, acqsiz, dtype=_np.float32) / metadata.task.rate

        print_message("done loading time bases.", verbose=verbose)
        
        trigs = PulseTriggers(
            videos=videos,
            B=pulseB,
            V=pulseV,
        )
        return trigs, trigs.as_timebases(raw_t)

