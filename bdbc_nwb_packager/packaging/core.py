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
    stdio as _stdio,
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
        imgB = _np.array(src["sync_pulse/img_acquisition_start_b"], dtype=_np.uint32).ravel() - 1
        imgV = _np.array(src["sync_pulse/img_acquisition_start_v"], dtype=_np.uint32).ravel() - 1

        if 'vid_acquisition_start' in src['sync_pulse'].keys():
            videoPulse = _np.array(src["sync_pulse/vid_acquisition_start"], dtype=_np.uint32).ravel() - 1
            videoTime  = _np.array(src["tick_in_second/vid"], dtype=_np.float32).ravel()
        else:
            _stdio.message('***found no video pulses', verbose=True)
            videoPulse = None
            videoTime  = None

        trigs = PulseTriggers(
            videos=videoPulse,
            B=imgB,
            V=imgV,
        )
        timebase = Timebases(
            raw=_np.array(src["tick_in_second/raw"], dtype=_np.float32).ravel(),
            videos=videoTime,
            B=_np.array(src["tick_in_second/img_b"], dtype=_np.float32).ravel(),
            V=_np.array(src["tick_in_second/img_v"], dtype=_np.float32).ravel(),
        )
        return trigs, timebase


def validate_timebase_with_imaging(
    rawfile: PathLike,
    triggers: PulseTriggers,
    timebases: Timebases,
    verbose: bool = True,
) -> Tuple[PulseTriggers, Timebases]:
    num_frames = dict()
    with _h5.File(rawfile, 'r') as src:
        num_frames['B'] = src['image/Ib'].shape[0]
        num_frames['V'] = src['image/Iv'].shape[0]

    for chan in num_frames.keys():
        pulses = getattr(triggers, chan)
        timebase = getattr(timebases, chan)

        num_pulses = pulses.size
        if num_pulses < num_frames[chan]:
            raise ValueError(f"the number of frames ({num_frames[chan]}) is larger  than the number of pulses ({num_pulses})")
        elif num_pulses > num_frames[chan]:
            _stdio.message(f"--> trimming {chan} pulses: {num_pulses} --> {num_frames[chan]}")
            triggers = triggers._replace(**{chan: pulses[:num_frames[chan]]})
        else:
            pass

        num_ticks = timebase.size
        if num_ticks < num_frames[chan]:
            raise ValueError(f"the number of frames ({num_frames[chan]}) is larger  than the number of ticks ({num_ticks})")
        elif num_ticks > num_frames[chan]:
            _stdio.message(f"--> trimming {chan} ticks: {num_ticks} --> {num_frames[chan]}")
            timebases = timebases._replace(**{chan: timebase[:num_frames[chan]]})
        else:
            pass

    return (triggers, timebases)


def validate_timebase_with_videos(
    paths: _paths.PathSettings,
    triggers: PulseTriggers,
    timebases: Timebases,
    tolerance: int = 1,
    verbose: bool = True,
) -> Tuple[PulseTriggers, Timebases]:
    if not paths.has_behavior_videos():
        triggers = triggers._replace(videos=None)
        timebases = timebases._replace(videos=None)
    return (triggers, timebases)

