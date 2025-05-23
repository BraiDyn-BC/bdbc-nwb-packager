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
from typing import Optional
from typing_extensions import Self
from dataclasses import dataclass

import numpy as _np
import numpy.typing as _npt
import h5py as _h5

from .types import (
    PathLike,
)
from . import (
    logging as _logging,
    configure as _configure,
    file_metadata as _file_metadata,
)

Timebase = _npt.NDArray[_np.float32]
Indices  = _npt.NDArray[_np.integer]


@dataclass
class Timebases:
    raw: Timebase
    videos: Timebase
    B: Timebase
    V: Timebase

    @property
    def dFF(self) -> Timebase:
        """the timebase for hemodynamics-corrected signals"""
        return self.B

    def replace(
        self,
        raw: Optional[Timebase] = None,
        videos: Optional[Timebase] = None,
        B: Optional[Timebase] = None,
        V: Optional[Timebase] = None,
    ) -> Self:
        alt = dict(raw=raw, videos=videos, B=B, V=V)
        fields = dict()
        for fld, val in alt.items():
            if val is None:
                fields[fld] = getattr(self, fld)
            else:
                fields[fld] = val
        return self.__class__(**fields)


@dataclass
class PulseTriggers:
    videos: Indices
    B: Indices
    V: Indices

    @property
    def dFF(self) -> Indices:
        """the pulse triggers for hemodynamics-corrected signals"""
        return self.B

    def as_timebases(
        self,
        ref: _npt.NDArray[_np.floating]
    ) -> Timebase:
        return Timebases(
            raw=ref,
            videos=ref[self.videos] if self.videos is not None else None,
            B=ref[self.B],
            V=ref[self.V],
        )

    def replace(
        self,
        videos: Optional[Indices] = None,
        B: Optional[Indices] = None,
        V: Optional[Indices] = None,
    ) -> Self:
        alt = dict(videos=videos, B=B, V=V)
        fields = dict()
        for fld, val in alt.items():
            if val is None:
                fields[fld] = getattr(self, fld)
            else:
                fields[fld] = val
        return self.__class__(**fields)


def read_timebases(
    metadata: _file_metadata.Metadata,
    rawfile: PathLike,
    verbose: bool = True,
) -> tuple[PulseTriggers, Timebases]:
    with _h5.File(rawfile, 'r') as src:
        # NOTE: indexing is in the MATLAB format:
        # need to subtract 1 to convert to the Python format indices
        imgB = _np.array(src["sync_pulse/img_acquisition_start_b"], dtype=_np.uint32).ravel() - 1
        imgV = _np.array(src["sync_pulse/img_acquisition_start_v"], dtype=_np.uint32).ravel() - 1

        if 'vid_acquisition_start' in src['sync_pulse'].keys():
            videoPulse = _np.array(src["sync_pulse/vid_acquisition_start"], dtype=_np.uint32).ravel() - 1
            videoTime  = _np.array(src["tick_in_second/vid"], dtype=_np.float32).ravel()
        else:
            _logging.warning("found no video pulses")
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


def validate_timebase_with_rawdata(
    rawfile: PathLike,
    triggers: PulseTriggers,
    timebases: Timebases,
    verbose: bool = True,
) -> tuple[PulseTriggers, Timebases]:
    with _h5.File(rawfile, 'r') as src:
        num_columns, num_samples = src['behavior_raw/data'].shape
    num_timepoints = timebases.raw.size
    if num_timepoints < num_samples:
        raise ValueError(f"the number of timepoints ({num_timepoints}) is smaller than the number of samples ({num_samples})")
    elif num_timepoints > num_samples:
        _logging.debug(f"trimming raw ticks: {num_timepoints} --> {num_samples}")
        timebases = timebases.replace(raw=timebases.raw[:num_samples])
    else:
        pass
    return (triggers, timebases)


def validate_timebase_with_imaging(
    rawfile: PathLike,
    triggers: PulseTriggers,
    timebases: Timebases,
    verbose: bool = True,
) -> tuple[PulseTriggers, Timebases]:
    num_frames = dict()
    with _h5.File(rawfile, 'r') as src:
        num_frames['B'] = src['image/Ib'].shape[0]
        num_frames['V'] = src['image/Iv'].shape[0]

    for chan in num_frames.keys():
        pulses = getattr(triggers, chan)
        timebase = getattr(timebases, chan)

        num_pulses = pulses.size
        if num_pulses < num_frames[chan]:
            raise ValueError(f"the number of frames ({num_frames[chan]}) is larger than the number of pulses ({num_pulses})")
        elif num_pulses > num_frames[chan]:
            _logging.debug(f"trimming {chan} pulses: {num_pulses} --> {num_frames[chan]}")
            triggers = triggers.replace(**{chan: pulses[:num_frames[chan]]})
        else:
            pass

        num_ticks = timebase.size
        if num_ticks < num_frames[chan]:
            raise ValueError(f"the number of frames ({num_frames[chan]}) is larger  than the number of ticks ({num_ticks})")
        elif num_ticks > num_frames[chan]:
            _logging.debug(f"trimming {chan} ticks: {num_ticks} --> {num_frames[chan]}")
            timebases = timebases.replace(**{chan: timebase[:num_frames[chan]]})
        else:
            pass

    return (triggers, timebases)


def validate_timebase_with_videos(
    paths: _configure.PathSettings,
    triggers: PulseTriggers,
    timebases: Timebases,
    tolerance: int = 1,
    verbose: bool = True,
) -> tuple[PulseTriggers, Timebases]:
    if not paths.has_behavior_videos():
        triggers = triggers.replace(videos=None)
        timebases = timebases.replace(videos=None)
    return (triggers, timebases)
