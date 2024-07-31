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

from typing import Tuple, Generator, Union, Optional
from collections import namedtuple as _namedtuple

import numpy as _np
import pandas as _pd
from pynwb import (
    TimeSeries as _TimeSeries,
)
from pynwb.behavior import (
    SpatialSeries as _SpatialSeries,
    PupilTracking as _PupilTracking,
    EyeTracking as _EyeTracking,
)

from .. import (
    paths as _paths,
)
from . import (
    core as _core,
)

Tracking = Union[_EyeTracking, _PupilTracking]

DESCRIPTION = {
    'center_x': 'x-position of the center of the pupil',
    'center_y': 'y-position of the center of the pupil',
    'diameter': 'pupil diameter extracted from the video of the right eye',
}


class PupilFittingData(_namedtuple('PupilFittingData', (
    'eye',  # EyeTracking
    'pupil_dia',  # PupilTracking
))):
    def items(self) -> Generator[Tuple[str, Tracking], None, None]:
        for fld, val in zip(self._fields, self):
            yield fld, val


def empty_data(
    timebases: _core.Timebases,
) -> Optional[PupilFittingData]:
    # t = timebases.videos
    # values = _np.empty((t.size,), dtype=_np.float32)
    # values.fill(_np.nan)
    
    # center_x = _SpatialSeries(
    #     name = 'center_x',
    #     description = f"(data missing) {DESCRIPTION['center_x']}",
    #     data = values,
    #     timestamps = t,
    #     reference_frame = 'top left',
    #     unit = 'pixels'
    # )
    # center_y = _SpatialSeries(
    #     name = 'center_y',
    #     description = f"(data missing) {DESCRIPTION['center_y']}",
    #     data = values,
    #     timestamps = t,
    #     reference_frame = 'top left',
    #     unit = 'pixels'
    # )
    # eye = _EyeTracking(name="eye_position", spatial_series=center_x)
    # eye.add_spatial_series(spatial_series=center_y)

    # pupil_dia = _TimeSeries(
    #     name="diameter",
    #     description=f"(data missing) {DESCRIPTION['diameter']}",
    #     data=data.D.values,
    #     timestamps=t,
    #     unit="pixels",
    # )
    # pupil = _PupilTracking(time_series=pupil_dia, name="pupil_tracking")
    # return PupilFittingData(eye=eye, pupil_dia=pupil)
    return None


def load_pupil_fitting(
    paths: _paths.PathSettings,
    timebases: _core.Timebases,
    verbose: bool = True,
) -> Optional[PupilFittingData]:
    if not paths.source.pupilfitting.exists():
        _core.print_message(f"***pupil fitting results do not exist", verbose=verbose)
        return empty_data(timebases)
    
    data = _pd.read_hdf(paths.source.pupilfitting, key="df_with_missing")
    t = timebases.videos
    center_x = _SpatialSeries(
        name = 'center_x',
        description = DESCRIPTION['center_x'],
        data = data.cx.values,
        timestamps = t,
        reference_frame = 'top left',
        unit = 'pixels'
    )
    center_y = _SpatialSeries(
        name = 'center_y',
        description = DESCRIPTION['center_y'],
        data = data.cy.values,
        timestamps = t,
        reference_frame = 'top left',
        unit = 'pixels'
    )
    eye = _EyeTracking(name="eye_position", spatial_series=center_x)
    eye.add_spatial_series(spatial_series=center_y)

    pupil_dia = _TimeSeries(
        name="diameter",
        description=DESCRIPTION['diameter'],
        data=data.D.values,
        timestamps=t,
        unit="pixels",
    )
    pupil = _PupilTracking(time_series=pupil_dia, name="pupil_tracking")
    return PupilFittingData(eye=eye, pupil_dia=pupil)
