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

from typing import Callable, Tuple
from typing_extensions import Self
from pathlib import Path
from collections import namedtuple as _namedtuple

import numpy as _np
import numpy.typing as _npt
import pandas as _pd

VALIDATION_ALPHA = 2.5  # must be smaller (e.g. 0.5)
VALIDATION_THRESHOLD = 0.88  # must be higher (e.g. 0.9999)


class PointEstimation(_namedtuple('PointEstimation', (
    'x',
    'y',
))):
    def apply(
        self,
        fn: Callable[[_npt.NDArray], _npt.NDArray]
    ) -> Self:
        return self.__class__(
            **dict((fld, fn(val)) for fld, val in zip(self._fields, self))
        )

    def stack(self, axis: int = 1) -> _npt.NDArray:
        return _np.stack(self, axis=axis)


class IndexRanges(_namedtuple('IndexRanges', (
    'pulses',
    'frames',
))):
    pass


def validate_keypoint(
    dlcdata: _pd.DataFrame,
    keypoint: str,
    alpha: float = VALIDATION_ALPHA,
    threshold: float = VALIDATION_THRESHOLD,
) -> PointEstimation:
    def _by_percentile(v):
        return _np.logical_and(
            v >= _np.nanpercentile(v, alpha),
            v <= _np.nanpercentile(v, 100 - alpha),
        )    
    scorer = dlcdata.columns[0][0]
    x = _np.array(dlcdata[scorer, keypoint, 'x'].values)
    y = _np.array(dlcdata[scorer, keypoint, 'y'].values)
    
    # likelihood-based filtering
    valid = dlcdata[scorer, keypoint, 'likelihood'].values >= threshold
    x[~valid] = _np.nan
    y[~valid] = _np.nan
    
    # percentile-based filtering
    valid = _np.logical_and(
        _by_percentile(x),
        _by_percentile(y),
    )
    x[~valid] = _np.nan
    y[~valid] = _np.nan
    return PointEstimation(x, y)


def validate_index_ranges(
    num_frames: int,
    num_pulses: int,
    mismatch_tolerance: int = 1
) -> IndexRanges:
    """a temporary solution until sizes of timebases/videos are more nicely handled."""
    delta = num_frames - num_pulses
    if delta == 0:
        pulserange = slice(None, None)
        framerange = slice(None, None)
    elif 0 < delta <= mismatch_tolerance:
        pulserange  = slice(None, None)
        framerange = slice(0, num_pulses)
    elif (-1 * mismatch_tolerance) <= delta < 0:
        pulserange = slice(0, num_frames)
        framerange = slice(None, None)
    else:
        raise RuntimeError(f"untolerable mismatch: {num_pulses} pulses vs {num_frames} frames")
    return pulserange, framerange


def prepare_table_results(
    tabpath: Path,
    srcvideo: object,
    t_video: _npt.NDArray[_np.floating],
    triggers: _npt.NDArray[_np.integer],
    entry_path: str = 'df_with_missing',
    mismatch_tolerance: int = 1,
) -> Tuple[_npt.NDArray[_np.floating], _npt.NDArray[_np.integer], _pd.DataFrame]:
    # FIXME: this procedure must be (formally)
    # during registration of videos
    tclip, vclip = validate_index_ranges(
        num_frames=srcvideo.num_frames,
        num_pulses=t_video.size,
        mismatch_tolerance=mismatch_tolerance,
    )
    
    tab = _pd.read_hdf(tabpath, key=entry_path)
    tab = tab.iloc[vclip]
    t = t_video[tclip]
    trigs = triggers[tclip]
    assert tab.shape[0] == t.size
    return (t, trigs, tab)

