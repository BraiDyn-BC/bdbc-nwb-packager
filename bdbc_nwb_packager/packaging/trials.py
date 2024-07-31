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

from typing import Dict
from pathlib import Path

import numpy as _np
import numpy.typing as _npt
import h5py as _h5
import pandas as _pd
import pynwb as _nwb

from . import (
    core as _core,
)

#
# FIXME
#
# 1. Why are the trial-start / lever timing / reward timing variables in 'frame indices' format?
#    wouldn't it be better use the direct time stamps based on the DAQ log?
# 2. Do the reaction time / lever timing refer to the 'onset' of pull behavior?
# 3. Probably better to explicitly specify the 'end of the trial' based on the task state in the DAQ log
#

PathLike = _core.PathLike

COLUMNS_RAW = ('TrialStartBidx', 'LeverTimingBidx', 'RewardTimingBidx', 'ReactionTimeMs', 'TaskPullTimeMs')
COLUMNS_NWB = {
    "lever": "timings of successful lever pulls [s]",
    "reward": "reward delivery timinigs (NaNs indicating failed trials) [s]",
    "reaction_time": "reaction time, from the cue onset to the successful lever pull (NaNs indicating failed trials) [s]",
    "required_pull_duration": "task-specified minimum lever pull duration for the trial to be successful [s]",
}


def convert_row_to_nwb(row) -> Dict[str, float]:
    return {
        'start_time': row.Start,
        'stop_time': row.Stop,
        'lever': row.LeverPull,
        'reward': row.Reward,
        'reaction_time': row.ReactionTime,
        'required_pull_duration': row.RequiredPullDuration,
    }


def load_trials(
    rawfile: PathLike,
    timebases: _core.Timebases,
) -> _pd.DataFrame:
    
    t = timebases.B
    
    def _as_python_indices(x) -> _npt.NDArray[_np.int32]:
        x = _np.array(x, copy=True)
        x[_np.isnan(x)] = 0
        return x.astype(_np.int32) - 1
    
    def _interpret_index(idx) -> float:
        if idx >= 0:
            return t[int(idx)]
        else:
            return _np.nan
    
    def _process(row) -> Dict[str, float]:
        return {
            'Start': _interpret_index(row.TrialStartBidx),
            'Stop': _interpret_index(row.TrialStopBidx),
            'LeverPull': _interpret_index(row.LeverTimingBidx),
            'Reward': _interpret_index(row.RewardTimingBidx),
            'ReactionTime': row.ReactionTimeMs / 1000,
            'RequiredPullDuration': row.TaskPullTimeMs / 1000,
        }
    
    with _h5.File(rawfile, "r") as src:
        raw = _np.array(src["behavior_ds/trial_info"], dtype=_np.float32)
    trials_base = _pd.DataFrame(data=dict((col, raw[i]) for i, col in enumerate(COLUMNS_RAW)))
    
    # note that the indexing is in the MATLAB stype
    for col in COLUMNS_RAW[:3]:
        trials_base[col] = _as_python_indices(trials_base[col])
    trials_base['TrialStopBidx'] = _np.concatenate([trials_base.TrialStartBidx[1:] + 1, (t.size - 1,)])
    return _pd.DataFrame(data=[_process(row) for _, row in trials_base.iterrows()])


def write_trials(
    nwbfile: _nwb.NWBFile,
    trials: _pd.DataFrame,
    verbose: bool = True,
):
    for name, desc in COLUMNS_NWB.items():
        nwbfile.add_trial_column(name=name, description=desc)

    for _, row in trials.iterrows():
        nwbfile.add_trial(
            **convert_row_to_nwb(row)
        )
    _core.print_message("done registration of trials.", verbose=verbose)
