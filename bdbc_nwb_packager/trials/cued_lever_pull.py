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

import numpy as _np
import pandas as _pd

from . import (
    common as _common,
)


COLUMN_TYPES = {
    'required_pull_duration': 'value',
    'start': 'time',  # required
    'end': 'time',  # required
    'lever': 'time',
    'reaction_time': 'value',
    'reward': 'value',
}

DESCRIPTION = {
    "lever": "timings of successful lever pulls [s]",
    "reward": "reward delivery timinigs (NaNs indicating failed trials) [s]",
    "reaction_time": "reaction time, from the cue onset to the successful lever pull (NaNs indicating failed trials) [s]",
    "required_pull_duration": "task-specified minimum lever pull duration for the trial to be successful [s]",
}


def extract_trials(daq: _pd.DataFrame, rate: float = 5000.0) -> _pd.DataFrame:
    # task state: WAITING(0), CUED(1), REWARDED(2)
    state = daq.Task_state.values.astype(_np.int8)
    task_state = _common.extract_blocks(state)

    # (supra-threshold) lever responses
    resps = _common.extract_blocks(daq.Over_thr > 0)
    resps = resps.loc[resps.value == True].drop(['value'], axis=1).start.values
    
    trials = []
    statesize = task_state.shape[0]
    respsize = resps.shape[0]
    stateoffset = 0
    respoffset = 0
    while stateoffset < statesize:
        if task_state.value[stateoffset] != 0:  # WAITING
            stateoffset += 1
            continue
        if (stateoffset + 2) >= statesize:
            break
        assert task_state.value[stateoffset + 1] == 1  # CUED
        trial_start = task_state.start[stateoffset + 1]
        trial_end = task_state.stop[stateoffset + 1]

        # detect response
        resp = -1 # NA by default
        while (respoffset < respsize) and (resps[respoffset] < task_state.start[stateoffset + 1]):
            respoffset += 1
        if (respoffset < respsize) and (resps[respoffset] < task_state.stop[stateoffset + 1]):
            resp = resps[respoffset]
        reaction = (resp - trial_start) / rate if resp >= 0 else _np.nan

        # detect trial type
        if task_state.value[stateoffset + 2] == 2:  # REWARDED
            rewarded = True
            stateoffset += 3
        else:
            rewarded = False
            stateoffset += 2
        trials.append({
            'required_pull_duration': daq.Pull_dur[trial_start] / 5000,
            'start': trial_start,
            'end': trial_end,
            'lever': resp,
            'reaction_time': reaction,
            'reward': rewarded,
        })
    return _pd.DataFrame(data=trials)

