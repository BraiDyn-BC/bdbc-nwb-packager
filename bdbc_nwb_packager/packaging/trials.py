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
from time import time as _now

import numpy as _np
import numpy.typing as _npt
import h5py as _h5
import pandas as _pd
import pynwb as _nwb
from hdmf.common import DynamicTable as _DynamicTable

from .. import (
    stdio as _stdio,
    trials as _trials,
    alignment as _alignment,
)
from . import (
    core as _core,
)


PathLike = _core.PathLike


def load_trials(
    rawfile: PathLike,
    timebases: _core.Timebases,
    tasktype: str = 'cued_lever_pull',
    sampling_rate: float = 5000.0,
    verbose: bool = True,
) -> _pd.DataFrame:
    task = getattr(_trials, tasktype)
    _stdio.message(f"loaded task type: '{tasktype}'", verbose=verbose)
    _stdio.message("loading trial data...", end=' ', verbose=verbose)
    start = _now()
    daq = _trials.load_raw_daq(rawfile)
    stop = _now()
    _stdio.message(f"done (took {(stop - start):.1f} sec).", verbose=verbose)
    trials = task.extract_trials(daq, rate=sampling_rate)
    _stdio.message("done parsing trials.", verbose=verbose)
    trials['start_time'] = timebases.raw[trials.start.values]
    trials['stop_time']  = timebases.raw[trials.end.values]
    return trials


def setup_downsampled_trials(
    trials: _pd.DataFrame,
    timebases: _core.Timebases,
    tasktype: str = 'cued_lever_pull',
    verbose: bool = True,
) -> _DynamicTable:
    task = getattr(_trials, tasktype)
    trials_ds = _alignment.align_trials_to_pulses(
        trials,
        timebases.raw,
        timebases.dFF,
        columnsettings=task.COLUMN_TYPES,
    )
    trials_ds['start_time'] = timebases.dFF[trials_ds.start.values]
    trials_ds['stop_time']  = timebases.dFF[trials_ds.end.values]

    taskname = tasktype.replace('_', '-')
    trials = _DynamicTable(
        name='trials',
        description=f"downsampled trials of the {taskname} task"
    )
    trials.add_column('start_time', description='the starting time of the trial')
    trials.add_column('stop_time', description='the ending time of the trial')
    for name, desc in task.DESCRIPTION.items():
        trials.add_column(name=name, description=desc)

    columns = []
    for col in task.COLUMN_TYPES.keys():
        if col == 'start':
            col = 'start_time'
        elif col == 'end':
            col = 'stop_time'
        columns.append(col)

    for _, row in trials_ds.iterrows():
        trials.add_row(
            **dict((fld, trials_ds[fld].values) for fld in columns)
        )
    _stdio.message(f"set up downsampled trials.", verbose=verbose)
    return trials


def write_trials(
    nwbfile: _nwb.NWBFile,
    trials: _pd.DataFrame,
    tasktype: str = 'cued_lever_pull',
    verbose: bool = True,
):
    task = getattr(_trials, tasktype)
    for name, desc in task.DESCRIPTION.items():
        nwbfile.add_trial_column(name=name, description=desc)

    columns = []
    for col in task.COLUMN_TYPES.keys():
        if col == 'start':
            col = 'start_time'
        elif col == 'end':
            col = 'stop_time'
        columns.append(col)

    for _, row in trials.iterrows():
        nwbfile.add_trial(
            **dict((fld, row[fld]) for fld in columns)
        )
    _stdio.message("done registration of trials.", verbose=verbose)

