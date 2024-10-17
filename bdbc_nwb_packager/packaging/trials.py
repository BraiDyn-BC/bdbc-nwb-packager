# MIT License
#
# Copyright (c) 2024 Keisuke Sehara, Ryo Aoki and Shoya Sugimoto
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

from typing import Union
from time import time as _now

import numpy as _np
import numpy.typing as _npt
import h5py as _h5
import pandas as _pd
import pynwb as _nwb
from pynwb.epoch import TimeIntervals as _TimeIntervals

from .. import (
    stdio as _stdio,
    trials as _trials,
)
from . import (
    core as _core,
)


PathLike = _core.PathLike


def index_to_timestamp(
    vals: _npt.NDArray[_np.integer],
    time: _npt.NDArray[_np.floating]
) -> _npt.NDArray[_np.float32]:
    out = _np.empty((vals.size,), dtype=_np.float32)
    for i, v in enumerate(vals):
        if v >= 0:
            out[i] = time[v]
        else:
            out[i] = _np.nan
    return out


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
    for col, typ in task.COLUMN_TYPES.items():
        if col in ('start', 'end'):
            continue
        elif typ == 'time':
            trials[f"{col}_time"] = index_to_timestamp(
                trials[col].values,
                timebases.raw,
            )
    return trials


def load_downsampled_trials(
    rawfile: PathLike,
) -> _pd.DataFrame:
    with _h5.File(rawfile, 'r') as src:
        # Loading from hdf5 file
        data        = _np.array(src["behavior_ds/trial_info/data"], dtype=_np.float32).T
        label       = _np.array(src["behavior_ds/trial_info/label"]).ravel()
        label       = [lab.decode('utf-8') for lab in label]  # convert to utf-8
        # convert to dataframe
        trials_ds = _pd.DataFrame(data, columns=label)
        return trials_ds


def format_trials(
    trials: _pd.DataFrame,
    tasktype: str = 'cued_lever_pull',
) -> _pd.DataFrame:
    task = getattr(_trials, tasktype)
    data = dict()
    for name, typ in task.COLUMN_TYPES.items():
        if name == 'start':
            data['start_time'] = trials['start_time'].values
        elif name == 'end':
            data['stop_time'] = trials['stop_time'].values
        elif typ == 'time':
            data[name] = trials[f"{name}_time"].values
        else:
            data[name] = trials[name].values
    return _pd.DataFrame(data=data)


def write_trials(
    parent: Union[_nwb.NWBFile, _nwb.base.ProcessingModule],
    trials: _pd.DataFrame,
    tasktype: str = 'cued_lever_pull',
    verbose: bool = True,
):
    task = getattr(_trials, tasktype)
    if isinstance(parent, _nwb.NWBFile):
        table = None
        _add_column = parent.add_trial_column
        _add_row = parent.add_trial

        def _finalize(tab):
            pass

    else:
        taskname = tasktype.replace('_', '-')
        table = _TimeIntervals(
            name='trials',
            description=f"downsampled trials of the {taskname} task"
        )
        _add_column = table.add_column
        _add_row = table.add_row

        def _finalize(tab):
            parent.add(tab)

    for name, desc in task.DESCRIPTION.items():
        _add_column(name=name, description=desc)

    formatted = format_trials(trials, tasktype=tasktype)

    for _, row in formatted.iterrows():
        _add_row(
            **row.to_dict(),
        )
    _finalize(table)
    _stdio.message("done registration of trials.", verbose=verbose)
