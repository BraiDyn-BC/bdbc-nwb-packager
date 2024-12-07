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
from typing import Optional, Union

import numpy as _np
import pandas as _pd
import h5py as _h5
import pynwb as _nwb

from ..types import PathLike
from . import (
    spec as _spec,
)


def load_trials(
    rawfile: PathLike,
) -> Optional[_pd.DataFrame]:
    with _h5.File(rawfile, 'r') as src:
        # Loading from hdf5 file
        data   = _np.array(src["behavior_raw/trial_info/data"], dtype=_np.float32).T
        labels = _np.array(src["behavior_raw/trial_info/label"]).ravel()
        labels = [lab.decode('utf-8') for lab in labels]  # convert to utf-8

        # validation
        # there can be sessions without any trials (i.e. resting-state)
        if data.shape[0] == 0:
            return None

        # convert to dataframe
        trials_raw = _pd.DataFrame(data, columns=labels)
        return trials_raw


def load_downsampled_trials(
    rawfile: PathLike,
) -> _pd.DataFrame:
    with _h5.File(rawfile, 'r') as src:
        # Loading from hdf5 file
        data  = _np.array(src["behavior_ds/trial_info/data"], dtype=_np.float32).T
        labels = _np.array(src["behavior_ds/trial_info/label"]).ravel()
        labels = [lab.decode('utf-8') for lab in labels]  # convert to utf-8

        # validation
        # there can be sessions without any trials (i.e. resting-state)
        if data.shape[0] == 0:
            return None

        # convert to dataframe
        trials_ds = _pd.DataFrame(data, columns=labels)
        return trials_ds


def write_trials(
    parent: Union[_nwb.NWBFile, _nwb.base.ProcessingModule],
    trials: _pd.DataFrame,
    tasktype: str = 'cued_lever_pull',
    verbose: bool = True,
):
    trial_spec = _spec.get_spec(tasktype)
    if trial_spec is None:
        raise ValueError(f"unknown task type: {tasktype}")

    if isinstance(parent, _nwb.NWBFile):
        table = None
        _add_column = parent.add_trial_column
        _add_row = parent.add_trial

        def _finalize(tab):
            pass

    else:
        taskname = tasktype.replace('_', '-')
        table = _nwb.epoch.TimeIntervals(
            name='trials',
            description=f"downsampled trials of the {taskname} task"
        )
        _add_column = table.add_column
        _add_row = table.add_row

        def _finalize(tab):
            parent.add(tab)

    for column in trial_spec.task_specific_columns:
        _add_column(name=column.name, description=column.description)

    for trial in trial_spec.iter_trials_from(trials):
        _add_row(**trial)
    _finalize(table)
