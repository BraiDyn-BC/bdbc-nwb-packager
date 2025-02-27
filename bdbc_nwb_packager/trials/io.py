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
from typing import Optional, Union

import numpy as _np
import pandas as _pd
import h5py as _h5
import pynwb as _nwb

import bdbc_session_explorer as _sessx
from ..types import PathLike
from .. import (
    logging as _logging,
)
from . import (
    spec as _spec,
)


def trials_from_group(
    group: _h5.Group,
    trialspec: _sessx.TrialSpec
) -> Optional[_spec.Trials]:
    data   = _np.array(group["data"], dtype=_np.float32).T
    labels = _np.array(group["label"]).ravel()
    labels = [lab.decode('utf-8') for lab in labels]  # convert to utf-8
    _logging.debug(f"trial table shape: {data.shape}")
    _logging.debug(f"trial columns: {labels}")

    # validation
    # there can be sessions without any trials (i.e. resting-state)
    if data.shape[0] == 0:
        return None

    # convert to dataframe
    table = _pd.DataFrame(data, columns=labels)
    trialspec = _spec.TrialSpec.from_dict(trialspec)
    return _spec.Trials(table=table, metadata=trialspec)


def load_trials(
    rawfile: PathLike,
    trialspec: _sessx.TrialSpec
) -> Optional[_spec.Trials]:
    with _h5.File(rawfile, 'r') as src:
        # Loading from hdf5 file
        return trials_from_group(src['behavior_raw/trial_info'], trialspec=trialspec)


def load_downsampled_trials(
    rawfile: PathLike,
    trialspec: _sessx.TrialSpec
) -> _pd.DataFrame:
    with _h5.File(rawfile, 'r') as src:
        # Loading from hdf5 file
        return trials_from_group(src['behavior_ds/trial_info'], trialspec=trialspec)


def write_trials(
    parent: Union[_nwb.NWBFile, _nwb.base.ProcessingModule],
    trials: _spec.Trials,
    verbose: bool = True,
):
    is_root = isinstance(parent, _nwb.NWBFile)
    desc = f"trials of the '{trials.metadata.name}' session"
    if not is_root:
        desc = "downsampled " + desc
    trials_table = _nwb.epoch.TimeIntervals(
        name='trials',
        description=desc
    )

    if is_root:
        def _finalize(tab):
            parent.trials = tab
    else:
        def _finalize(tab):
            parent.add(tab)

    for column in trials.metadata.required_columns:
        # a dirty hack to override descriptions
        _logging.debug(f"writing column: {column.name}")
        desc = column.format_description()
        _logging.debug(f"column description: {desc}")
        trials_table[column.name].fields['description'] = desc
    for column in trials.metadata.task_specific_columns:
        _logging.debug(f"writing column: {column.name}")
        desc = column.format_description()
        _logging.debug(f"column description: {desc}")
        trials_table.add_column(name=column.name, description=desc)

    for trial in trials.iter_trials_as_dict():
        trials_table.add_row(**trial)
    _finalize(trials_table)
