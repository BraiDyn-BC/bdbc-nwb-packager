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

from typing import Optional, Union, Iterable, ClassVar, Dict, Generator
from typing_extensions import Self
from dataclasses import (
    dataclass,
)

import numpy as _np
import h5py as _h5
import pandas as _pd
import pynwb as _nwb
from pynwb.epoch import TimeIntervals as _TimeIntervals

from .. import (
    stdio as _stdio,
    # trials as _trials,   # UNUSED
)
from . import (
    core as _core,
)


PathLike = _core.PathLike

# TODO: load the dictionary below from e.g. a JSON file in the future...
CUED_LEVER_PULL = {
    'columns': [
        {
            'name': 'trial_start',
            'output_name': 'start_time',
            'description': 'the timing of the cue onset',
        },
        {
            'name': 'trial_end',
            'output_name': 'stop_time',
            'description': 'the timing when the trial outcome is determined',
        },
        {
            'name': 'pull_onset',
            'description': 'the timing when the animal started to pull the lever',
        },
        {
            'name': 'reaction_time',
            'description': "the time interval (in seconds) from the cue onset to when the animal started to pull the lever",
        },
        {
            'name': 'pull_duration_for_success',
            'description': 'the duration (in seconds) that the animal was required to pull the lever to obtain reward for the trial',
        },
        {
            'name': 'trial_outcome',
            'data_type': 'int',  # FIXME
            'description': 'the outcome of the trial',
        }
    ]
}

SENSORY_STIM = {
    'columns': [
        {
            'name': 'trial_start',
            'output_name': 'start_time',
            'description': 'the timing of the stimulus onset',
        },
        {
            'name': 'trial_end',
            'output_name': 'stop_time',
            'description': 'the timing of the stimulus offset',
        },
        {
            'name': 'stim_modality',
            'data_type': 'int', # FIXME
            'description': "the modality of the stimulus: `visual (1)`, a flash of LED in front of the animal's eye; `auditory (2)`, a brief buzz of white noise from the speaker on the front-left side of the animal; `somatosensory (3)`, a brief vibration to the right whisker-pad of the animal"
        }
    ],
}


FieldType = Union[str, int, float]


def parse_data_type(typespec: str) -> type:
    if typespec == 'str':
        return str
    elif typespec == 'int':
        return int
    elif typespec == 'float':
        return float
    else:
        raise ValueError(f"expected one of ('str', 'int', 'float'), got {repr(typespec)}")


@dataclass
class ColumnSpec:
    input_name: str
    output_name: str
    data_type: type = float
    description: str = ''

    @property
    def name(self):
        return self.output_name

    def get_value_from(self, row: Dict[str, FieldType]) -> FieldType:
        return self.data_type(row[self.input_name])

    def to_dict(self) -> Dict[str, str]:
        return {
            'input_name': self.input_name,
            'output_name': self.output_name,
            'data_type': self.data_type.__name__,
            'description': self.description,
        }

    @classmethod
    def from_dict(cls, dct: Dict[str, str]) -> Self:
        return cls(
            input_name=str(dct['name']),
            output_name=str(dct.get('output_name', dct['name'])),
            data_type=parse_data_type(str(dct.get('data_type', 'float'))),
            description=str(dct.get('description', ''))
        )


@dataclass
class TrialSpec:
    columns: Iterable[ColumnSpec] = ()
    REQUIRED_COLUMNS: ClassVar[Iterable[str]] = ('start_time', 'stop_time')

    def __post_init__(self):
        self.columns = tuple(self.columns)
        if (len(self.columns) == 0) or any(
            col not in self.column_names for col in self.REQUIRED_COLUMNS
        ):
            raise ValueError('at least two columns (`start_time` and `stop_time`) are needed')

    @property
    def column_names(self) -> Iterable[str]:
        return tuple(col.name for col in self.columns)

    @property
    def required_columns(self) -> Generator[ColumnSpec, None, None]:
        for column in self.columns:
            if column.name in self.REQUIRED_COLUMNS:
                yield column

    @property
    def task_specific_columns(self) -> Generator[ColumnSpec, None, None]:
        for column in self.columns:
            if column.name in self.REQUIRED_COLUMNS:
                continue
            yield column

    def iter_trials_from(
        self,
        trials: _pd.DataFrame
    ) -> Generator[Dict[str, FieldType], None, None]:
        for _, row in trials.iterrows():
            row = row.to_dict()
            yield dict((col.name, col.get_value_from(row)) for col in self.columns)

    def to_dict(self) -> Dict[str, Union[str, Iterable[Dict[str, str]]]]:
        return {
            'columns': tuple(column.to_dict() for column in self.columns),
        }

    @classmethod
    def from_dict(cls, dct: Dict[str, Union[str, Iterable[Dict[str, str]]]]) -> Self:
        return cls(
            columns=tuple(ColumnSpec.from_dict(spec) for spec in dct.get('columns', ())),
        )


TASK_TYPES = dict()
TASK_TYPES['cued-lever-pull'] = TrialSpec.from_dict(CUED_LEVER_PULL)
TASK_TYPES['sensory-stim'] = TrialSpec.from_dict(SENSORY_STIM)


# def index_to_timestamp(
#     vals: _npt.NDArray[_np.integer],
#     time: _npt.NDArray[_np.floating]
# ) -> _npt.NDArray[_np.float32]:
#     out = _np.empty((vals.size,), dtype=_np.float32)
#     for i, v in enumerate(vals):
#         if v >= 0:
#             out[i] = time[v]
#         else:
#             out[i] = _np.nan
#     return out


# def load_trials(
#     rawfile: PathLike,
#     timebases: _core.Timebases,
#     tasktype: str = 'cued_lever_pull',
#     sampling_rate: float = 5000.0,
#     verbose: bool = True,
# ) -> _pd.DataFrame:
#     task = getattr(_trials, tasktype)
#     _stdio.message(f"loaded task type: '{tasktype}'", verbose=verbose)
#     _stdio.message("loading trial data...", end=' ', verbose=verbose)
#     start = _now()
#     daq = _trials.load_raw_daq(rawfile)
#     stop = _now()
#     _stdio.message(f"done (took {(stop - start):.1f} sec).", verbose=verbose)
#     trials = task.extract_trials(daq, rate=sampling_rate)
#     _stdio.message("done parsing trials.", verbose=verbose)
#     trials['start_time'] = timebases.raw[trials.start.values]
#     trials['stop_time']  = timebases.raw[trials.end.values]
#     for col, typ in task.COLUMN_TYPES.items():
#         if col in ('start', 'end'):
#             continue
#         elif typ == 'time':
#             trials[f"{col}_time"] = index_to_timestamp(
#                 trials[col].values,
#                 timebases.raw,
#             )
#     return trials


# def load_trials(
#     rawfile: PathLike,
# ) -> _pd.DataFrame:
#     raise NotImplementedError

def load_trials(
    rawfile: PathLike,
) -> Optional[_pd.DataFrame]:
    with _h5.File(rawfile, 'r') as src:
        # Loading from hdf5 file
        data   = _np.array(src["behavior_raw/trial_info/data"], dtype=_np.float32).T
        labels = _np.array(src["behavior_raw/trial_info/label"]).ravel()
        labels = [lab.decode('utf-8') for lab in labels]  # convert to utf-8

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
        data        = _np.array(src["behavior_ds/trial_info/data"], dtype=_np.float32).T
        label       = _np.array(src["behavior_ds/trial_info/label"]).ravel()
        label       = [lab.decode('utf-8') for lab in label]  # convert to utf-8
        # convert to dataframe
        trials_ds = _pd.DataFrame(data, columns=label)
        return trials_ds


# def format_trials(
#     trials: _pd.DataFrame,
#     tasktype: str = 'cued_lever_pull',
# ) -> _pd.DataFrame:
#     task = getattr(_trials, tasktype)
#     data = dict()
#     for name, typ in task.COLUMN_TYPES.items():
#         if name == 'start':
#             data['start_time'] = trials['start_time'].values
#         elif name == 'end':
#             data['stop_time'] = trials['stop_time'].values
#         elif typ == 'time':
#             data[name] = trials[f"{name}_time"].values
#         else:
#             data[name] = trials[name].values
#     return _pd.DataFrame(data=data)


def write_trials(
    parent: Union[_nwb.NWBFile, _nwb.base.ProcessingModule],
    trials: _pd.DataFrame,
    tasktype: str = 'cued_lever_pull',
    verbose: bool = True,
):
    trial_spec: TrialSpec = TASK_TYPES.get(tasktype, None)
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
        table = _TimeIntervals(
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

