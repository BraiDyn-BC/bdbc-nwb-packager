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
from typing import Optional, Union, Iterable, Iterator, ClassVar
from typing_extensions import Self
from dataclasses import dataclass

import pandas as _pd

from . import configs as _configs


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

    def get_value_from(self, row: dict[str, FieldType]) -> FieldType:
        return self.data_type(row[self.input_name])

    def to_dict(self) -> dict[str, str]:
        return {
            'input_name': self.input_name,
            'output_name': self.output_name,
            'data_type': self.data_type.__name__,
            'description': self.description,
        }

    @classmethod
    def from_dict(cls, dct: dict[str, str]) -> Self:
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
    def required_columns(self) -> Iterator[ColumnSpec]:
        for column in self.columns:
            if column.name in self.REQUIRED_COLUMNS:
                yield column

    @property
    def task_specific_columns(self) -> Iterator[ColumnSpec]:
        for column in self.columns:
            if column.name in self.REQUIRED_COLUMNS:
                continue
            yield column

    def iter_trials_from(
        self,
        trials: _pd.DataFrame
    ) -> Iterator[dict[str, FieldType]]:
        for _, row in trials.iterrows():
            row = row.to_dict()
            yield dict((col.name, col.get_value_from(row)) for col in self.columns)

    def to_dict(self) -> dict[str, Union[str, Iterable[dict[str, str]]]]:
        return {
            'columns': tuple(column.to_dict() for column in self.columns),
        }

    @classmethod
    def from_dict(cls, dct: dict[str, Union[str, Iterable[dict[str, str]]]]) -> Self:
        return cls(
            columns=tuple(ColumnSpec.from_dict(spec) for spec in dct.get('columns', ())),
        )


TASK_TYPES = dict()
TASK_TYPES['cued-lever-pull'] = TrialSpec.from_dict(_configs.CUED_LEVER_PULL)
TASK_TYPES['sensory-stim'] = TrialSpec.from_dict(_configs.SENSORY_STIM)


def get_spec(tasktype: str) -> Optional[TrialSpec]:
    return TASK_TYPES.get(tasktype, None)
