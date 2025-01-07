# MIT License
#
# Copyright (c) 2024-2025 Keisuke Sehara and Ryo Aoki
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
from typing import Iterable
from typing_extensions import Self

import pandas as _pd

from .spec import (
    TrialSpec,
    Trials,
)


PUNCTUATION = '.,;:'
FlagItem = dict[str, int | str]
Mappable = int | float | _pd.DataFrame | TrialSpec


class FlagMapper:
    name: str
    mapdict: dict[int, str]
    description: str

    def __init__(self, name: str, mapdict: dict[int, str], description: str):
        self.name = name
        self.mapdict = mapdict
        self.description = description

    def map(self, value: Mappable) -> Mappable:
        if isinstance(value, _pd.DataFrame):
            value = value.copy()
            value[self.name] = value[self.name].map(self.map)
            return value
        elif value.__class__.__name__ == 'TrialSpec':
            # the expression above is a dirty hack so as not for Python
            # to understand the 'TrialSpec' class
            # (if one uses `isinstance(value, TrialSpec)` instead, there
            #  seems to be a chance of making distinction between
            #  one TrialSpec class and another)
            value = value.deepcopy()
            columnidx = value.column_index(self.name)
            column = value.columns[columnidx]
            column.data_type = str
            column.description = f"{column.description}: {self.description}"
            return value
        try:
            return self.mapdict[int(value)]
        finally:
            pass

    @classmethod
    def from_cfg(cls, name: str, items: Iterable[FlagItem]) -> Self:
        def _as_desc(item):
            return item['description'].replace(';', ' --').strip().rstrip(PUNCTUATION)
        mapdict = dict((item['value'], item['name']) for item in items)
        description = "; ".join(f"`{item['name']} ({item['value']})`, {_as_desc(item)}" for item in items)
        return cls(name=name, mapdict=mapdict, description=description)

    @classmethod
    def from_flagdict(cls, flagdict: dict[str, Iterable[FlagItem]]) -> tuple[Self]:
        return tuple(cls.from_cfg(name, items) for name, items in flagdict.items())


def map_flags_to_categories(trials: Trials) -> Trials:
    mappers = FlagMapper.from_flagdict(trials.flags)
    table = trials.table
    metadata = trials.metadata
    for mapper in mappers:
        table = mapper.map(table)
        metadata = mapper.map(metadata)
    return Trials(table=table, flags={}, metadata=metadata)
