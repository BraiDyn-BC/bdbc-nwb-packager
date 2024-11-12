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
"""metadata structures related to session description (apart from recordings)."""

from typing import Optional, ClassVar
from typing_extensions import Self
from datetime import datetime
from dataclasses import dataclass

from .common import (
    JSONLike,
    MetadataParseError,
)


@dataclass
class SessionMetadata:
    session_type: str
    description: str
    experimenter: str
    start_time: Optional[datetime]
    lab: str
    institution: str
    notes: str

    @classmethod
    def from_dict(
        cls,
        dct: JSONLike,
    ) -> Self:
        sesstype = dct.get('session_type')
        desc = dct.get('session_description', None)
        notes = dct.get('session_notes', None)
        try:
            start = datetime.strptime(
                dct['session_start_time'], "%Y/%m/%d %H:%M:%S"
            ).astimezone(None)  # assumes local timezone
        except ValueError:
            start = None
        if start is None:
            print(dct)
            raise MetadataParseError(
                f"failed to parse session start time: '{dct['session_start_time']}'"
            )
        exper = dct.get('experimenter')
        lab = dct.get('lab')
        inst = dct.get('institution')
        return cls(
            session_type=sesstype,
            description=desc,
            experimenter=exper,
            start_time=start,
            lab=lab,
            institution=inst,
            notes=notes,
        )

    def __post_init__(self):
        if self.session_type is None:
            raise MetadataParseError('unspecified metadata: session type')
        if self.description is None:
            raise MetadataParseError('unspecified metadata: task description')
        if self.experimenter is None:
            raise MetadataParseError('unspecified metadata: experimenter')
        if self.lab is None:
            raise MetadataParseError('unspecified metadata: laboratory')
        if self.institution is None:
            raise MetadataParseError('unspecified metadata: institution')
        if self.notes is None:
            raise MetadataParseError('unspecified metadata: session notes')


@dataclass
class SubjectMetadata:
    ID: str
    species: str
    strain: str
    genotype: str
    sex: str
    date_of_birth: Optional[datetime]
    age: str  # TODO
    baseweight: float
    weight: float
    description: str
    AGE_REFERENCE: ClassVar[str] = 'birth'

    @property
    def age_reference(self) -> str:
        return self.AGE_REFERENCE

    @classmethod
    def from_dict(cls, dct: JSONLike) -> Self:
        ID = dct['subject_id']
        species = dct['species']
        strain = dct['strain']
        genotype = dct['genotype']
        sex = dct['sex']
        try:
            DoB = datetime.strptime(
                dct['date_of_birth'], "%Y-%m-%d"
            ).astimezone(None)  # assumes local timezone
        except ValueError:
            DoB = None
        if DoB is None:
            print(dct)
            raise MetadataParseError(
                f"failed to parse subject DoB: '{dct['date_of_birth']}'"
            )
        age = dct['age']
        if not isinstance(age, str):
            raise ValueError(f'age is {type(age).__name__}, not str')
        baseweight = float(dct['base_weight']) / 1000  # g --> kg
        weight = float(dct['weight']) / 1000  # g --> kg
        desc = dct.get('subject_description', '')
        return cls(
            ID=ID,
            species=species,
            strain=strain,
            genotype=genotype,
            sex=sex,
            date_of_birth=DoB,
            age=age,
            baseweight=baseweight,
            weight=weight,
            description=desc,
        )
