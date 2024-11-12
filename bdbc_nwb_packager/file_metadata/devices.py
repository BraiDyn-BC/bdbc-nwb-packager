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
"""device-related metadata structures."""

from typing import ClassVar
from typing_extensions import Self
from dataclasses import dataclass

from .common import JSONLike


@dataclass
class DeviceMetadata:
    name: str
    manufacturer: str
    description: str


@dataclass
class BehaviorVideosMetadata(DeviceMetadata):
    RATE: ClassVar[float] = 120.0

    @property
    def rate(self) -> float:
        return self.RATE

    @classmethod
    def from_dict(cls, dct: JSONLike) -> Self:
        return cls(
            name=dct['video_device'],
            manufacturer=dct['video_device_manufacturer'],
            description=dct['video_device_description'],
        )
