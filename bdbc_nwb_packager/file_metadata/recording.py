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
"""set of metadata structures related to session recordings."""

from typing import ClassVar
from typing_extensions import Self
from dataclasses import dataclass

from .common import JSONLike
from .devices import (
    DeviceMetadata as _DeviceMetadata,
)


@dataclass
class TaskRecordingMetadata:
    device: _DeviceMetadata
    raw_labels: tuple[str]
    downsampled_labels: tuple[str]
    RATE: ClassVar[float] = 5000.0

    @property
    def rate(self) -> float:
        return self.RATE

    @classmethod
    def from_dict(cls, dct: JSONLike) -> Self:
        dev = _DeviceMetadata(
            name=dct['bhv_device'],
            manufacturer=dct['bhv_device_manufacturer'],
            description=dct['bhv_device_description'],
        )
        rlab = dct['bhv_raw_labels']
        dlab = dct['bhv_ds_labels']
        return cls(
            device=dev,
            raw_labels=rlab,
            downsampled_labels=dlab,
        )


@dataclass
class ImagingPlaneMetadata:
    name: str
    excitation: float
    emission: float
    pixel_size: tuple[float]
    frame_rate: float
    description: str


@dataclass
class ImagingMetadata:
    location: str
    indicator: str
    device: _DeviceMetadata
    planes: tuple[ImagingPlaneMetadata]

    @classmethod
    def from_dict(cls, dct: JSONLike) -> Self:
        loc = dct['location']
        ind = dct['indicator']
        dev = _DeviceMetadata(
            name=dct['img_device'],
            manufacturer=dct['img_device_manufacturer'],
            description=dct['img_device_description'],
        )
        pix = dct['imaging_pixel_size']
        rate = dct['imaging_frame_rate']
        channel_names = (
            dct['exc_order1'].upper(),
            dct['exc_order2'].upper(),
        )
        chans = tuple(
            ImagingPlaneMetadata(
                chan,
                excitation=float(dct['exc_wavelength'][i]),
                emission=float(dct['emi_wavelength'][i]),
                pixel_size=pix,
                frame_rate=rate,
                description=dct[f'imaging_plane_description{i + 1}'],
            ) for i, chan in enumerate(channel_names)
        )
        return cls(
            location=loc,
            indicator=ind,
            device=dev,
            planes=chans,
        )

    @property
    def B(self) -> ImagingPlaneMetadata:
        return [pln for pln in self.planes if pln.name == 'B'][0]

    @property
    def V(self) -> ImagingPlaneMetadata:
        return [pln for pln in self.planes if pln.name == 'V'][0]
