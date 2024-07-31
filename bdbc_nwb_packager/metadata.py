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

from typing import Any, Dict, Tuple, Union
from typing_extensions import Self
from pathlib import Path
from collections import namedtuple as _namedtuple
from datetime import datetime as _datetime
import sys as _sys

import numpy as _np
import numpy.typing as _npt
import h5py as _h5
import pandas as _pd

PathLike = Union[str, Path]


def _message(msg: str, end: str = '\n', verbose: bool = True):
    if verbose:
        print(msg, end=end, file=_sys.stderr, flush=True)


class Metadata(_namedtuple('Metadata', ('basedict', 'session', 'subject', 'task', 'imaging', 'videos'))):
    @property
    def session_name(self) -> str:
        return f"{self.subject.ID}_{self.session.start_time.strftime('%Y-%m-%d')}_{self.session.session_type}"


class SessionMetadata(_namedtuple('SessionMetadata', (
    'session_type',
    'description',
    'experimenter',
    'start_time',
    'lab',
    'institution',
))):
    SESSION_TYPES = {
        'task': 'task',
        'rest': 'resting-state',
        'ss': 'sensory-stim',
    }
    
    @classmethod
    def from_dict(
        cls,
        dct: Dict[str, Any],
        session_type: str = 'task',
    ) -> Self:
        desc = dct['session_description']
        start = _datetime.strptime(
            dct['session_start_time'], "%Y/%m/%d %H:%M:%S"
        ).astimezone(None)  # assumes local timezone
        exper = dct['experimenter']
        lab = dct['lab']
        inst = dct['institution']
        if session_type in cls.SESSION_TYPES.keys():
            session_type = cls.SESSION_TYPES[session_type]
        return cls(
            session_type=session_type,
            description=desc,
            experimenter=exper,
            start_time=start,
            lab=lab,
            institution=inst,
        )


class SubjectMetadata(_namedtuple('SubjectMetadata', (
    'ID',
    'species',
    'strain',
    'genotype',
    'sex',
    'date_of_birth',
    'age',
    'baseweight',
    'weight',
    'description',
))):
    AGE_REFERENCE = 'birth'

    @property
    def age_reference(self) -> str:
        return self.AGE_REFERENCE
    
    @classmethod
    def from_dict(cls, dct: Dict[str, Any]) -> Self:
        ID = dct['subject_id']
        species = dct['species']
        strain = dct['strain']
        genotype = dct['genotype']
        sex = dct['sex']
        DoB = _datetime.strptime(
            dct['date_of_birth'], "%Y-%m-%d"
        ).astimezone(None)  # assumes local timezone
        age = dct['age']
        baseweight = dct['baseweight']
        weight = dct['weight']
        desc = dct['description']
        return cls(
            ID=ID,
            species=species,
            strain=strain,
            genotype=genotype,
            sex=sex,
            date_of_birth=DoB,
            age=age,
            baseweight=baseweight / 1000,  # to kg
            weight=weight / 1000,  # to kg
            description=desc,
        )


class TaskRecordingMetadata(_namedtuple('TaskRecordingMetadata', (
    'device',
    'raw_labels',
    'downsampled_labels',
))):
    RATE = 5000.0

    @property
    def rate(self) -> float:
        return self.RATE
    
    @classmethod
    def from_dict(cls, dct: Dict[str, Any]) -> Self:
        dev = DeviceMetadata(
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


class ImagingPlaneMetadata(_namedtuple('ImagingPlaneMetadata', (
    'name',
    'excitation',
    'emission',
    'pixel_size',
    'frame_rate',
    'description',
))):
    pass


class ImagingMetadata(_namedtuple('ImagingMetadata', (
    'location',
    'indicator',
    'device',
    'planes',
))):
    @classmethod
    def from_dict(cls, dct: Dict[str, Any]) -> Self:
        loc = dct['location']
        ind = dct['indicator']
        dev = DeviceMetadata(
            name=dct['img_device'],
            manufacturer=dct['img_device_manufacturer'],
            description=dct['img_device_description'],
        )
        pix = dct['imaging_pixel_size']
        rate = dct['imaging_frame_rate']
        chans = (
            ImagingPlaneMetadata(
                'B',
                excitation=float(dct['exc_wavelength'][1]),
                emission=float(dct['emi_wavelength']),
                pixel_size=pix,
                frame_rate=rate,
                description=dct['imaging_plane_description_b'],
            ),
            ImagingPlaneMetadata(
                'V',
                excitation=float(dct['exc_wavelength'][0]),
                emission=float(dct['emi_wavelength']),
                pixel_size=pix,
                frame_rate=rate,
                description=dct['imaging_plane_description_v'],
            ),
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


class BehaviorVideosMetadata(_namedtuple('BehaviorVideosMetadata', (
    'device',
))):
    RATE = 120.0

    @property
    def rate(self) -> float:
        return self.RATE
    
    @classmethod
    def from_dict(cls, dct: Dict[str, Any]) -> Self:
        dev = DeviceMetadata(
            name=dct['video_device'],
            manufacturer=dct['video_device_manufacturer'],
            description=dct['video_device_description'],
        )
        return cls(device=dev)


class DeviceMetadata(_namedtuple('DeviceMetadata', (
    'name',
    'manufacturer',
    'description',
))):
    pass


class ROISetMetadata(_namedtuple('ROISetMetadata', (
    'transform',
    'rois'
))):
    def transform_as_table(self) -> _pd.DataFrame:
        tab = _pd.DataFrame(data={
            'x_in': self.transform[:, 0],
            'y_in': self.transform[:, 1],
            'c_in': self.transform[:, 2],
        })
        tab.index = ['x_out', 'y_out']
        return tab


class SingleROIMetadata(_namedtuple('SingleROIMetadata', (
    'name',
    'mask',
    'description'
))):
    def __repr__(self) -> str:
        return f"SingleROIMetadata(name='{self.name}', ...)"


def metadata_from_rawdata(rawfile: PathLike, session_type: str = 'task') -> Metadata:
    basedict = read_metadata_as_dict(rawfile)
    return Metadata(
        basedict=basedict,
        session=SessionMetadata.from_dict(basedict, session_type=session_type),
        subject=SubjectMetadata.from_dict(basedict),
        task=TaskRecordingMetadata.from_dict(basedict),
        imaging=ImagingMetadata.from_dict(basedict),
        videos=BehaviorVideosMetadata.from_dict(basedict),
    )


def read_metadata_as_dict(h5file: PathLike) -> Dict[str, Any]:
    def pythonify_(entry: _h5.Dataset) -> Any:
        content = _np.array(entry).ravel()
        if _np.issubdtype(content.dtype, _np.integer):
            if content.size == 1:
                return int(content[0])
            else:
                return tuple(int(v) for v in content)
        elif _np.issubdtype(content.dtype, _np.floating):
            if content.size == 1:
                return float(content[0])
            else:
                return tuple(float(v) for v in content)
        else:
            # assumes byte string
            return content[0].decode('utf-8')
    
    def swap_key_(dct, key_old: str, key_new: str):
        if (key_new not in dct.keys()) and (key_old in dct.keys()):
            dct[key_new] = dct.pop(key_old)
    
    def fill_(dct, key: str, default: Any):
        if key not in dct.keys():
            dct[key] = default

    def as_string_items_(entry: _h5.Dataset) -> Tuple[str]:
        content = _np.array(entry).ravel()
        return tuple(item.decode('utf-8') for item in content)
    
    with _h5.File(h5file, 'r') as src:
        group = src['metadata']
        metadata = dict((key.lower(), pythonify_(group[key])) for key in group.keys())
        metadata['bhv_raw_labels'] = as_string_items_(src['behavior_raw/label'])
        metadata['bhv_ds_labels']  = as_string_items_(src['behavior_ds/label'])
    
    swap_key_(metadata, 'spiecies', 'species')
    swap_key_(metadata, 'mouseid', 'subject_id')
    swap_key_(metadata, 'dob', 'date_of_birth')
    swap_key_(metadata, 'img_devicemanufacturer', 'img_device_manufacturer')
    swap_key_(metadata, 'img_devicedescription', 'img_device_description')
    swap_key_(metadata, 'imagingpixelsize', 'imaging_pixel_size')
    swap_key_(metadata, 'imagingframerate', 'imaging_frame_rate')
    swap_key_(metadata, 'imagingplane_descriptionb', 'imaging_plane_description_b')
    swap_key_(metadata, 'imagingplane_descriptionv', 'imaging_plane_description_v')
    swap_key_(metadata, 'bhv_devicemanufacturer', 'bhv_device_manufacturer')
    swap_key_(metadata, 'bhv_devicedescription', 'bhv_device_description')
    swap_key_(metadata, 'video_devicemanufacturer', 'video_device_manufacturer')
    swap_key_(metadata, 'video_devicedescription', 'video_device_description')
    fill_(metadata, 'description', '')
    return metadata


def read_roi_metadata(
    mesofile: PathLike,
    verbose: bool = True,
) -> ROISetMetadata:
    def _as_mask(entry: _h5.Dataset) -> _npt.NDArray[bool]:
        mask = _np.array(entry, dtype=_np.uint8)
        mask = mask / mask.max()
        return mask > 0.5
    
    with _h5.File(mesofile, 'r') as src:
        trans = _np.array(src['transform/atlas_to_data'], dtype=_np.float32)
        rois = []
        for side in ('left', 'right'):
            group = src['rois'][side]
            for name in group.keys():
                if name == 'outline':
                    continue
                roi = SingleROIMetadata(
                    name=f"{name}_{side[0]}",
                    mask=_as_mask(group[name]),
                    description=f"{group[name].attrs['description']}, {side} hemisphere",
                )
                rois.append(roi)
        _message(f"read {len(rois)} ROIs from the mesoscaler registration file.", verbose=verbose)
        return ROISetMetadata(
            transform=trans,
            rois=tuple(rois),
        )
