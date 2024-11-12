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
"""the high-level entry points for the other modules, when dealing with NWB metadata."""
from typing import Optional
from dataclasses import dataclass

import numpy as _np
import numpy.typing as _npt
import h5py as _h5

import bdbc_session_explorer as _sessx

from ..stdio import (
    message as _message,
)
from .common import (
    JSONLike,
    PathLike,
    read_metadata_as_dict as _read_metadata_as_dict,
)
from .devices import (
    BehaviorVideosMetadata as _BehaviorVideosMetadata,
)
from .session import (
    SessionMetadata as _SessionMetadata,
    SubjectMetadata as _SubjectMetadata,
)
from .recording import (
    TaskRecordingMetadata as _TaskRecordingMetadata,
    ImagingMetadata as _ImagingMetadata,
)
from .rois import (
    SingleROIMetadata as _SingleROIMetadata,
    ROISetMetadata as _ROISetMetadata,
)


@dataclass
class Metadata:
    basedict: JSONLike
    session: _SessionMetadata
    subject: _SubjectMetadata
    task: _TaskRecordingMetadata
    imaging: _ImagingMetadata
    videos: _BehaviorVideosMetadata

    @property
    def session_name(self) -> str:
        return f"{self.subject.ID}_{self.session.start_time.strftime('%Y-%m-%d')}_{self.session.session_type}"


def read_recordings_metadata(
    session: _sessx.Session,
    rawfile: PathLike,
    override: Optional[JSONLike] = None,
) -> Metadata:
    basedict = _read_metadata_as_dict(rawfile)
    # overwrite session metadata using `session`
    if session is not None:
        basedict['session_notes'] = session.comments
        basedict['session_type']  = session.type
    if override is not None:
        basedict.update(override)
    return Metadata(
        basedict=basedict,
        session=_SessionMetadata.from_dict(basedict),
        subject=_SubjectMetadata.from_dict(basedict),
        task=_TaskRecordingMetadata.from_dict(basedict),
        imaging=_ImagingMetadata.from_dict(basedict),
        videos=_BehaviorVideosMetadata.from_dict(basedict),
    )


def read_roi_metadata(
    mesofile: PathLike,
    verbose: bool = True,
) -> _ROISetMetadata:
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
                roi = _SingleROIMetadata(
                    name=f"{name}_{side[0]}",
                    mask=_as_mask(group[name]),
                    description=f"{group[name].attrs['description']}, {side} hemisphere",
                )
                rois.append(roi)
        _message(f"read {len(rois)} ROIs from the mesoscaler registration file.", verbose=verbose)
        return _ROISetMetadata(
            transform=trans,
            rois=tuple(rois),
        )
