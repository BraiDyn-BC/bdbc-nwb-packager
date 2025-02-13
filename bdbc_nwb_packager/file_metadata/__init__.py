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
"""(data-)classes and procedures related to the handling of NWB metadata"""

from importlib import reload as _reload  # DEBUG

from . import (
    common,
    devices,
    session,
    recording,
    rois,
    highlevel,
)

_reload(common)  # DEBUG
_reload(devices)  # DEBUG
_reload(session)  # DEBUG
_reload(recording)  # DEBUG
_reload(rois)  # DEBUG
_reload(highlevel)  # DEBUG

# error types
MetadataParseError = common.MetadataParseError

# exported classes (by submodule)
DeviceMetadata = devices.DeviceMetadata
BehaviorVideosMetadata = devices.BehaviorVideosMetadata

SessionMetadata = session.SessionMetadata
SubjectMetadata = session.SubjectMetadata

TaskRecordingMetadata = recording.TaskRecordingMetadata
ImagingPlaneMetadata = recording.ImagingPlaneMetadata
ImagingMetaData = recording.ImagingMetadata

SingleROIMetadata = rois.SingleROIMetadata
ROISetMetadata = rois.ROISetMetadata

Metadata = highlevel.Metadata

# exported procedures
read_metadata_as_dict = common.read_metadata_as_dict
read_recordings_metadata = highlevel.read_recordings_metadata
read_roi_metadata = highlevel.read_roi_metadata
