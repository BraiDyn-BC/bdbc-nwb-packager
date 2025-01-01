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

from importlib import reload as _reload  # DEBUG

from . import (
    logging,
    types,
    configure,
    file_metadata,
    timebases,
    daq,
    trials,
    videos,
    imaging,
    rois,
    tracking,
    packaging,
    batch,
)

_reload(logging)  # DEBUG
_reload(types)  # DEBUG
_reload(configure)  # DEBUG
_reload(file_metadata)  # DEBUG
_reload(timebases)  # DEBUG
_reload(daq)  # DEBUG
_reload(trials)  # DEBUG
_reload(videos)  # DEBUG
_reload(imaging)  # DEBUG
_reload(rois)  # DEBUG
_reload(tracking)  # DEBUG
_reload(packaging)  # DEBUG
_reload(batch)  # DEBUG

# classes
PathSettings = configure.PathSettings

Metadata = file_metadata.Metadata
ROISetMetadata = file_metadata.ROISetMetadata

Timebases = timebases.Timebases
PulseTriggers = timebases.PulseTriggers

# procedures
setup_path_settings = configure.setup_path_settings

read_recordings_metadata = file_metadata.read_recordings_metadata
read_roi_metadata = file_metadata.read_roi_metadata

read_timebases = timebases.read_timebases

iterate_raw_daq_recordings = daq.iterate_raw_daq_recordings
iterate_downsampled_daq_recordings = daq.iterate_downsampled_daq_recordings

process = packaging.process

run_batch = batch.run_batch
