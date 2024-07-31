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

from importlib import reload as _reload

from . import (
    core,
    trials,
    daq,
    videos,
    imaging,
    rois,
    dlc,
    pupil,
    procs,
)

_reload(core)
_reload(trials)
_reload(daq)
_reload(videos)
_reload(imaging)
_reload(rois)
_reload(dlc)
_reload(pupil)
_reload(procs)

Timebases = core.Timebases
ImagingData = imaging.ImagingData
SingleROISignal = rois.SingleROISignal

load_timebases = core.load_timebases

load_trials = trials.load_trials
write_trials = trials.write_trials

iterate_raw_daq_recordings = daq.iterate_raw_daq_recordings
iterate_downsampled_daq_recordings = daq.iterate_downsampled_daq_recordings

write_videos = videos.write_videos

load_imaging_data = imaging.load_imaging_data
setup_imaging_device = imaging.setup_imaging_device
write_imaging_data = imaging.write_imaging_data

compute_roi_signals = rois.compute_roi_signals
setup_roi_segmentation = rois.setup_roi_segmentation
setup_roisignals_entry = rois.setup_roisignals_entry
write_roi_entries = rois.write_roi_entries

iterate_pose_estimations = dlc.iterate_pose_estimations
load_pupil_fitting = pupil.load_pupil_fitting

configure_nwb_file = procs.configure_nwb_file
package_nwb = procs.package_nwb
