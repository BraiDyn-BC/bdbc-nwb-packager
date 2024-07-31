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

from collections import namedtuple as _namedtuple
from time import time as _now
import shutil as _shutil

from pynwb import NWBFile as _NWBFile
from pynwb.image import ImageSeries as _ImageSeries

from .. import (
    paths as _paths,
    metadata as _metadata,
)
from . import (
    core as _core,
)


DESCRIPTION = {
    'body': 'view of the upper body from the bottom',
    'face': 'view of the face on the right side',
    'eye': 'view of the right eye',
}


class VideoEntries(_namedtuple('VideoEntries', (
    'body',  # ImageSeries
    'face',  # ImageSeries
    'eye',   # ImageSeries
))):
    pass


def write_videos(
    nwbfile: _NWBFile,
    metadata: _metadata.Metadata,
    timebases: _core.Timebases,
    paths: _paths.PathSettings,
    verbose: bool = True,
) -> VideoEntries:
    
    relfiles = paths.destination.videos.relative_to(paths.destination.session_dir)
    t = timebases.videos
    device = nwbfile.create_device(
        name=metadata.videos.device.name,
        description=metadata.videos.device.description,
        manufacturer=metadata.videos.device.manufacturer,
    )
    
    entries = dict()
    for view in VideoEntries._fields:
        srcpath = getattr(paths.source.videos, view)
        dstpath = getattr(paths.destination.videos, view)
        if srcpath is None:
            _core.print_message(f"***skipping {view} video: video file does not exist", verbose=verbose)
            continue
        if not dstpath.parent.exists():
            dstpath.parent.mkdir(parents=True)
        
        _core.print_message(f"copying {view} video...", end=' ', verbose=verbose)
        start = _now()
        _shutil.copy(srcpath, dstpath)
        stop = _now()
        _core.print_message(f"done (took {(stop - start):.1f} sec).", verbose=verbose)

        desc = DESCRIPTION[view]
        entry = _ImageSeries(
            name=f"{view}_video",
            description=f"behavioral video acquisition, {desc}.",
            unit="n.a.",
            format="external", 
            external_file=[str(getattr(relfiles, view))],
            starting_frame=[0],
            timestamps=t,
            device=device,
        )
        nwbfile.add_acquisition(entry)
        entries[view] = entry

    _core.print_message("done registering the video files.", verbose=verbose)
    return VideoEntries(**entries)
