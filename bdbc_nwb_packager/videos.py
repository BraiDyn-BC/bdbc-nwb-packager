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
from typing import Optional
from dataclasses import dataclass
from time import time as _now
import shutil as _shutil

from pynwb import NWBFile as _NWBFile
from pynwb.image import ImageSeries as _ImageSeries

from . import (
    logging as _logging,
    configure as _configure,
    file_metadata as _file_metadata,
    timebases as _timebases,
)


VIEWS = {
    'body': 'view of the upper body from the bottom',
    'face': 'view of the face on the right side',
    'eye': 'view of the right eye',
}


@dataclass
class VideoEntries:
    body: Optional[_ImageSeries]
    face: Optional[_ImageSeries]
    eye: Optional[_ImageSeries]


def write_videos(
    nwbfile: _NWBFile,
    metadata: _file_metadata.Metadata,
    timebases: _timebases.Timebases,
    paths: _configure.PathSettings,
    copy_files: bool = True,
    verbose: bool = True,
) -> VideoEntries:
    relfiles = paths.destination.videos.relative_to(
        paths.destination.session_dir
    )
    t = timebases.videos
    device = nwbfile.create_device(
        name=metadata.videos.name,
        description=metadata.videos.description,
        manufacturer=metadata.videos.manufacturer,
    )

    entries = dict()
    for view in VIEWS.keys():
        srcpath = getattr(paths.source.videos, view).path
        dstpath = getattr(paths.destination.videos, view)  # note no need of 'path'
        if srcpath is None:
            _logging.warning(f"skipping {view} video: video file does not exist")
            entries[view] = None
            continue
        if not dstpath.parent.exists():
            dstpath.parent.mkdir(parents=True)

        if copy_files:
            _logging.info(f"copying {view} video...")
            start = _now()
            _shutil.copy(srcpath, dstpath)
            stop = _now()
            _logging.info(f"done copying video (took {(stop - start):.1f} sec).")

        desc = VIEWS[view]
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
        _logging.debug(f"registering the {view} video to the NWB file")
        nwbfile.add_acquisition(entry)
        entries[view] = entry

    _logging.info(f"done registering {len(entries)}/{len(VIEWS)} video files.")
    return VideoEntries(**entries)
