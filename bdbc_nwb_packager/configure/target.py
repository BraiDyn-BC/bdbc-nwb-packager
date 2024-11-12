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
"""the 'destination' part of the path configuration (definitely simpler than the 'source' side)"""

from typing import Optional
from typing_extensions import Self
from pathlib import Path
from dataclasses import dataclass

import bdbc_session_explorer as _sessx


# TODO: what if the channel configs changed? (maybe the use of dict's would be better?)
@dataclass
class ImagingDataFiles:
    B: Path
    V: Path

    def relative_to(self, ref: Path) -> Self:
        return self.__class__(
            B=self.B.relative_to(ref),
            V=self.V.relative_to(ref),
        )


# TODO: what if the views changed? (maybe the use of dict's would be better?)
@dataclass
class DestinationVideoFiles:
    body: Path
    face: Path
    eye: Path

    def relative_to(self, ref: Path) -> Self:
        return self.__class__(
            body=self.body.relative_to(ref),
            face=self.face.relative_to(ref),
            eye=self.eye.relative_to(ref),
        )


@dataclass
class DestinationPaths:
    nwb: Path
    imaging: ImagingDataFiles
    videos: DestinationVideoFiles

    @property
    def session_dir(self) -> Path:
        return self.nwb.parent


def setup_destination_paths(
    session: _sessx.Session,
    nwbroot: Optional[Path] = None,
) -> DestinationPaths:
    nwbroot = _sessx.publication_root_dir(nwbroot)
    animal   = session.escaped_animal
    sessname = f"{animal}_{session.longdate}_{session.longtype}-{session.longday}"
    sessdir  = nwbroot / animal / sessname
    nwbfile  = sessdir / f"{sessname}.nwb"
    imgdir   = sessdir / 'imaging'
    videodir = sessdir / 'videos'

    images   = ImagingDataFiles(
        B=(imgdir / f"{sessname}_B.tiff"),
        V=(imgdir / f"{sessname}_V.tiff"),
    )
    videos   = DestinationVideoFiles(
        body=(videodir / f"{sessname}_body.mp4"),
        face=(videodir / f"{sessname}_face.mp4"),
        eye=(videodir / f"{sessname}_eye.mp4"),
    )
    return DestinationPaths(
        nwb=nwbfile,
        imaging=images,
        videos=videos,
    )
