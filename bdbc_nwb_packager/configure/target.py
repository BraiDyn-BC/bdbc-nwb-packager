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
"""the 'destination' part of the path configuration (definitely simpler than the 'source' side)"""

from typing import Optional
from typing_extensions import Self
from pathlib import Path
from dataclasses import dataclass

import bdbc_session_explorer as _sessx


# TODO: what if the channel configs changed? (maybe the use of dict's would be better?)
@dataclass
class ImagingDataFiles:
    format: str
    B: Path
    V: Path

    def relative_to(self, ref: Path) -> Self:
        return self.__class__(
            format=self.format,
            B=self.B.relative_to(ref),
            V=self.V.relative_to(ref),
        )


# TODO: what if the views changed? (maybe the use of dict's would be better?)
@dataclass
class BehaviorVideoFiles:
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
class DefaultPackaging:
    root: Path
    session: _sessx.Session
    save_imaging_in_avi: bool = False

    def __post_init__(self):
        self.root = _sessx.publication_root_dir(self.root)

    @property
    def animal(self) -> str:
        return self.session.escaped_animal

    @property
    def session_name(self) -> str:
        return f"{self.animal}_{self.session.longdate}_{self.session.longtype}-{self.session.longday}"

    @property
    def session_dir(self) -> Path:
        return self.root / self.animal / self.session_name

    @property
    def imaging_dir(self) -> Path:
        return self.session_dir / 'imaging'

    @property
    def video_dir(self) -> Path:
        return self.session_dir / 'videos'

    @property
    def nwbfile(self) -> Path:
        return self.session_dir / f"{self.session_name}.nwb"

    @property
    def imaging_data_files(self) -> ImagingDataFiles:
        imgdir = self.imaging_dir
        sessname = self.session_name
        fmt = 'avi' if self.save_imaging_in_avi == True else 'tiff'
        return ImagingDataFiles(
            format=fmt,
            B=(imgdir / f"{sessname}_B.{fmt}"),
            V=(imgdir / f"{sessname}_V.{fmt}"),
        )

    @property
    def behavior_video_files(self) -> BehaviorVideoFiles:
        videodir = self.video_dir
        sessname = self.session_name
        return BehaviorVideoFiles(
            body=(videodir / f"{sessname}_body.mp4"),
            face=(videodir / f"{sessname}_face.mp4"),
            eye=(videodir / f"{sessname}_eye.mp4"),
        )


@dataclass
class DANDIPackaging:
    root: Path
    session: _sessx.Session
    save_imaging_in_avi: bool = False

    def __post_init__(self):
        self.root = _sessx.publication_root_dir(self.root)

    @property
    def subject_name(self) -> str:
        return f"sub-{self.session.animal}"

    @property
    def subject_dir(self) -> Path:
        return self.root / self.subject_name

    @property
    def modalities(self) -> list[str]:
        mods = ['widefield']
        if self.session.has_any_videos():
            mods.append('behavior')
        return mods

    @property
    def session_base(self) -> str:
        return f"{self.subject_name}_ses-{self.session.longdate}-{self.session.longtype}-{self.session.longday}"

    @property
    def session_name(self) -> str:
        mods = '+'.join(self.modalities)
        return f"{self.session_base}_{mods}"

    @property
    def resource_dir(self) -> Path:
        return self.subject_dir / self.session_name

    @property
    def nwbfile(self) -> Path:
        return self.subject_dir / f"{self.session_name}.nwb"

    @property
    def imaging_data_files(self) -> ImagingDataFiles:
        imgdir = self.resource_dir
        basename = self.session_base
        fmt = 'avi' if self.save_imaging_in_avi == True else 'tiff'
        return ImagingDataFiles(
            format=fmt,
            B=(imgdir / f"{basename}_widefield-B.{fmt}"),
            V=(imgdir / f"{basename}_widefield-V.{fmt}"),
        )

    @property
    def behavior_video_files(self) -> BehaviorVideoFiles:
        videodir = self.resource_dir
        basename = self.session_base
        return BehaviorVideoFiles(
            body=(videodir / f"{basename}_body-camera.mp4"),
            face=(videodir / f"{basename}_face-camera.mp4"),
            eye=(videodir / f"{basename}_eye-camera.mp4"),
        )


@dataclass
class DestinationPaths:
    nwb: Path
    imaging: ImagingDataFiles
    videos: BehaviorVideoFiles

    @property
    def session_dir(self) -> Path:
        return self.nwb.parent


def setup_destination_paths(
    session: _sessx.Session,
    nwbroot: Optional[Path] = None,
    packaging_format: str = 'default',
    save_imaging_in_avi: bool = False,
) -> DestinationPaths:
    if packaging_format == 'default':
        packaging = DefaultPackaging(
            root=nwbroot,
            session=session,
            save_imaging_in_avi=save_imaging_in_avi
        )
    elif packaging_format == 'dandi':
        packaging = DANDIPackaging(
            root=nwbroot,
            session=session,
            save_imaging_in_avi=save_imaging_in_avi,
        )
    else:
        raise ValueError(f"expected one of ('default', 'dandi'), got '{packaging_format}'")
    return DestinationPaths(
        nwb=packaging.nwbfile,
        imaging=packaging.imaging_data_files,
        videos=packaging.behavior_video_files,
    )
