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
"""the 'source' part of the path configuration"""

from typing import Optional
from typing_extensions import Self
from pathlib import Path
from dataclasses import dataclass
import json as _json

import bdbc_session_explorer as _sessx

from ..types import (
    PathLike,
    PathsLike,
    maybe_path as _maybe_path,
)
from .. import (
    logging as _logging,
)


@dataclass
class SourceVideoFile:
    path: Optional[Path]
    width: int
    height: int
    num_frames: int

    @classmethod
    def empty(cls) -> Self:
        return cls(path=None, width=0, height=0, num_frames=0)

    @classmethod
    def from_path(cls, path: Optional[PathLike]) -> Self:
        if path is None:
            return cls.empty()
        metapath = path.with_name(f"METADATA_{path.stem}.json")
        with open(metapath, 'r') as src:
            videometa = _json.load(src)
        return cls(
            path=path,
            width=videometa['width'],
            height=videometa['height'],
            num_frames=videometa['num_frames'],
        )

    def __post_init__(self):
        self.path = _maybe_path(self.path)
        self.width = int(self.width)
        self.height = int(self.height)
        self.num_frames = int(self.num_frames)

    def relative_to(self, ref: Path) -> Self:
        return self.__class__(
            path=self.path.relative_to(ref),
            width=self.width,
            height=self.height,
            num_frames=self.num_frames,
        )


# TODO: what if the views changed? (maybe the use of dict's would be better?)
@dataclass
class SourceVideoFiles:
    body: SourceVideoFile
    face: SourceVideoFile
    eye: SourceVideoFile

    @classmethod
    def empty(cls) -> Self:
        return cls(
            body=SourceVideoFile.empty(),
            face=SourceVideoFile.empty(),
            eye=SourceVideoFile.empty()
        )

    @classmethod
    def from_video_files(
        cls,
        videofiles: _sessx.VideoFiles,
    ) -> Self:
        return cls(
            body=SourceVideoFile.from_path(videofiles.body),
            face=SourceVideoFile.from_path(videofiles.face),
            eye=SourceVideoFile.from_path(videofiles.eye),
        )

    @classmethod
    def from_paths(
        cls,
        body: Optional[PathLike] = None,
        face: Optional[PathLike] = None,
        eye: Optional[PathLike] = None,
    ) -> Self:
        return cls(
            body=SourceVideoFile.from_path(body),
            face=SourceVideoFile.from_path(face),
            eye=SourceVideoFile.from_path(eye),
        )

    def __post_init__(self):
        if not isinstance(self.body, SourceVideoFile):
            self.body = SourceVideoFile.from_path(self.body)
        if not isinstance(self.face, SourceVideoFile):
            self.face = SourceVideoFile.from_path(self.face)
        if not isinstance(self.eye, SourceVideoFile):
            self.eye = SourceVideoFile.from_path(self.eye)


# TODO: what if the views changed? (maybe the use of dict's would be better?)
@dataclass
class DLCResultFiles:
    body: Optional[Path]
    face: Optional[Path]
    eye: Optional[Path]

    @classmethod
    def empty(cls) -> Self:
        return cls(
            body=None,
            face=None,
            eye=None,
        )

    @classmethod
    def from_session_results(
        cls,
        results: _sessx.DLCOutputFiles
    ) -> Self:
        return cls(
            body=results.body.path,
            face=results.face.path,
            eye=results.eye.path,
        )

    def __post_init__(self):
        self.body = _maybe_path(self.body)
        self.face = _maybe_path(self.face)
        self.eye  = _maybe_path(self.eye)

    def has_any_results(self) -> bool:
        return any((getattr(self, view) is not None) for view in ('body', 'face', 'eye'))


@dataclass
class SourcePaths:
    rawdata: Path
    videos: SourceVideoFiles
    mesoscaler: Path
    deeplabcut: DLCResultFiles
    pupilfitting: Optional[Path]


def setup_source_paths(
    session: _sessx.Session,
    rawroot: Optional[PathsLike] = None,
    videoroot: Optional[PathLike] = None,
    mesoroot: Optional[PathLike] = None,
    body_results_root: Optional[PathLike] = None,
    face_results_root: Optional[PathLike] = None,
    eye_results_root: Optional[PathLike] = None,
    pupilroot: Optional[PathLike] = None,
) -> SourcePaths:
    rawfile  = _sessx.locate_rawdata_file(session, rawroot)
    mesofile = _sessx.locate_mesoscaler_file(session, mesoroot=mesoroot)
    if (mesofile is None) or (not mesofile.exists()):
        raise FileNotFoundError(str(mesofile))
    sessvideos = _sessx.video_files_from_session(
        session,
        videoroot=videoroot,
        error_handling='message',
    )
    videos = SourceVideoFiles.from_video_files(sessvideos)
    session_dlc_results = _sessx.dlc_output_files_from_session(
        session,
        body=body_results_root,
        face=face_results_root,
        eye=eye_results_root,
    )
    dlcfiles = DLCResultFiles.from_session_results(session_dlc_results)
    # TODO:
    #   run DeepLabCut in case any of the output is missing?
    if dlcfiles.eye is not None:
        pupfile  = _sessx.locate_pupil_file(session, pupilroot=pupilroot)
        if (pupfile is None) or (not pupfile.exists()):
            # TODO: probably run fit_pupil?
            _logging.warning(f"pupil file not found: {str(pupfile)}")
    else:
        pupfile = None
    return SourcePaths(
        rawdata=rawfile,
        videos=videos,
        mesoscaler=mesofile,
        deeplabcut=dlcfiles,
        pupilfitting=pupfile,
    )
