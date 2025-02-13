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
"""the session-level configuration logic (that gathers the 'source' and the 'destination' parts)"""

from typing import Optional, ClassVar, Iterator
from typing_extensions import Self
from pathlib import Path
from dataclasses import dataclass

import bdbc_session_explorer as _sessx

from ..types import (
    PathLike,
    PathsLike,
)
from .source import (
    SourcePaths,
    setup_source_paths as _setup_source_paths,
)
from .target import (
    DestinationPaths,
    setup_destination_paths as _setup_destination_paths,
)


# TODO: what if the views changed? (maybe the use of dict's would be better?)
@dataclass
class DLCModelConfigs:
    body: Path
    face: Path
    eye: Path
    FIELDS: ClassVar[tuple[str]] = ('body', 'face', 'eye')

    @classmethod
    def setup(
        cls,
        bodymodeldir: Optional[PathLike] = None,
        facemodeldir: Optional[PathLike] = None,
        eyemodeldir: Optional[PathLike] = None,
    ) -> Self:
        configs = _sessx.dlc_config_files(
            body=bodymodeldir,
            face=facemodeldir,
            eye=eyemodeldir,
        )
        return cls(**configs)

    def items(self) -> Iterator[tuple[str, Path]]:
        for fld in self.FIELDS:
            yield (fld, getattr(self, fld))


@dataclass
class PathSettings:
    session: _sessx.Session
    source: SourcePaths
    destination: DestinationPaths
    dlc_configs: DLCModelConfigs

    def has_behavior_videos(self) -> bool:
        return self.session.has_any_videos()


def setup_path_settings(
    session: _sessx.Session,
    rawroot: Optional[PathsLike] = None,
    videoroot: Optional[PathLike] = None,
    mesoroot: Optional[PathLike] = None,
    body_results_root: Optional[PathLike] = None,
    face_results_root: Optional[PathLike] = None,
    eye_results_root: Optional[PathLike] = None,
    pupilroot: Optional[PathLike] = None,
    nwbroot: Optional[Path] = None,
    bodymodeldir: Optional[PathLike] = None,
    facemodeldir: Optional[PathLike] = None,
    eyemodeldir: Optional[PathLike] = None,
) -> Optional[PathSettings]:
    if not session.has_rawdata():
        return None
    source  = _setup_source_paths(
        session,
        rawroot=rawroot,
        videoroot=videoroot,
        mesoroot=mesoroot,
        body_results_root=body_results_root,
        face_results_root=face_results_root,
        eye_results_root=eye_results_root,
        pupilroot=pupilroot,
    )
    dlc_configs = DLCModelConfigs.setup(
        bodymodeldir=bodymodeldir,
        facemodeldir=facemodeldir,
        eyemodeldir=eyemodeldir,
    )
    dest    = _setup_destination_paths(
        session,
        nwbroot=nwbroot,
    )
    return PathSettings(
        session=session,
        source=source,
        dlc_configs=dlc_configs,
        destination=dest,
    )
