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

from typing import Union, Optional, Tuple, Generator, Iterable
from typing_extensions import Self
from pathlib import Path
from collections import namedtuple as _namedtuple
from datetime import datetime as _datetime
import sys as _sys
import os as _os
import warnings as _warnings

import bdbc_session_explorer as _sessx

from . import (
    stdio as _stdio,
)


PathLike = Union[str, Path]
PathsLike = Union[PathLike, Iterable[PathLike]]
PATHSEP  = ':'


class PathSettings(_namedtuple('PathSettings', (
    'session',
    'source',
    'dlc_configs',
    'destination',
))):
    def has_behavior_videos(self) -> bool:
        return self.session.has_any_videos()


class SourcePaths(_namedtuple('SourcePaths', (
    'rawdata',
    'videos',
    'mesoscaler',
    'deeplabcut',
    'pupilfitting',
))):
    pass


class DestinationPaths(_namedtuple('DestinationPaths', (
    'nwb',
    'imaging',
    'videos',
))):
    @property
    def session_dir(self) -> Path:
        return self.nwb.parent


class DLCModelConfigs(_namedtuple('DLCModelConfigs', (
    'body',
    'face',
    'eye',
))):
    def items(self) -> Generator[Tuple[str, Path], None, None]:
        for fld, val in zip(self._fields, self):
            yield (fld, val)


class ImagingDataFiles(_namedtuple('ImagingDataFiles', (
    'B',
    'V',
))):
    def relative_to(self, ref: Path) -> Self:
        return self.__class__(
            B=self.B.relative_to(ref),
            V=self.V.relative_to(ref),
        )


class VideoDataFiles(_namedtuple('VideoDataFiles', (
    'body',
    'face',
    'eye',
))):
    def relative_to(self, ref: Path) -> Self:
        return self.__class__(
            body=self.body.relative_to(ref),
            face=self.face.relative_to(ref),
            eye=self.eye.relative_to(ref),
        )


def warn_message(msg: str, end='\n'):
    print(msg, end=end, file=_sys.stderr, flush=True)


def ensure_root_dir(
    name: str,
    envname: str,
    root: Optional[PathLike] = None,
) -> Path:
    if root is None:
        root = _os.environ.get(envname, None)
        if root is None:
            raise ValueError(f"specify `{name}` or the {envname} environment variable")
    return Path(root)


def dlc_model_dir(view: str, root: Optional[PathLike] = None) -> Path:
    return ensure_root_dir(
        name=f'{view}modeldir',
        envname=f'DLC_{view.upper()}MODEL_DIR',
        root=root
    )


def sessions_root_dir(root: Optional[PathLike] = None) -> Path:
    return ensure_root_dir(name='sessionroot', envname='SESSIONS_ROOT_DIR', root=root)


def mesoscaler_root_dir(root: Optional[PathLike] = None) -> Path:
    return ensure_root_dir(name='mesoroot', envname='MESOSCALER_ROOT_DIR', root=root)


def videos_root_dir(root: Optional[PathLike] = None) -> Path:
    return ensure_root_dir(name='videoroot', envname='VIDEOS_ROOT_DIR', root=root)


def dlcresults_root_dir(root: Optional[PathLike] = None) -> Path:
    return ensure_root_dir(name='dlcroot', envname='DLCRESULTS_ROOT_DIR', root=root)


def pupilfitting_root_dir(root: Optional[PathLike] = None) -> Path:
    return ensure_root_dir(name='pupilroot', envname='PUPILFITTING_ROOT_DIR', root=root)


def publication_root_dir(root: Optional[PathLike] = None) -> Path:
    return ensure_root_dir(name='nwbroot', envname='PUBLICATION_ROOT_DIR', root=root)


def rawdata_root_dirs(rootdirs: Optional[Iterable[PathLike]] = None) -> Tuple[Path]:
    if rootdirs is None:
        rootdirs = _os.environ.get('RAWDATA_ROOT_DIRS', None)
        if rootdirs is None:
            raise ValueError("specify `rawroot` or the RAWDATA_ROOT_DIRS environment variable")
        return tuple(Path(item) for item in str(rootdirs).split(PATHSEP) if len(item) > 0)
    elif isinstance(rootdirs, (str, Path)):
        return tuple(Path(item) for item in str(rootdirs).split(PATHSEP) if len(item) > 0)
    else:
        return tuple(Path(item) for item in rootdirs)


def setup_source_paths(
    session: _sessx.Session,
    rawroot: Optional[PathsLike] = None,
    videoroot: Optional[PathLike] = None,
    mesoroot: Optional[PathLike] = None,
    dlcroot: Optional[PathLike] = None,
    pupilroot: Optional[PathLike] = None,
) -> SourcePaths:
    rawroot   = rawdata_root_dirs(rawroot)
    videoroot = videos_root_dir(videoroot)
    mesoroot  = mesoscaler_root_dir(mesoroot)
    dlcroot   = dlcresults_root_dir(dlcroot)
    pupilroot = pupilfitting_root_dir(pupilroot)
    
    rawfile  = _sessx.locate_rawdata_file(session, rawroot)
    mesofile = _sessx.locate_mesoscaler_file(session, mesoroot=mesoroot)
    if (mesofile is None) or (not mesofile.exists()):
        raise FileNotFoundError(str(mesofile))
    videos = _sessx.video_files_from_session(
        session,
        videoroot=videoroot,
        error_handling='message',
    )
    dlcfiles = _sessx.dlc_output_files_from_session(session, dlcroot=dlcroot)
    # TODO:
    #   run DeepLabCut in case any of the output is missing?
    
    pupfile  = _sessx.locate_pupil_file(session, pupilroot=pupilroot)
    if (pupfile is None) or (not pupfile.exists()):
        # TODO: probably run fit_pupil?
        _sessx.core.message(f"***pupil file not found: {str(pupfile)}", verbose=True)
    return SourcePaths(
        rawdata=rawfile,
        videos=videos,
        mesoscaler=mesofile,
        deeplabcut=dlcfiles,
        pupilfitting=pupfile,
    )


def setup_destination_paths(
    session: _sessx.Session,
    nwbroot: Optional[Path] = None,
) -> DestinationPaths:
    nwbroot = publication_root_dir(nwbroot)
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
    videos   = VideoDataFiles(
        body=(videodir / f"{sessname}_body.mp4"),
        face=(videodir / f"{sessname}_face.mp4"),
        eye=(videodir / f"{sessname}_eye.mp4"),
    )
    return DestinationPaths(
        nwb=nwbfile,
        imaging=images,
        videos=videos,
    )


def setup_model_configs(
    **kwargs
) -> DLCModelConfigs:
    cfg = dict((fld, dlc_model_dir(fld, arg) / 'config.yaml') \
               for fld, arg in kwargs.items())
    return DLCModelConfigs(**cfg)


def setup_path_settings(
    session: _sessx.Session,
    rawroot: Optional[PathsLike] = None,
    videoroot: Optional[PathLike] = None,
    mesoroot: Optional[PathLike] = None,
    dlcroot: Optional[PathLike] = None,
    pupilroot: Optional[PathLike] = None,
    nwbroot: Optional[Path] = None,
    bodymodeldir: Optional[PathLike] = None,
    facemodeldir: Optional[PathLike] = None,
    eyemodeldir: Optional[PathLike] = None,
) -> Optional[PathSettings]:
    if not session.has_rawdata():
        return None
    source  = setup_source_paths(
        session,
        rawroot=rawroot,
        videoroot=videoroot,
        mesoroot=mesoroot,
        dlcroot=dlcroot,
        pupilroot=pupilroot,
    )
    dlc_configs = setup_model_configs(
        body=bodymodeldir,
        face=facemodeldir,
        eye=eyemodeldir,
    )
    dest    = setup_destination_paths(
        session,
        nwbroot=nwbroot,
    )
    return PathSettings(
        session=session,
        source=source,
        dlc_configs=dlc_configs,
        destination=dest,
    )

