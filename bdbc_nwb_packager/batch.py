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

from typing import Optional, Generator, Callable, Union, Iterable, Dict, Any
from pathlib import Path
from datetime import datetime as _datetime
from time import time as _now

import bdbc_session_explorer as _sessx

from . import (
    stdio as _stdio,
    paths as _paths,
    packaging as _packaging,
)

PathLike = Union[str, Path]
PathsLike = Union[PathLike, Iterable[PathLike]]


def run_batch(
    animal: Optional[str] = None,
    fromdate: Optional[str] = None,
    todate: Optional[str] = None,
    type: Optional[str] = None,
    copy_videos: bool = True,
    register_rois: bool = True,
    write_imaging_frames: bool = True,
    add_downsampled: bool = True,
    override_metadata: Optional[str] = None,
    overwrite: bool = False,
    verbose: bool = True,
    tasktype: str = 'cued_lever_pull',
    sessroot: Optional[PathLike] = None,
    rawroot: Optional[PathsLike] = None,
    videoroot: Optional[PathLike] = None,
    mesoroot: Optional[PathLike] = None,
    dlcroot: Optional[PathLike] = None,
    pupilroot: Optional[PathLike] = None,
    nwbroot: Optional[Path] = None,
    body_model_dir: Optional[PathLike] = None,
    face_model_dir: Optional[PathLike] = None,
    eye_model_dir: Optional[PathLike] = None,
):
    override_metadata = parse_overridden_metadata(override_metadata)

    for sess in filter_sessions(
        animal=animal,
        fromdate=fromdate,
        todate=todate,
        type=type,
        sessions_root_dir=sessroot,
    ):
        _stdio.message(f"[{sess.batch}/{sess.animal} {sess.longdate} ({sess.longtype})]", verbose=verbose)
        if not sess.has_rawdata():
            _stdio.message("***no raw data file", end='\n\n', verbose=verbose)
            continue
        start = _now()
        paths = _paths.setup_path_settings(
            session=sess,
            rawroot=rawroot,
            videoroot=videoroot,
            mesoroot=mesoroot,
            dlcroot=dlcroot,
            pupilroot=pupilroot,
            nwbroot=nwbroot,
            bodymodeldir=body_model_dir,
            facemodeldir=face_model_dir,
            eyemodeldir=eye_model_dir,
        )
        _packaging.package_nwb(
            paths=paths,
            tasktype=tasktype,
            copy_videos=copy_videos,
            register_rois=register_rois,
            write_imaging_frames=write_imaging_frames,
            add_downsampled=add_downsampled,
            override_metadata=override_metadata,
            overwrite=overwrite,
            verbose=verbose,
        )
        stop = _now()
        _stdio.message(
            f"(took {(stop - start) / 60:.1f} min to process this session)",
            end='\n\n',
            verbose=verbose
        )


def parse_overridden_metadata(spec: Optional[str]) -> Optional[Dict[str, Any]]:

    def _as_int(v) -> Optional[int]:
        try:
            return int(v)
        except ValueError:
            return None

    def _as_float(v) -> Optional[float]:
        try:
            return float(v)
        except ValueError:
            return None

    def _normalize(v: str) -> Union[str, int, float]:
        if (iv := _as_int(v)) is not None:
            return iv
        elif (fv := _as_float(v)) is not None:
            return fv
        else:
            return v

    if spec is None:
        return None
    rawspecs = tuple(item.strip() for item in spec.split(',') if len(item) > 0)
    if len(rawspecs) == 0:
        return None
    specs = []
    for rawspec in rawspecs:
        if len(rawspec) == 0:
            continue
        elif '=' not in rawspec:
            _stdio.message("***unknown format for metadata spec: '{rawspec}'", verbose=True)
            continue
        fld, rawval = tuple(item.strip() for item in rawspec.split('='))
        specs.append((fld, _normalize(rawval)))
    return dict(specs)


def filter_sessions(
    animal: Optional[str] = None,
    fromdate: Optional[str] = None,
    todate: Optional[str] = None,
    type: Optional[str] = None,
    sessions_root_dir: Optional[PathLike] = None,
) -> Generator[_sessx.Session, None, None]:
    is_animal = matcher.animal(animal)
    is_date = matcher.date(from_date=fromdate, to_date=todate)
    is_type = matcher.session_type(type)

    def _matches(session: _sessx.Session) -> bool:
        return is_animal(session.animal) and is_date(session.date) and is_type(session.type)

    sessions_root_dir = _paths.sessions_root_dir(sessions_root_dir)
    yield from (sess for sess in _sessx.iterate_sessions(sessions_root_dir)
                if _matches(sess))


class matcher:
    @staticmethod
    def matches_all(query: str) -> bool:
        return True

    @staticmethod
    def animal(ref: Optional[str]) -> Callable[[str], bool]:
        if ref is None:
            return matcher.matches_all
        else:
            refs = tuple(item.strip() for item in ref.split(','))

            def match(query: str) -> bool:
                return (query in refs)

            return match

    @staticmethod
    def from_date(ref: Optional[str]) -> Callable[[_datetime], bool]:
        if ref is None:
            return matcher.matches_all
        else:
            ref = _datetime.strptime(ref, '%y%m%d')

            def match(query: _datetime) -> bool:
                return (query >= ref)

            return match

    @staticmethod
    def to_date(ref: Optional[str]) -> Callable[[_datetime], bool]:
        if ref is None:
            return matcher.matches_all
        else:
            ref = _datetime.strptime(ref, '%y%m%d')

            def match(query: _datetime) -> bool:
                return (query <= ref)

            return match

    @staticmethod
    def date(
        from_date: Optional[str],
        to_date: Optional[str]
    ) -> Callable[[str], bool]:
        from_date = matcher.from_date(from_date)
        to_date = matcher.to_date(to_date)

        def match(query: _datetime) -> bool:
            return from_date(query) and to_date(query)

        return match

    @staticmethod
    def session_type(
        ref: Optional[str]
    ) -> Callable[[str], bool]:
        if ref is None:
            return matcher.matches_all
        else:
            mapping = dict(ss='sensory-stim', rest='resting-state')
            refs = tuple(item.strip() for item in ref.split(','))
            norm = []
            for ref in refs:
                if ref in mapping.keys():
                    ref = mapping[ref]
                if ref not in ('task', 'resting-state', 'sensory-stim'):
                    raise ValueError(f"expected one of ('task', 'resting-state', 'sensory-stim'), got '{ref}'")
                norm.append(ref)
            refs = tuple(norm)

            def match(query: str) -> bool:
                return (query in refs)
            return match
