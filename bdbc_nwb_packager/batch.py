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

from typing import Optional, Generator, Callable, Union, Iterable
from pathlib import Path
from datetime import datetime as _datetime
from time import time as _now
import sys as _sys

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
    overwrite: bool = False,
    verbose: bool = True,
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
    for sess in filter_sessions(
        animal=animal,
        fromdate=fromdate,
        todate=todate,
        type=type,
        sessions_root_dir=sessroot,
    ):
        _stdio.message(f"[{sess.batch}/{sess.animal} {sess.longdate} ({sess.longtype})]", verbose=verbose)
        if not sess.has_rawdata():
            _stdio.message(f"***no raw data file", end='\n\n', verbose=verbose)
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
            overwrite=overwrite,
            verbose=verbose,
        )
        stop = _now()
        _stdio.message(
            f"(took {(stop - start) / 60:.1f} min to process this session)",
            end='\n\n',
            verbose=verbose
        )


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
    yield from (sess for sess in _sessx.iterate_sessions(sessions_root_dir) \
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

