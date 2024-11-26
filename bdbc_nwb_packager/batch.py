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
"""the procedure(s) related to batch-processing of multiple sessions.
Also used as the entry point of the terminal command `package-nwb`"""

from typing import Optional, Union, Any
from pathlib import Path
from time import time as _now

import bdbc_session_explorer as _sessx

from .types import (
    PathLike,
    PathsLike,
)
from . import (
    stdio as _stdio,
    packaging as _packaging,
)


def run_batch(
    animal: Optional[str] = None,
    batch: Optional[str] = None,
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
    tasktype: str = 'cued-lever-pull',
    sessroot: Optional[PathLike] = None,
    rawroot: Optional[PathsLike] = None,
    videoroot: Optional[PathLike] = None,
    mesoroot: Optional[PathLike] = None,
    body_results_root: Optional[PathLike] = None,
    face_results_root: Optional[PathLike] = None,
    eye_results_root: Optional[PathLike] = None,
    pupilroot: Optional[PathLike] = None,
    nwbroot: Optional[Path] = None,
    body_model_dir: Optional[PathLike] = None,
    face_model_dir: Optional[PathLike] = None,
    eye_model_dir: Optional[PathLike] = None,
):
    override_metadata = parse_overridden_metadata(override_metadata)

    for sess in _sessx.iterate_sessions(
        animal=animal,
        batch=batch,
        fromdate=fromdate,
        todate=todate,
        type=type,
        sessions_root_dir=sessroot,
        verbose=verbose,
    ):
        _stdio.message(f"[{sess.batch}/{sess.animal} {sess.longdate} ({sess.longtype})]", verbose=verbose)
        if not sess.has_rawdata():
            _stdio.message("***no raw data file", end='\n\n', verbose=verbose)
            continue
        start = _now()
        _packaging.process(
            session=sess,
            tasktype=tasktype,
            rawroot=rawroot,
            videoroot=videoroot,
            mesoroot=mesoroot,
            body_results_root=body_results_root,
            face_results_root=face_results_root,
            eye_results_root=eye_results_root,
            pupilroot=pupilroot,
            bodymodeldir=body_model_dir,
            facemodeldir=face_model_dir,
            eyemodeldir=eye_model_dir,
            nwbroot=nwbroot,
            override_metadata=override_metadata,
            verbose=verbose,
        )
        stop = _now()
        _stdio.message(
            f"(took {(stop - start) / 60:.1f} min to process this session)",
            end='\n\n',
            verbose=verbose
        )


def parse_overridden_metadata(spec: Optional[str]) -> Optional[dict[str, Any]]:

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
