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
    logging as _logging,
    packaging as _packaging,
)


def find_missing(
    animal: Optional[str] = None,
    batch: Optional[str] = None,
    fromdate: Optional[str] = None,
    todate: Optional[str] = None,
    type: Optional[str] = None,
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
    logger = _logging.get_logger(file_output=True, prefix='batch_')

    missing = []
    for sess in _sessx.iterate_sessions(
        animal=animal,
        batch=batch,
        fromdate=fromdate,
        todate=todate,
        type=type,
        sessions_root_dir=sessroot,
    ):
        if not sess.has_rawdata():
            continue
        if _packaging.is_missing(
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
        ):
            missing.append(sess)
    for sess in missing:
        logger.info(f"{sess.batch}/{sess.animal}/{sess.longdate} ({sess.longtype})")
    logger.info(f"--> log file: {_logging.get_file_path()}")


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
    level = _logging.INFO if verbose else _logging.WARNING
    logger = _logging.get_logger(file_output=True, console_level=level, prefix='batch_')
    logger.info(f"log file: {_logging.get_file_path()}")

    override_metadata = parse_overridden_metadata(override_metadata)
    sessions_processed = []
    sessions_without_rawdata = []
    sessions_with_problem = []

    for sess in _sessx.iterate_sessions(
        animal=animal,
        batch=batch,
        fromdate=fromdate,
        todate=todate,
        type=type,
        sessions_root_dir=sessroot,
        verbose=verbose,
    ):
        sessions_processed.append(sess)
        logger.info(f"====== {sess.batch}/{sess.animal} {sess.longdate} ({sess.longtype}) ======")
        if not sess.has_rawdata():
            logger.warning(f"{sess.batch}/{sess.animal}/{sess.longdate}({sess.longtype}): no raw data file")
            sessions_without_rawdata.append(sess)
            logger.info("========================")
            continue
        start = _now()
        try:
            _packaging.process(
                session=sess,
                copy_videos=copy_videos,
                register_rois=register_rois,
                write_imaging_frames=write_imaging_frames,
                add_downsampled=add_downsampled,
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
                overwrite=overwrite,
                verbose=verbose,
            )
        except OSError:
            raise
        except BaseException as e:
            _logging.exception(e)
            sessions_with_problem.append((sess, e))
        stop = _now()
        _logging.info(f"====== (took {(stop - start) / 60:.1f} min) ======")

    # finalize
    _logging.info(f"processed {len(sessions_processed)} sessions")
    if len(sessions_without_rawdata) > 0:
        _logging.info(f"--> {len(sessions_without_rawdata)}/{len(sessions_processed)} sessions without raw data: ")
        for sess in sessions_without_rawdata:
            _logging.info(f"  - {sess.animal} ({sess.batch}) {sess.longdate} ({sess.longtype})")
    if len(sessions_with_problem) > 0:
        _logging.info(f"--> {len(sessions_with_problem)}/{len(sessions_processed)} sessions with problems: ")
        for sess, e in sessions_with_problem:
            _logging.info(f"  - {sess.animal} ({sess.batch}) {sess.longdate} ({sess.longtype})")
            _logging.exception(e)
    logger.info(f"--> log file: {_logging.get_file_path()}")


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
            _logging.warning("unknown format for metadata spec: '{rawspec}'")
            continue
        fld, rawval = tuple(item.strip() for item in rawspec.split('='))
        specs.append((fld, _normalize(rawval)))
    return dict(specs)
