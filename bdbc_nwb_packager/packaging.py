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
"""packaging procedures (based on the other modules)"""

from typing import Optional, Any
from typing_extensions import Self
from dataclasses import dataclass
from datetime import datetime as _datetime
from uuid import uuid4 as _uuid4
import warnings as _warnings

import pynwb as _nwb
from hdmf.build.warnings import (
    DtypeConversionWarning as _DtypeConversionWarning,
)

import bdbc_session_explorer as _sessx

from .types import (
    PathLike,
    PathsLike,
)
from . import (
    logging as _logging,
    configure as _configure,
    file_metadata as _file_metadata,
    timebases as _timebases,
    daq as _daq,
    trials as _trials,
    videos as _videos,
    imaging as _imaging,
    rois as _rois,
    tracking as _tracking,
)


@dataclass
class PackagingEnvironment:
    """the temporary storage of variables during the
    packaging procedure"""
    verbose: bool = True
    paths: Optional[_configure.PathSettings] = None
    metadata: Optional[_file_metadata.Metadata] = None
    trialspec: Optional[_sessx.TrialSpec] = None
    timebases: Optional[_timebases.Timebases] = None
    triggers: Optional[_timebases.PulseTriggers] = None
    has_trials_flag: bool = False
    nwbfile: Optional[_nwb.NWBFile] = None
    imgsetup: Optional[_imaging.NWBImagingSetup] = None
    imaging: Optional[_imaging.ImagingData] = None
    has_behavior_flag: bool = False
    downsampled: Optional[object] = None  # TODO

    def loaded_trials(self) -> bool:
        return self.has_trials_flag

    def has_videos(self) -> bool:
        return (self.timebases is not None) and (self.timebases.videos is not None)

    def loaded_tracking(self) -> bool:
        return self.has_behavior_flag

    def load_timebases(
        self,
        mismatch_tolerance: int = 0,
    ) -> Self:
        """loads the timebase/trigger info into this environment"""
        return load_timebases_impl(
            self,
            mismatch_tolerance=mismatch_tolerance,
        )

    def configure_nwbfile(self) -> Self:
        self.nwbfile = configure_nwbfile_impl(
            metadata=self.metadata,
            verbose=self.verbose,
        )
        return self

    def configure_downsampled_module(self) -> Self:
        self.downsampled = self.nwbfile.create_processing_module(
            name="downsampled",
            description="validated session data, down-sampled to the time base of imaging dF/F"
        )
        return self

    def add_raw_recordings(self, add_to_nwb: bool = True) -> Self:
        if add_to_nwb:
            for ts in _daq.iterate_raw_daq_recordings(
                metadata=self.metadata,
                rawfile=self.paths.source.rawdata,
                timebases=self.timebases,
                verbose=self.verbose,
            ):
                _logging.debug(f"add: {ts.name}")
                self.nwbfile.add_acquisition(ts)
        return self

    def add_downsampled_recordings(
        self,
    ) -> Self:
        for ts in _daq.iterate_downsampled_daq_recordings(
            metadata=self.metadata,
            rawfile=self.paths.source.rawdata,
            timebases=self.timebases,
            verbose=self.verbose
        ):
            _logging.debug(f"add: {ts.name}")
            self.downsampled.add(ts)
        return self

    def add_trials(self, add_to_nwb: bool = True) -> Self:
        return add_trials_impl(self, add_to_nwb=add_to_nwb, downsample=False)

    def add_downsampled_trials(
        self,
    ) -> Self:
        return add_trials_impl(self, downsample=True)

    def add_behavior_videos(
        self,
        copy_videos: bool = True,
    ) -> Self:
        if copy_videos:
            if self.has_videos():
                _videos.write_videos(
                    nwbfile=self.nwbfile,
                    metadata=self.metadata,
                    timebases=self.timebases,
                    paths=self.paths,
                    copy_files=True,
                    verbose=self.verbose
                )
            else:
                _logging.info(
                    'skip copying behavior videos (no videos were found)',
                )
        else:
            _logging.info(
                'skip copying behavior videos (so configured)',
            )
            _videos.write_videos(
                nwbfile=self.nwbfile,
                metadata=self.metadata,
                timebases=self.timebases,
                paths=self.paths,
                copy_files=False,
                verbose=self.verbose
            )
        return self

    def add_imaging_data(
        self,
        to_be_written: bool = True,
        used_for_rois: bool = True,
    ) -> Self:
        return add_imaging_data_impl(
            self,
            to_be_written=to_be_written,
            used_for_rois=used_for_rois,
        )

    def add_rois(
        self,
        register_rois: bool = True,
    ) -> Self:
        if register_rois:
            flat = self.imaging.flatten()
            meta = _file_metadata.read_roi_metadata(
                self.paths.source.mesoscaler,
                verbose=self.verbose
            )
            _rois.write_roi_entries(
                nwbfile=self.nwbfile,
                metadata=self.metadata,
                roimeta=meta,
                flattened_data=flat,
                setup=self.imgsetup,
                verbose=self.verbose,
            )
        else:
            _logging.info(
                'skip registering ROI data',
            )
        return self

    def add_tracking(self, add_to_nwb: bool = True) -> Self:
        return add_tracking_impl(
            self,
            add_to_nwb=add_to_nwb,
            downsample=False
        )

    def add_downsampled_tracking(self) -> Self:
        return add_tracking_impl(
            self,
            downsample=True
        )

    def write_nwb_file(self, mode: str = 'w') -> Self:
        outfile = self.paths.destination.nwb
        with _warnings.catch_warnings():
            _warnings.simplefilter('ignore', category=_DtypeConversionWarning)
            if not outfile.parent.exists():
                outfile.parent.mkdir(parents=True)
            with _nwb.NWBHDF5IO(
                outfile,
                mode=mode,
                manager=_nwb.get_manager(),
            ) as out:
                out.write(self.nwbfile)
        _logging.info(
            f"saved NWB file to: '{outfile}'",
        )
        return self


def is_missing(
    session: _sessx.Session,
    tasktype: str = 'cued-lever-pull',
    rawroot: Optional[PathsLike] = None,
    videoroot: Optional[PathLike] = None,
    mesoroot: Optional[PathLike] = None,
    nwbroot: Optional[PathLike] = None,
    body_results_root: Optional[PathLike] = None,
    face_results_root: Optional[PathLike] = None,
    eye_results_root: Optional[PathLike] = None,
    pupilroot: Optional[PathLike] = None,
    bodymodeldir: Optional[PathLike] = None,
    facemodeldir: Optional[PathLike] = None,
    eyemodeldir: Optional[PathLike] = None,
) -> bool:
    paths = _configure.setup_path_settings(
        session=session,
        rawroot=rawroot,
        videoroot=videoroot,
        mesoroot=mesoroot,
        body_results_root=body_results_root,
        face_results_root=face_results_root,
        eye_results_root=eye_results_root,
        pupilroot=pupilroot,
        bodymodeldir=bodymodeldir,
        facemodeldir=facemodeldir,
        eyemodeldir=eyemodeldir,
        nwbroot=nwbroot,
    )
    outfile = paths.destination.nwb
    return not outfile.exists()


def process(
    session: _sessx.Session,
    copy_videos: bool = True,
    register_rois: bool = True,
    write_imaging_frames: bool = True,
    add_downsampled: bool = True,
    only_downsampled: bool = False,
    override_metadata: Optional[dict[str, Any]] = None,
    overwrite: bool = False,
    verbose: bool = True,
    rawroot: Optional[PathsLike] = None,
    videoroot: Optional[PathLike] = None,
    mesoroot: Optional[PathLike] = None,
    nwbroot: Optional[PathLike] = None,
    body_results_root: Optional[PathLike] = None,
    face_results_root: Optional[PathLike] = None,
    eye_results_root: Optional[PathLike] = None,
    pupilroot: Optional[PathLike] = None,
    bodymodeldir: Optional[PathLike] = None,
    facemodeldir: Optional[PathLike] = None,
    eyemodeldir: Optional[PathLike] = None,
) -> Optional[_nwb.NWBFile]:
    """returns an NWB file in case it is newly computed."""
    paths = _configure.setup_path_settings(
        session=session,
        rawroot=rawroot,
        videoroot=videoroot,
        mesoroot=mesoroot,
        body_results_root=body_results_root,
        face_results_root=face_results_root,
        eye_results_root=eye_results_root,
        pupilroot=pupilroot,
        bodymodeldir=bodymodeldir,
        facemodeldir=facemodeldir,
        eyemodeldir=eyemodeldir,
        nwbroot=nwbroot,
    )
    if paths is None:
        _logging.warning("unknown error: paths failed to be configured")
        return
    outfile = paths.destination.nwb
    if outfile.exists() and (not overwrite):
        _logging.warning(f"file already exists: '{outfile}'")
        return
    metadata = _file_metadata.read_recordings_metadata(
        paths.session,
        paths.source.rawdata,
        override=override_metadata,
    )

    env = PackagingEnvironment(
        paths=paths,
        metadata=metadata,
        trialspec=session.trialspec,
        verbose=verbose,
    )
    env = env.configure_nwbfile()
    env = env.load_timebases()
    env = env.add_raw_recordings(add_to_nwb=(not only_downsampled))
    env = env.add_trials(add_to_nwb=(not only_downsampled))
    env = env.add_behavior_videos(copy_videos=copy_videos and (not only_downsampled))
    env = env.add_imaging_data(
        to_be_written=write_imaging_frames and (not only_downsampled),
        used_for_rois=register_rois,
    )
    env = env.add_rois(register_rois=register_rois)
    env = env.add_tracking(add_to_nwb=(not only_downsampled))
    if add_downsampled:
        env = env.configure_downsampled_module()
        env = env.add_downsampled_recordings()
        env = env.add_downsampled_trials()
        env = env.add_downsampled_tracking()
    env = env.write_nwb_file()
    return env.nwbfile


def load_timebases_impl(
    env: PackagingEnvironment,
    mismatch_tolerance: int = 0,
) -> PackagingEnvironment:
    """loads the timebase/trigger info into this environment"""
    triggers, timebases = _timebases.read_timebases(
        metadata=env.metadata,
        rawfile=env.paths.source.rawdata,
        verbose=env.verbose
    )
    triggers, timebases = _timebases.validate_timebase_with_rawdata(
        rawfile=env.paths.source.rawdata,
        triggers=triggers,
        timebases=timebases,
        verbose=env.verbose
    )
    triggers, timebases = _timebases.validate_timebase_with_imaging(
        rawfile=env.paths.source.rawdata,
        triggers=triggers,
        timebases=timebases,
        verbose=env.verbose
    )
    triggers, timebases = _timebases.validate_timebase_with_videos(
        paths=env.paths,
        triggers=triggers,
        timebases=timebases,
        tolerance=mismatch_tolerance,
        verbose=env.verbose,
    )
    env.triggers = triggers
    env.timebases = timebases
    return env


def configure_nwbfile_impl(
    metadata: _file_metadata.Metadata,
    verbose: bool = True
) -> _nwb.NWBFile:
    nwbfile = _nwb.NWBFile(
        session_description=metadata.session.description,
        identifier=str(_uuid4()),  # required
        session_start_time=metadata.session.start_time,  # required
        session_id=metadata.session_name,  # optional
        experimenter=metadata.session.experimenter,  # optional
        lab=metadata.session.lab,  # optional
        institution=metadata.session.institution,  # optional
        notes=metadata.session.notes,  # optional
    )
    nwbfile.subject = _nwb.file.Subject(
        age=metadata.subject.age,
        age__reference=metadata.subject.age_reference,
        description=metadata.subject.description,
        genotype=metadata.subject.genotype,
        sex=metadata.subject.sex,
        species=metadata.subject.species,
        subject_id=metadata.subject.ID,
        weight=metadata.subject.weight,
        date_of_birth=_datetime.combine(metadata.subject.date_of_birth, _datetime.min.time()).astimezone(None),
        strain=metadata.subject.strain,
    )
    _logging.info("configured an NWB file")
    return nwbfile


def add_trials_impl(
    env: PackagingEnvironment,
    add_to_nwb: bool = True,
    downsample: bool = False,
) -> PackagingEnvironment:
    # FIXME: override the "task type"
    # to correctly identify the columns
    # upon `write_trials`
    # (there may be a better way, though...)
    if not add_to_nwb:
        return env
    if env.paths.session.type != 'task':
        env.tasktype = env.paths.session.type

    if downsample:
        trials = _trials.load_downsampled_trials(
            env.paths.source.rawdata,
            trialspec=env.trialspec,
        )
        parent = env.downsampled
    else:
        trials = _trials.load_trials(
            env.paths.source.rawdata,
            trialspec=env.trialspec,
        )
        parent = env.nwbfile

    if trials is not None:
        _trials.write_trials(
            parent,
            trials,
            verbose=env.verbose,
        )
    else:
        if downsample:
            _logging.warning('no downsampled trials to be processed')
        else:
            _logging.warning('no trials to be processed')
        return env

    if downsample:
        _logging.info(
            f"done registering {trials.table.shape[0]} downsampled trials",
        )
    else:
        _logging.info(
            f"done registering {trials.table.shape[0]} trials",
        )
    env.has_trials_flag = True
    return env


def add_imaging_data_impl(
    env: PackagingEnvironment,
    to_be_written: bool = True,
    used_for_rois: bool = True,
) -> PackagingEnvironment:
    read_frames = (to_be_written or used_for_rois)
    env.imaging = _imaging.load_imaging_data(
        rawfile=env.paths.source.rawdata,
        timebases=env.timebases,
        read_frames=read_frames,
        verbose=env.verbose,
    )
    env.imgsetup = _imaging.setup_imaging_device(
        metadata=env.metadata,
        nwbfile=env.nwbfile,
        verbose=env.verbose,
    )
    _imaging.write_imaging_data(
        nwbfile=env.nwbfile,
        destination=env.paths.destination,
        frames=env.imaging,
        setup=env.imgsetup,
        write_frames=to_be_written,
        verbose=env.verbose
    )
    return env


def add_tracking_impl(
    env: PackagingEnvironment,
    add_to_nwb: bool = True,
    downsample: bool = False,
) -> PackagingEnvironment:
    if not env.paths.source.deeplabcut.has_any_results():
        if env.paths.session.has_any_videos():
            raise RuntimeError("DLC results not found")
        _logging.warning('skip registration of behavior tracking: no DeepLabCut output files were found.')
        return env

    if downsample:
        behav = env.downsampled
    elif (not add_to_nwb):
        env.has_behavior_flag = True
        return env
    else:
        behav = env.nwbfile.create_processing_module(
            name="behavior", description="Processed behavioral data"
        )

    for pose in _tracking.iterate_pose_estimations(
        paths=env.paths,
        timebases=env.timebases,
        triggers=env.triggers,
        downsample=downsample,
        verbose=env.verbose,
    ):
        _logging.debug(f"adding: {pose.name}")
        behav.add(pose)

    pupil = _tracking.load_pupil_fitting(
        paths=env.paths,
        timebases=env.timebases,
        triggers=env.triggers,
        downsample=downsample,
        verbose=env.verbose
    )
    if pupil is not None:
        for tracking in pupil:
            behav.add(tracking)

    env.has_behavior_flag = True
    return env
