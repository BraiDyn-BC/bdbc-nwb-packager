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

from typing import Optional, Dict, Any
from uuid import uuid4 as _uuid4
from datetime import datetime as _datetime
import warnings as _warnings

import numpy as _np
import numpy.typing as _npt
import h5py as _h5
import pynwb as _nwb
from hdmf.build.warnings import (
    DtypeConversionWarning as _DtypeConversionWarning,
)

from .. import (
    stdio as _stdio,
    metadata as _metadata,
    paths as _paths,
)
from . import (
    core as _core,
    trials as _trials,
    daq as _daq,
    videos as _videos,
    imaging as _imaging,
    rois as _rois,
    dlc as _dlc,
    pupil as _pupil,
)

PathLike = _core.PathLike


def configure_nwb_file(
    metadata: _metadata.Metadata,
    verbose: bool = True,
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
        date_of_birth = _datetime.combine(metadata.subject.date_of_birth, _datetime.min.time()).astimezone(None),
        strain = metadata.subject.strain,
    )
    _stdio.message("configured an NWB file.", verbose=verbose)
    return nwbfile


def package_nwb(
    paths: _paths.PathSettings,
    tasktype: str = 'cued_lever_pull',
    copy_videos: bool = True,
    register_rois: bool = True,
    write_imaging_frames: bool = True,
    add_downsampled: bool = True,
    override_metadata: Optional[Dict[str, None]] = None,
    overwrite: bool = False,
    verbose: bool = True,
) -> _nwb.NWBFile:
    outfile = paths.destination.nwb
    if outfile.exists() and (not overwrite):
        _stdio.message(f"***file already exists: '{outfile}'", verbose=verbose)
        # FIXME: load the contents from the file to return
        return
    
    metadata = _metadata.metadata_from_rawdata(
        paths.session,
        paths.source.rawdata,
        override=override_metadata,
    )
    nwbfile = configure_nwb_file(metadata)

    # raw DAQ data
    triggers, timebases = _core.load_timebases(metadata, paths.source.rawdata)
    for ts in _daq.iterate_raw_daq_recordings(metadata, paths.source.rawdata, timebases):
        nwbfile.add_acquisition(ts)

    # if paths.session.type == 'task':
    #     # add trials
    #     trials = _trials.load_trials(
    #         paths.source.rawdata,
    #         timebases,
    #         tasktype=tasktype,
    #         verbose=verbose,
    #     )
    #     _trials.write_trials(
    #         nwbfile,
    #         trials,
    #         tasktype=tasktype,
    #         verbose=verbose,
    #     )
    # else:
    #     trials = None

    _videos.write_videos(
        nwbfile=nwbfile,
        metadata=metadata,
        timebases=timebases,
        paths=paths,
        copy_files=copy_videos,
        verbose=verbose,
    )

    # imaging data
    frames = _imaging.load_imaging_data(
        paths.source.rawdata,
        timebases=timebases,
        read_frames=write_imaging_frames or register_rois,
        verbose=verbose
    )
    setup  = _imaging.setup_imaging_device(metadata, nwbfile, verbose=verbose)
    _imaging.write_imaging_data(
        nwbfile=nwbfile, 
        destination=paths.destination,
        frames=frames,
        setup=setup,
        write_frames=write_imaging_frames,
        verbose=verbose,
    )

    # rois
    if register_rois:
        fdata  = frames.flatten(verbose=verbose)
        roimeta = _metadata.read_roi_metadata(paths.source.mesoscaler)
        _rois.write_roi_entries(
            nwbfile=nwbfile,
            metadata=metadata,
            roimeta=roimeta,
            flattened_data=fdata,
            setup=setup,
            verbose=verbose,
        )
    else:
        _stdio.message('***skip registering ROI data', verbose=verbose)

    if paths.has_behavior_videos():
        # DLC results / pupil
        behav = nwbfile.create_processing_module(
            name="behavior", description="Processed behavioral data"
        )
        for pose in _dlc.iterate_pose_estimations(
            paths=paths,
            timebases=timebases,
            triggers=triggers,
            downsample=False,
            verbose=verbose,
        ):
            behav.add(pose)
    
        _stdio.message(f"registering pupil fitting...", end=' ', verbose=verbose)
        pupil =  _pupil.load_pupil_fitting(
            paths=paths,
            timebases=timebases,
            triggers=triggers,
            downsample=False,
            verbose=verbose
        )
        if pupil is not None:
            for tracking in pupil:
                behav.add(tracking)
            _stdio.message("done.", verbose=verbose)
    else:
        behav = None

    # downsampled DAQ data
    # ====== Fix _daq.iterate_downsampled_daq_recordings and _trials.load_downsampled_trials(
    if add_downsampled:
        ds = nwbfile.create_processing_module(
            name="downsampled",
            description="validated session data, down-sampled to the time base of imaging dF/F"
        )

        for ts in _daq.iterate_downsampled_daq_recordings(
            metadata,
            paths.source.rawdata,
            timebases
        ):
            ds.add(ts)

        if trials is not None:
            trials_ds = _trials.load_downsampled_trials(
                trials,
                timebases,
                tasktype=tasktype,
                verbose=verbose
            )
            # _load_downsampled_trialsを書き換えて読み込んでDF出力をつくる
            # trials_ds = _trials._load_downsampled_trials(
            #     rawfile = paths.source.rawdata
            # )
            _trials.write_trials(
                ds,
                trials_ds,
                tasktype=tasktype,
                verbose=verbose,
            )

        if behav is not None:
            for pose in _dlc.iterate_pose_estimations(
                paths=paths,
                triggers=triggers,
                timebases=timebases,
                downsample=True,
                verbose=verbose,
            ):
                ds.add(pose)

            _stdio.message(f"registering pupil fitting...", end=' ', verbose=verbose)
            pupil =  _pupil.load_pupil_fitting(
                paths=paths,
                timebases=timebases,
                triggers=triggers,
                downsample=True,
                verbose=verbose
            )
            if pupil is not None:
                for tracking in pupil:
                    ds.add(tracking)
                _stdio.message("done.", verbose=verbose)

    with _warnings.catch_warnings():
        _warnings.simplefilter('ignore', category=_DtypeConversionWarning)
        with _nwb.NWBHDF5IO(
            outfile, 
            mode="w",
            manager=_nwb.get_manager(),
        ) as out:
            out.write(nwbfile)
    
    _stdio.message(f"saved NWB file to: '{outfile}'", verbose=verbose)
    return nwbfile

