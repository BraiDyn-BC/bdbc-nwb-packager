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
    _core.print_message("configured an NWB file.", verbose=verbose)
    return nwbfile


def package_nwb(
    paths: _paths.PathSettings,
    overwrite: bool = False,
    verbose: bool = True,
) -> _nwb.NWBFile:
    outfile = paths.destination.nwb
    if outfile.exists() and (not overwrite):
        _core.print_message(f"***file already exists: '{outfile}'", verbose=verbose)
        # FIXME: load the contents from the file to return
        return
    
    metadata = _metadata.metadata_from_rawdata(paths.source.rawdata, session_type=paths.session.type)
    nwbfile = configure_nwb_file(metadata)

    # raw DAQ data (incl. trials)
    timebases = _core.load_timebases(metadata, paths.source.rawdata)
    trials = _trials.load_trials(paths.source.rawdata, timebases)
    _trials.write_trials(nwbfile, trials, verbose=verbose)
    for ts in _daq.iterate_raw_daq_recordings(metadata, paths.source.rawdata, timebases):
        nwbfile.add_acquisition(ts)

    # downsampled DAQ data
    ds = nwbfile.create_processing_module(
        name="downsampled", description="downsampled DAQ acquisition"
    )
    for ts in _daq.iterate_downsampled_daq_recordings(metadata, paths.source.rawdata, timebases):
        ds.add(ts)

    _videos.write_videos(
        nwbfile=nwbfile,
        metadata=metadata,
        timebases=timebases,
        paths=paths,
        verbose=verbose,
    )

    # imaging data
    frames = _imaging.load_imaging_data(paths.source.rawdata, timebases=timebases, verbose=verbose)
    setup  = _imaging.setup_imaging_device(metadata, nwbfile, verbose=verbose)
    _imaging.write_imaging_data(
        nwbfile=nwbfile, 
        destination=paths.destination,
        frames=frames,
        setup=setup,
    )

    # rois
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

    if paths.has_behavior_videos():
        # DLC results / pupil
        behav = nwbfile.create_processing_module(
            name="behavior", description="Processed behavioral data"
        )
        for pose in _dlc.iterate_pose_estimations(
            nwbfile=nwbfile,
            timebases=timebases,
            paths=paths,
            verbose=verbose,
        ):
            behav.add(pose)
    
        pupil =  _pupil.load_pupil_fitting(
            paths=paths,
            timebases=timebases,
            verbose=verbose
        )
        if pupil is not None:
            for tracking in pupil:
                behav.add(tracking)

    with _warnings.catch_warnings():
        _warnings.simplefilter('ignore', category=_DtypeConversionWarning)
        with _nwb.NWBHDF5IO(
            outfile, 
            mode="w",
            manager=_nwb.get_manager(),
        ) as out:
            out.write(nwbfile)
    
    _core.print_message(f"saved NWB file to: '{outfile}'", verbose=verbose)
    return nwbfile
