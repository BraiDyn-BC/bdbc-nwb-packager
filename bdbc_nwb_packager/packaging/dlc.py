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

from typing import Generator
from collections import namedtuple as _namedtuple
from time import time as _now
from uuid import uuid4 as _uuid4
import warnings as _warnings

from pynwb import (
    NWBFile as _NWBFile,
)
from neuroconv.datainterfaces import (
    DeepLabCutInterface as _DeepLabCutInterface,
)
from ndx_pose import (
    PoseEstimationSeries as _PoseEstimationSeries,
    PoseEstimation as _PoseEstimation,
)

from .. import (
    paths as _paths,
)
from . import (
    core as _core,
)

PathLike = _core.PathLike

NAME_MAPPINGS = {
    'body': 'forelimb',
    'face': 'face',
    'eye': 'eye',
}


def iterate_pose_estimations(
    nwbfile: _NWBFile,
    timebases: _core.Timebases,
    paths: _paths.PathSettings,
    verbose: bool = True,
) -> Generator[_PoseEstimation, None, None]:
    #
    # IMPLEMENTATION NOTE:
    #
    # Here we convert each interface once to a temporal NWB file,
    # then copy its contents to the main NWB file with renamed PoseEstimation
    # (though it is admittedly hacky and complicated...)
    #
    # Probably it would be better instead to directly read from
    # the DeepLabCut output file to create a PoseEstimation entry.
    #
    def _setup_temporal_nwb(view: str, target: str) -> _NWBFile:
        entry = _DeepLabCutInterface(
            file_path=getattr(paths.source.deeplabcut, view),
            config_file_path=getattr(paths.dlc_configs, view),
            subject_name=nwbfile.subject.subject_id,
            verbose=True,
        )
        meta = entry.get_metadata()
        meta["NWBFile"].update(session_start_time=nwbfile.session_start_time)
        entry.set_aligned_timestamps(timebases.videos)
        
        # add to a "temporal" nwbfile
        tempfile = _NWBFile(
            session_description=f"{view}:{target}", 
            identifier=str(_uuid4()),  # required
            session_start_time=nwbfile.session_start_time,  # required
            experimenter=[])
        entry.add_to_nwbfile(nwbfile=tempfile, metadata=meta)
        return tempfile

    def _setup_pose_estimation(tempfile: _NWBFile) -> _PoseEstimation:
        keypoints = []
        view, model = tempfile.session_description.split(':')
        for kpt in tempfile.processing["behavior"]["PoseEstimation"].nodes:
            data = tempfile.processing["behavior"]["PoseEstimation"][kpt].data
            confidence = tempfile.processing["behavior"]["PoseEstimation"][kpt].confidence
            keypoint = _PoseEstimationSeries(
                name=f"{model}_{kpt}",
                description=f"Keypoint '{kpt}' from the {view} video.",
                data=data,
                unit="pixels",
                reference_frame="(0,0) corresponds to the top left corner of the video.",
                timestamps=timebases.videos,
                confidence=confidence,
                confidence_definition="Softmax output of the deep neural network.",
            )
            keypoints.append(keypoint)
            
        pose = _PoseEstimation(
            name=f"PoseEstimation_{model}",
            pose_estimation_series = keypoints,
            original_videos=tempfile.processing["behavior"]["PoseEstimation"].original_videos,
            dimensions=tempfile.processing["behavior"]["PoseEstimation"].dimensions,
            scorer=tempfile.processing["behavior"]["PoseEstimation"].scorer,
            source_software="DeepLabCut",
            nodes=tempfile.processing["behavior"]["PoseEstimation"].nodes,
        )
        videopath = getattr(paths.destination.videos.relative_to(paths.destination.session_dir), view)
        pose.fields['original_videos'] = str(videopath)
        return pose

    _core.print_message('registering DeepLabCut:', end=' ', verbose=verbose)
    start = _now()
    with _warnings.catch_warnings():
        # ignore warnings: about :
        # - 'model metadata'
        # - 'associated video' (later set in _setup_pose_estimation())
        # - pandas.DataFrame.groupby() with axis=1
        #
        _warnings.filterwarnings('ignore', category=UserWarning, message='Metadata')
        _warnings.filterwarnings('ignore', category=UserWarning, message='The video file corresponding to')
        _warnings.filterwarnings('ignore', category=FutureWarning, message='DataFrame.groupby with axis=1')
        for view, model in NAME_MAPPINGS.items():
            if getattr(paths.source.deeplabcut, view) is None:
                # FIXME: make up empty PoseEstimation?
                _core.print_message(f"***missing the {view} model results", verbose=verbose)
                continue
            _core.print_message(f"{view} model...", end=' ', verbose=verbose)
            tempfile = _setup_temporal_nwb(view, model)
            yield _setup_pose_estimation(tempfile)
    stop = _now()
    _core.print_message(f"done (took {(stop - start):.1f} sec).", verbose=verbose)
