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

import numpy as _np
import pandas as _pd
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
    stdio as _stdio,
)
from . import (
    core as _core,
    videos as _videos,
)

PathLike = _core.PathLike

NAME_MAPPINGS = {
    'body': 'forelimb',
    'face': 'face',
    'eye': 'eye',
}


def iterate_pose_estimations(
    paths: _paths.PathSettings,
    timebases: _core.Timebases,
    mismatch_tolerance: int = 1,
    verbose: bool = True,
) -> _PoseEstimation:
    """load DeepLabCut results from corresponding HDF5 files,
    and returns a generator iterating over PoseEstimation objects."""

    # TODO: this procedure must be (formally) during
    # e.g. imaging/videos registration
    def _setup_clips(num_frames, num_pulses):
        """a temporary solution until sizes of timebases/videos are nicely handled."""
        delta = num_frames - num_pulses
        if delta == 0:
            timeclip = slice(None, None)
            videoclip = slice(None, None)
        elif 0 < delta < mismatch_tolerarnce:
            timeclip  = slice(None, None)
            videoclip = slice(0, num_pulses)
        elif (-1 * mismatch_tolerance) <= delta < 0:
            timeclip = slice(0, num_frames)
            videoclip = slice(None, None)
        else:
            raise RuntimeError(f"untolerable mismatch: {num_pulses} pulses vs {num_frames} frames")
        return timeclip, videoclip

    _stdio.message('registering DeepLabCut:', end=' ', verbose=verbose)
    destvideos = paths.destination.videos.relative_to(paths.destination.session_dir)
    for view, model in NAME_MAPPINGS.items():
        srcvideo = getattr(paths.source.videos, view)
        # TODO: handle cases with `srcvideo.path is None`
        if srcvideo.path is None:
            _stdio.message(
                f"***missing the {view} video...",
                end=' ',
                verbose=verbose
            )
            continue
        tclip, vclip = _setup_clips(srcvideo.num_frames, timebases.videos.size)
        
        dlcpath = getattr(paths.source.deeplabcut, view)
        if dlcpath is None:
            _stdio.message(
                f"***missing the {view} model results...",
                end=' ',
                verbose=verbose
            )
            continue
        dlctab = _pd.read_hdf(dlcpath, key='df_with_missing')
        scorer = dlctab.columns[0][0]
        dlctab = dlctab.iloc[vclip]
        t = timebases.videos[tclip]
        assert dlctab.shape[0] == t.size

        _stdio.message(f"{view} model...", end=' ', verbose=verbose)
        series = []
        keypoints = tuple(set(col[1] for col in dlctab.columns))
        
        # TODO: think over about what names may be appropriate
        pose_estimation_name = f"{view}_video_keypoints"
        node_names = [f"{kpt}" for kpt in keypoints]
        for kpt, node_name in zip(keypoints, node_names):
            data = _np.stack([
                dlctab[scorer, kpt, 'x'].values,
                dlctab[scorer, kpt, 'y'].values
            ], axis=1)
            series.append(_PoseEstimationSeries(
                name=node_name,
                description=f"Keypoint '{kpt}' from the {view} video.",
                data=data,
                unit='pixels',
                reference_frame="(0,0) corresponds to the top left corner of the video.",
                timestamps=t,
                confidence=_np.array(dlctab[scorer, kpt, 'likelihood'].values),
                confidence_definition="Softmax output of the deep neural network.",
            ))
        yield _PoseEstimation(
            name=pose_estimation_name,
            description=f"Estimated positions of keypoints from the {view} view frames using DeepLabCut.",
            pose_estimation_series=series,
            nodes=node_names,
            original_videos=[str(getattr(destvideos, view))],
            labeled_videos=[],
            dimensions=_np.array(
                [[srcvideo.width, srcvideo.height]], dtype=_np.uint16,
            ),  # pixel dimensions of the video
            scorer=scorer,
            source_software="DeepLabCut",
            source_software_version="2.3.10",
        )

