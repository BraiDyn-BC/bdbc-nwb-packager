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

from typing import Optional
import warnings as _warnings

import numpy as _np
from ndx_pose import (
    PoseEstimationSeries as _PoseEstimationSeries,
    PoseEstimation as _PoseEstimation,
)

from .. import (
    stdio as _stdio,
    configure as _configure,
    timebases as _timebases,
)
from . import (
    validation as _validation,
    alignment as _alignment,
)

NAME_MAPPINGS = {
    'body': 'forelimb',
    'face': 'face',
    'eye': 'eye',
}


def iterate_pose_estimations(
    paths: _configure.PathSettings,
    timebases: _timebases.Timebases,
    triggers: Optional[_timebases.PulseTriggers] = None,
    mismatch_tolerance: int = 0,
    downsample: bool = False,
    downsample_pcutoff: float = 0.2,
    verbose: bool = True,
) -> _PoseEstimation:
    """load DeepLabCut results from corresponding HDF5 files,
    and returns a generator iterating over PoseEstimation objects."""

    # NOTE:
    # The use of PoseEstimation and PoseEstimationSeries seems to elicit
    # PendingDeprecationWarning (that cannot be controlled from our side).
    # this is a temporary solution until the NWB group takes care of it.
    with _warnings.catch_warnings():
        _warnings.filterwarnings('ignore', category=PendingDeprecationWarning)

        destvideos = paths.destination.videos.relative_to(paths.destination.session_dir)
        for view, model in NAME_MAPPINGS.items():
            srcvideo = getattr(paths.source.videos, view)
            if srcvideo.path is None:
                _stdio.message(
                    f"***missing the {view} video",
                    verbose=verbose
                )
                continue
            dlcpath = getattr(paths.source.deeplabcut, view)
            if dlcpath is None:
                _stdio.message(
                    f"***missing the {view} model results",
                    end=' ',
                    verbose=verbose
                )
                continue
            elif downsample:
                _stdio.message(
                    f'registering downsampled estimations from the {view} video...',
                    end='',
                    verbose=verbose
                )
            else:
                _stdio.message(
                    f'registering estimations from the {view} video...',
                    end='',
                    verbose=verbose
                )

            t, trigs, dlctab = _validation.prepare_table_results(
                tabpath=dlcpath,
                srcvideo=srcvideo,
                t_video=timebases.videos,
                triggers=triggers.videos,
                mismatch_tolerance=mismatch_tolerance,
            )

            if downsample:
                t = timebases.dFF

                def _downsample(x):
                    u = _alignment.upsample(
                        x,
                        size=timebases.raw.size,
                        pulseidxx=trigs,
                    )
                    return _alignment.downsample(
                        u,
                        pulseidxx=triggers.dFF,
                        reduce=_np.nanmean,
                    )

            scorer = dlctab.columns[0][0]
            pose_estimation_name = f"{view}_video_keypoints"
            keypoints = tuple(set(col[1] for col in dlctab.columns))

            series = []
            # TODO: think over about what names may be appropriate
            node_names = [f"{kpt}" for kpt in keypoints]
            for kpt, node_name in zip(keypoints, node_names):
                if downsample:
                    # FIXME: this block may be removed
                    # when the DeepLabCut model become more efficient
                    data = _validation.validate_keypoint(
                        dlctab, kpt,
                        threshold=downsample_pcutoff,
                    ).apply(_downsample).stack()
                else:
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

            _stdio.message('done.', verbose=verbose)
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
