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

from typing import ClassVar
from typing_extensions import Self
from dataclasses import dataclass
from time import time as _now
import warnings as _warnings

import numpy as _np
import numpy.typing as _npt
from scipy.signal import (
    butter as _butter,
    filtfilt as _filtfilt,
)
from sklearn.linear_model import LinearRegression as _LinearRegression
from tqdm import tqdm as _tqdm

import pynwb as _nwb
from pynwb.ophys import (
    DfOverF as _DfOverF,
    ImageSegmentation as _ImageSegmentation,
    PlaneSegmentation as _PlaneSegmentation,
    RoiResponseSeries as _RoiResponseSeries,
)
from hdmf.common import DynamicTable as _DynamicTable

from . import (
    stdio as _stdio,
    file_metadata as _file_metadata,
    imaging as _imaging,
)


@dataclass
class ROISegmentation:
    root: _ImageSegmentation
    B: _ImageSegmentation
    V: _ImageSegmentation
    ENTRIES: ClassVar[dict[str, str]] = {
        'B': 'dFF_B',
        'V': 'dFF_V',
        'dFF': 'dFF',
    }
    CHANNELS: ClassVar[dict[str, str]] = {
        'B': 'dF/F responses (mean over each ROI mask) being calculated from the frames with 473 nm excitation',
        'V': 'dF/F responses (mean over each ROI mask) being calculated from the frames with 405 nm excitation',
        'dFF': 'hemodynamics-corrected dF/F calcium responses',
    }
    FRAMES: ClassVar[dict[str, str]] = {
        'B': 'wide-field calcium imaging, with 473 nm excitation',
        'V': 'wide-field calcium imaging, with 405 nm excitation',
    }
    SEGMENTATION_TYPES: ClassVar[dict[str, dict[str, str]]] = {
        'B': dict(
            name='ROIs',
            description="ROIs based on Allen CCF, estimated using the mesoscaler algorithm, in the frames with 473 nm excitation"
        ),
        'V': dict(
            name='ROIs_V',
            description="ROIs based on Allen CCF, estimated using the mesoscaler algorithm, in the frames with 473 nm excitation"
        ),
    }

    @property
    def dFF(self) -> _PlaneSegmentation:
        return self.B

    @property
    def frames(self) -> dict[str, _PlaneSegmentation]:
        return dict((chan, getattr(self, chan)) for chan in self.FRAMES.keys())

    @property
    def channels(self) -> dict[str, _PlaneSegmentation]:
        return dict((chan, getattr(self, chan)) for chan in self.CHANNELS.keys())

    @property
    def planes(self) -> dict[str, _PlaneSegmentation]:
        return self.frames

    def frame_description(self, frame: str) -> str:
        return self.FRAMES[frame]

    def channel_description(self, channel: str) -> str:
        return self.CHANNELS[channel]

    def channel_entry(self, channel: str) -> str:
        return self.ENTRIES[channel]

    def segmentation_info(self, frame: str) -> dict[str, str]:
        return self.SEGMENTATION_TYPES[frame].copy()


@dataclass
class SingleROISignal:
    metadata: _file_metadata.SingleROIMetadata
    time: _npt.NDArray[_np.floating]
    B: _npt.NDArray[_np.floating]
    V: _npt.NDArray[_np.floating]
    corrected: _npt.NDArray[_np.floating]
    slope: float
    intercept: float

    @property
    def dFF(self) -> _npt.NDArray[_np.floating]:
        return self.corrected


@dataclass
class SignalFilter:
    b: _npt.NDArray[_np.floating]
    a: _npt.NDArray[_np.floating]

    @classmethod
    def bandpass(
        cls,
        bp_range: tuple[float],
        bp_order: int,
        sampling_rate: float,
    ) -> Self:
        b, a = _butter(
            bp_order,
            bp_range,
            btype='bandpass',
            analog=False,
            output='ba',
            fs=sampling_rate
        )
        return cls(b, a)

    def __call__(self, x):
        return _filtfilt(self.b, self.a, x)


@dataclass
class CoefficientEstimation:
    slope: float
    intercept: float
    residuals: _npt.NDArray[_np.floating]

    @classmethod
    def fit(cls, V, B) -> Self:
        V = V.reshape((-1, 1))
        B = B.reshape((-1, 1))
        mod = _LinearRegression(fit_intercept=True).fit(V, B)
        res = B - mod.predict(V)
        return cls(
            slope=float(mod.coef_.ravel()[0]),
            intercept=float(mod.intercept_.ravel()[0]),
            residuals=res.ravel(),
        )


def compute_single_roi_signal(
    roi: _file_metadata.SingleROIMetadata,
    flattened_data: _imaging.ImagingData,
    signal_filter: SignalFilter,
) -> SingleROISignal:

    def _baseline(x):
        return _np.median(x)

    def _as_dFF(x):
        m = _baseline(x)
        return (x - m) / m

    def _half_frame_forward(V):
        interp = (V[1:] + V[:-1]) / 2
        return _np.concatenate([(V[0],), interp])

    mask = roi.mask.ravel()
    B = _as_dFF(flattened_data.B[:, mask].mean(1))
    V = _half_frame_forward(
        _as_dFF(flattened_data.V[:, mask].mean(1))
    )
    B = signal_filter(B)
    V = signal_filter(V)
    corr = CoefficientEstimation.fit(V, B)
    return SingleROISignal(
        metadata=roi,
        time=flattened_data.time,
        B=B,
        V=V,
        corrected=corr.residuals,
        slope=corr.slope,
        intercept=corr.intercept
    )


def compute_roi_signals(
    metadata: _file_metadata.Metadata,
    roimeta: _file_metadata.ROISetMetadata,
    flattened_data: _imaging.ImagingData,
    bp_range: tuple[float] = (0.01, 10),
    bp_order: int = 5,
    verbose: bool = True,
) -> tuple[SingleROISignal]:
    filt = SignalFilter.bandpass(
        bp_order=bp_order,
        bp_range=bp_range,
        sampling_rate=metadata.imaging.planes[1].frame_rate
    )
    rng = roimeta.rois
    if verbose:
        rng = _tqdm(rng, desc='processing rois')
    return tuple(compute_single_roi_signal(
        roi, flattened_data, filt
    ) for roi in rng)


def setup_transformation_entry(transform: _npt.NDArray) -> _DynamicTable:
    #
    # IMPLEMENTATION NOTE (KS)
    #
    # If it were a Pandas DataFrame, I would form a single table for a single matrix,
    # with each cell containing only a single value. This may sound weird, but
    # it would have a benefit of obtaining a matrix by the `to_numpy()` method.
    #
    # But the pyNWB DynamicTable implementation does not allow non-integer type
    # table indices, so the resulting table appears 'asymmetric', i.e. does not
    # look like containing a single matrix.
    #
    # So here we take the approach to include the whole matrix in a single cell
    # (although the use of a "single-cell table" appears to be an overcomplication).
    #
    tab = _DynamicTable(
        name="atlas_to_data_transform",
        description="affine transformation matrix from the 512x512 Allen CCF atlas to the 288x288 imaging data coordinates",
    )
    tab.add_column(
        name="affine_matrix",
        description="affine transformation matrix",
    )
    tab.add_row({"affine_matrix": transform})
    return tab


def setup_roi_segmentation(
    setup: _imaging.NWBImagingSetup,
    roimeta: _file_metadata.ROISetMetadata,
) -> ROISegmentation:
    segroot = _ImageSegmentation()
    # prepare planes (and the buffer for signals)
    planes = dict()
    for typ in ROISegmentation.FRAMES.keys():
        planes[typ] = segroot.create_plane_segmentation(
            imaging_plane=getattr(setup, typ),
            **(ROISegmentation.SEGMENTATION_TYPES[typ]),
        )
        planes[typ].add_column(name='roi_name', description='name of the ROIs', data=[])
        planes[typ].add_column(name='roi_description', description='description of the ROIs', data=[])
    return ROISegmentation(root=segroot, **planes)


def setup_roisignals_entry(
    roisigs: tuple[SingleROISignal],
    seg: ROISegmentation,
    verbose: bool = True,
) -> _DfOverF:
    _stdio.message('registering ROIs...', end=' ', verbose=verbose)
    start = _now()
    timebases = roisigs[0].time
    signals = dict()
    for typ in seg.channels.keys():
        signals[typ] = []

    with _warnings.catch_warnings():
        # NOTE: just to suppress known (probably harmless) warnings
        _warnings.filterwarnings(
            'ignore',
            category=UserWarning,
            module='hdmf'
        )

        # register rois (and collect signals)
        for roi in roisigs:
            for typ, pln in seg.frames.items():
                pln.add_roi(
                    roi_name=roi.metadata.name,
                    roi_description=roi.metadata.description,
                    image_mask=roi.metadata.mask,
                )
            for typ in seg.channels.keys():
                signals[typ].append(getattr(roi, typ).ravel())

        # register FOVs (i.e. B and V channel frames)
        FOVs = dict()
        for frame, pln in seg.frames.items():
            FOVs[frame] = pln.create_roi_table_region(
                region=[idx for idx in range(len(roisigs))],
                description=seg.frame_description(frame),
            )
        FOVs['dFF'] = FOVs['B']  # FIXME: ad hoc addition

        # register signals
        dff = _DfOverF()
        for typ, FOV in FOVs.items():
            sigs = _RoiResponseSeries(
                name=seg.channel_entry(typ),
                description=seg.channel_description(typ),
                data=_np.stack(signals[typ], axis=1),
                unit="a.u.",
                rois=FOV,
                timestamps=getattr(timebases, typ),
            )
            dff.add_roi_response_series(sigs)

    stop = _now()
    _stdio.message(f"done (took {(stop - start):.1f} sec).", verbose=verbose)
    return dff


def write_roi_entries(
    nwbfile: _nwb.NWBFile,
    metadata: _file_metadata.Metadata,
    roimeta: _file_metadata.ROISetMetadata,
    flattened_data: _imaging.ImagingData,
    setup: _imaging.NWBImagingSetup,
    verbose: bool = True,
) -> _nwb.ProcessingModule:
    roisigs = compute_roi_signals(
        metadata=metadata,
        roimeta=roimeta,
        flattened_data=flattened_data,
        verbose=verbose,
    )
    seg = setup_roi_segmentation(setup, roimeta)
    dff = setup_roisignals_entry(roisigs, seg, verbose=verbose)
    mod = nwbfile.create_processing_module(
        name="ophys", description="optical physiology processed data"
    )
    mod.add(seg.root)
    mod.add(dff)
    #
    # FIXME
    # would it be possible to add transformation entry into the ophys module?
    #
    tab = setup_transformation_entry(roimeta.transform)
    nwbfile.add_analysis(tab)
    return mod
