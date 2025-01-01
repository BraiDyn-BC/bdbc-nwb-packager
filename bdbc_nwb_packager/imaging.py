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
from typing import ClassVar, Optional
from typing_extensions import Self
from pathlib import Path
from dataclasses import dataclass
from time import time as _now

import numpy as _np
import numpy.typing as _npt
import h5py as _h5
import pynwb as _nwb
from tifffile import TiffWriter as _TiffWriter
from tqdm import tqdm as _tqdm

from .types import PathLike
from . import (
    logging as _logging,
    configure as _configure,
    file_metadata as _file_metadata,
    timebases as _timebases,
)


@dataclass
class ImagingData:
    time: _npt.NDArray[_np.floating]
    B: Optional[_npt.NDArray[_np.floating]]
    V: Optional[_npt.NDArray[_np.floating]]
    CHANNELS: ClassVar[tuple[str]] = ('B', 'V')

    def has_data(self) -> bool:
        return all((getattr(self, ch) is not None) for ch in self.CHANNELS)

    def flatten(self, verbose: bool = True) -> Self:
        if self.B.ndim == 2:
            return self
        data = dict(time=self.time)
        start = _now()
        for fld in self.CHANNELS:
            _logging.info(f"flattening {fld} frames...")
            frames = getattr(self, fld)
            data[fld] = frames.reshape((frames.shape[0], -1))
        stop = _now()
        _logging.info(f"done flattening frames (took {(stop - start) / 60:.1f} min).")
        return self.__class__(**data)


@dataclass
class NWBImagingSetup:
    device: object  # TODO
    acquisition: object  # TODO
    B: object  # TODO
    V: object  # TODO


def load_imaging_data(
    rawfile: PathLike,
    timebases: _timebases.Timebases,
    read_frames: bool = True,
    verbose: bool = True
) -> ImagingData:
    if read_frames:
        with _h5.File(rawfile, 'r') as src:
            start = _now()
            _logging.info("reading B frames...")
            im_B = _np.array(src["image/Ib"], dtype=_np.float32).transpose((0, 2, 1))  # (T, H, W)
            _logging.info("reading V frames...")
            im_V = _np.array(src["image/Iv"], dtype=_np.float32).transpose((0, 2, 1))
            stop = _now()
            _logging.info(f"done reading imaging data (took {(stop - start) / 60:.1f} min).")
    else:
        im_B = None,
        im_V = None,
    return ImagingData(time=timebases, B=im_B, V=im_V)


def setup_imaging_device(
    metadata: _file_metadata.Metadata,
    nwbfile: _nwb.NWBFile,
    verbose: bool = True,
) -> NWBImagingSetup:
    device = nwbfile.create_device(
        name=metadata.imaging.device.name,
        description=metadata.imaging.device.description,
        manufacturer=metadata.imaging.device.manufacturer,
    )
    acq = _nwb.ophys.OpticalChannel(
        name='LongpassFilter',
        description='535 nm long-pass filtered fluorescence',
        emission_lambda=metadata.imaging.B.emission,
    )

    # blue channel
    pln_B = nwbfile.create_imaging_plane(
        name="ImagingPlane_blue",
        optical_channel=acq,
        imaging_rate=metadata.imaging.B.frame_rate,
        description=metadata.imaging.B.description,
        device=device,
        excitation_lambda=metadata.imaging.B.excitation,
        indicator=metadata.imaging.indicator,
        location=metadata.imaging.location,
        grid_spacing=metadata.imaging.B.pixel_size,
        grid_spacing_unit="micrometers",
        origin_coords=[0.0, 0.0],
        origin_coords_unit="meters",
    )

    # UV channel
    pln_V = nwbfile.create_imaging_plane(
        name="ImagingPlane_UV",
        optical_channel=acq,
        imaging_rate=metadata.imaging.V.frame_rate,
        description=metadata.imaging.V.description,
        device=device,
        excitation_lambda=metadata.imaging.V.excitation,
        indicator=metadata.imaging.indicator,
        location=metadata.imaging.location,
        grid_spacing=metadata.imaging.V.pixel_size,
        grid_spacing_unit="micrometers",
        origin_coords=[0.0, 0.0],
        origin_coords_unit="meters",
    )
    _logging.info("done configuring the imaging setup.")
    return NWBImagingSetup(
        device=device,
        acquisition=acq,
        B=pln_B,
        V=pln_V,
    )


def write_imaging_data(
    nwbfile: _nwb.NWBFile,
    destination: _configure.DestinationPaths,
    frames: ImagingData,
    setup: NWBImagingSetup,
    write_frames: bool = True,
    verbose: bool = True,
):
    outfiles = destination.imaging
    if write_frames:
        for chan in frames.CHANNELS:
            _logging.debug(f"writing {chan} frames...")
            start = _now()
            outfile = Path(getattr(outfiles, chan))
            if not outfile.parent.exists():
                outfile.parent.mkdir(parents=True)
            data = getattr(frames, chan)
            with _TiffWriter(str(outfile), bigtiff=True) as out:
                rng = range(data.shape[0])
                if verbose:
                    rng = _tqdm(rng, desc=f"writing {chan} frames")
                for i in rng:
                    out.write(data[i], contiguous=True)
            stop = _now()
            _logging.debug(f"done writing {chan} frames (took {(stop - start):.1f} sec)")
    else:
        _logging.info('skip writing imaging frames')

    relfiles = outfiles.relative_to(destination.session_dir)
    _logging.debug("adding channels to registry...")
    start = _now()
    sig_B = _nwb.ophys.OnePhotonSeries(
        name='widefield_blue',
        description='widefield imaging data, blue excitation',
        imaging_plane=setup.B,
        unit="n.a.",
        external_file=[str(relfiles.B)],
        format="external",
        starting_frame=[0],
        timestamps=frames.time.B,
    )
    sig_V = _nwb.ophys.OnePhotonSeries(
        name='widefield_UV',
        description='widefield imaging data, UV excitation',
        imaging_plane=setup.V,
        unit="n.a.",
        external_file=[str(relfiles.V)],
        format="external",
        starting_frame=[0],
        timestamps=frames.time.V,
    )
    nwbfile.add_acquisition(sig_B)
    nwbfile.add_acquisition(sig_V)
    stop = _now()
    _logging.debug(f"done registering channels to the NWB file (took {(stop - start):.1f} sec).")
