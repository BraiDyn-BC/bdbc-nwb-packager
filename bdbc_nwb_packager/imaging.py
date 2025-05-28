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
from typing import ClassVar, Optional, Callable
from typing_extensions import Self
from pathlib import Path
from dataclasses import dataclass
from time import time as _now

import numpy as _np
import numpy.typing as _npt
import h5py as _h5
import pynwb as _nwb
from hdmf.backends.hdf5.h5_utils import H5DataIO as _H5DataIO
from tifffile import TiffWriter as _TiffWriter
from skvideo.io import FFmpegWriter as _FFmpegWriter
from tqdm import tqdm as _tqdm

from .types import PathLike
from . import (
    logging as _logging,
    configure as _configure,
    file_metadata as _file_metadata,
    timebases as _timebases,
)


def as_uint16(frame):
    return _np.clip(frame, 0, 65535).round().astype(_np.uint16)


@dataclass
class ImagingData:
    time: _timebases.Timebases
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
        name="sCMOS_camera",
        model_number=metadata.imaging.device.name,
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
    compression: Optional[str] = 'gzip',
    compression_level: Optional[int] = 4,
    verbose: bool = True,
):
    outfiles = destination.imaging
    if outfiles.format == 'internal':
        write_frames_internal(
            nwbfile=nwbfile,
            frames=frames,
            setup=setup,
            write_frames=write_frames,
            compression=compression,
            compression_level=compression_level,
            verbose=verbose,
        )
    elif outfiles.format == 'tiff':
        write_frames_external(
            outfiles=outfiles,
            frames=frames,
            write_fn=write_tiff,
            write_frames=write_frames,
            compression=compression,
            compression_level=compression_level,
            verbose=verbose,
        )
        register_channels_external(
            nwbfile=nwbfile,
            setup=setup,
            relfiles=outfiles.relative_to(destination.session_dir),
            timestamps=frames.time,
        )
    elif outfiles.format == 'avi':
        write_frames_external(
            outfiles=outfiles,
            frames=frames,
            write_fn=write_avi,
            write_frames=write_frames,
            compression=compression,
            compression_level=compression_level,
            verbose=verbose,
        )
        register_channels_external(
            nwbfile=nwbfile,
            setup=setup,
            relfiles=outfiles.relative_to(destination.session_dir),
            timestamps=frames.time,
        )
    else:
        raise ValueError(f"unexpected format: {outfiles.format}")


def write_frames_internal(
    nwbfile: _nwb.NWBFile,
    frames: ImagingData,
    setup: NWBImagingSetup,
    write_frames: bool = True,
    compression: Optional[str] = 'gzip',
    compression_level: Optional[int] = 6,
    frame_chunk: int = 1,
    verbose: bool = True,
):
    if compression is None:
        _logging.info("override compression algorithm to 'gzip'")
        compression = 'gzip'
    if compression_level is None:
        compression_level = 4

    _logging.info(f"writing channels to the NWB file (compression: '{compression} ({compression_level})')...")
    start = _now()
    _, W, H = frames.B.shape

    if write_frames == True:
        def _format(data):
            return _H5DataIO(
                data=as_uint16(data),
                chunks=(frame_chunk, W, H),
                compression=compression,
                compression_opts=compression_level,
            )
    else:
        def _format(data):
            return _np.zeros((W, H), dtype=_np.uint16)
        _logging.info("skip writing imaging data")

    _logging.debug("B channel...")
    sig_B = _nwb.ophys.OnePhotonSeries(
        name='widefield_blue',
        description='motion-corrected widefield imaging data, blue excitation',
        imaging_plane=setup.B,
        unit="n.a.",
        data=_format(frames.B),
        starting_frame=[0],
        timestamps=frames.time.B,
    )
    nwbfile.add_acquisition(sig_B)
    _logging.debug("V channel...")
    sig_V = _nwb.ophys.OnePhotonSeries(
        name='widefield_UV',
        description='motion-corrected widefield imaging data, UV excitation',
        imaging_plane=setup.V,
        unit="n.a.",
        data=_format(frames.V),
        starting_frame=[0],
        timestamps=frames.time.V,
    )
    nwbfile.add_acquisition(sig_V)
    stop = _now()
    _logging.info(f"done registering channels to the NWB file (took {(stop - start) / 60:.1f} min).")


def write_frames_external(
    outfiles: _configure.ImagingDataFiles,
    frames: ImagingData,
    write_fn: Callable,
    write_frames: bool = True,
    compression: Optional[str] = 'gzip',
    compression_level: Optional[int] = 4,
    verbose: bool = True,
):
    if write_frames:
        for chan in frames.CHANNELS:
            _logging.debug(f"writing {chan} frames...")
            start = _now()
            outfile = Path(getattr(outfiles, chan))
            if not outfile.parent.exists():
                outfile.parent.mkdir(parents=True)
            data = getattr(frames, chan)
            write_fn(
                chan=chan,
                data=data,
                outfile=outfile,
                compression=compression,
                compression_level=compression_level,
                verbose=verbose,
            )
            stop = _now()
            _logging.debug(f"done writing {chan} frames (took {(stop - start):.1f} sec)")
    else:
        _logging.info('skip writing imaging frames')


def register_channels_external(
    nwbfile: _nwb.NWBFile,
    setup: NWBImagingSetup,
    relfiles: _configure.ImagingDataFiles,
    timestamps: _timebases.Timebases,
):
    _logging.debug("adding channels to registry...")
    start = _now()
    sig_B = _nwb.ophys.OnePhotonSeries(
        name='widefield_blue',
        description='motion-corrected widefield imaging data, blue excitation',
        imaging_plane=setup.B,
        unit="n.a.",
        external_file=[str(relfiles.B)],
        format="external",
        starting_frame=[0],
        timestamps=timestamps.B,
    )
    sig_V = _nwb.ophys.OnePhotonSeries(
        name='widefield_UV',
        description='motion-corrected widefield imaging data, UV excitation',
        imaging_plane=setup.V,
        unit="n.a.",
        external_file=[str(relfiles.V)],
        format="external",
        starting_frame=[0],
        timestamps=timestamps.V,
    )
    nwbfile.add_acquisition(sig_B)
    nwbfile.add_acquisition(sig_V)
    stop = _now()
    _logging.debug(f"done registering channels to the NWB file (took {(stop - start):.1f} sec).")


def write_internal(
    chan: str,
    data: _npt.NDArray,
    outfile: Path,
    compression: str,
    compression_level: Optional[int],
    verbose: bool,
):
    raise NotImplementedError()


def write_tiff(
    chan: str,
    data: _npt.NDArray,
    outfile: Path,
    compression: str,
    compression_level: Optional[int],
    verbose: bool,
):
    assert outfile.suffix == '.tiff'
    if compression == 'gzip':
        _logging.info("overriding compression algorithm: gzip -> deflate")
        compression = 'deflate'
    with _TiffWriter(str(outfile), bigtiff=True) as out:
        rng = range(data.shape[0])
        if verbose:
            rng = _tqdm(
                rng,
                desc=f"writing {chan} frames",
                mininterval=1.0,
                smoothing=0.001,
            )
        for i in rng:
            out.write(
                as_uint16(data[i]),
                photometric='minisblack',
                compression=compression,
                compressionargs=dict(level=compression_level),
                metadata=None,
            )


def write_avi(
    chan: str,
    data: _npt.NDArray,
    outfile: Path,
    verbose: bool,
    **_
):
    odict = {
        '-vcodec': 'rawvideo',
        '-pix_fmt': 'gray16le',
        '-r': '30',
    }
    assert outfile.suffix == '.avi'
    with _FFmpegWriter(str(outfile), outputdict=odict) as out:
        rng = range(data.shape[0])
        if verbose:
            rng = _tqdm(
                rng,
                desc=f"writing {chan} frames",
                mininterval=1.0,
                smoothing=0.001,
            )
        for i in rng:
            out.writeFrame(
                as_uint16(data[i]),
            )
