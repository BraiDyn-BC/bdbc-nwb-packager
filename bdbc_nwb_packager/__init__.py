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

from importlib import reload as _reload

from . import (
    stdio,
    paths,
    metadata,
    trials,
    validation,
    alignment,
    packaging,
    batch,
)

_reload(stdio)
_reload(paths)
_reload(metadata)
_reload(trials)
_reload(validation)
_reload(alignment)
_reload(packaging)
_reload(batch)

PathSettings = paths.PathSettings
Metadata = metadata.Metadata
ROISetMetadata = metadata.ROISetMetadata
ImagingData = packaging.ImagingData
SingleROISignal = packaging.SingleROISignal

sessions_root_dir = paths.sessions_root_dir
setup_path_settings = paths.setup_path_settings
metadata_from_rawdata = metadata.metadata_from_rawdata
read_roi_metadata = metadata.read_roi_metadata
package_nwb = packaging.package_nwb

run_batch = batch.run_batch

