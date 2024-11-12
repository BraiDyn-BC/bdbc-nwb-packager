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
"""configuration of file paths related to processing of a single session."""

from importlib import reload as _reload  # DEBUG

from . import (
    source,
    target,
    session,
)

_reload(source)  # DEBUG
_reload(target)  # DEBUG
_reload(session)  # DEBUG

SourceVideoFile = source.SourceVideoFile
SourceVideoFiles = source.SourceVideoFiles
DLCResultFiles = source.DLCResultFiles
SourcePaths = source.SourcePaths

ImagingDataFiles = target.ImagingDataFiles
DestinationVideoFiles = target.DestinationVideoFiles
DestinationPaths = target.DestinationPaths

DLCModelConfigs = session.DLCModelConfigs
PathSettings = session.PathSettings

setup_source_paths = source.setup_source_paths
setup_destination_paths = target.setup_destination_paths
setup_path_settings = session.setup_path_settings
