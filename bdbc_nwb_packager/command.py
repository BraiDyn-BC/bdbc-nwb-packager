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

from argparse import ArgumentParser as _ArgumentParser

from . import batch as _batch

parser = _ArgumentParser(
    prog='package-nwb',
    description='batch-packages BraiDyn-BC sessions into NWB files',
    epilog='note that all the default directory settings are assumed to be in the environment variables.'
)
parser.add_argument(
    '-A', '--animal',
    default=None,
    help='specifies the animal (or animals, in a comma-separated list) to be processed'
)
parser.add_argument(
    '-E', '--fromdate',
    default=None,
    help='specifies the _e_arliest session date to be processed, in the YYMMDD format'
)
parser.add_argument(
    '-L', '--todate',
    default=None,
    help='specifies the _l_atest session date to be processed, in the YYMMDD format'
)
parser.add_argument(
    '-t', '--type',
    default=None,
    help="specifies the type (or types, in a comma-separated list) of the sessions to be processed: must be one of ('task', 'rest', 'ss')"
)
parser.add_argument(
    '--no-write-videos',
    action='store_false',
    dest='copy_videos',
    help='suppresses copying of videos to the publication directory',
)
parser.add_argument(
    '--no-write-imaging',
    action='store_false',
    dest='write_imaging_frames',
    help='suppresses writing out imaging frames to the publication directory',
)
parser.add_argument(
    '--no-write-rois',
    action='store_false',
    dest='register_rois',
    help='suppresses registration and calculation of ROI dF/F',
)
parser.add_argument(
    '--no-downsampled-data',
    action='store_false',
    dest='add_downsampled',
    help='suppresses generation and registration of downsampled data',
)
parser.add_argument(
    '-m' '--metadata',
    default=None,
    dest='override_metadata',
    help="comma-separated metadata entries, in a 'field=value' format, to override entries in the raw-data files",
)
parser.add_argument(
    '-f', '--force',
    action='store_true',
    dest='overwrite',
    help="ignores and overwrites any existing output file(s)"
)
parser.add_argument(
    '-q', '--quiet',
    action='store_false',
    dest='verbose',
    help="suppresses the verbose output (which is the default mode of this command)"
)


def batch_package_nwb(args=None):
    if args is None:
        specs = vars(parser.parse_args())
    else:
        specs = vars(parser.parse_args(args))
    _batch.run_batch(**specs)
