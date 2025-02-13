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
"""ROI-related metadata structures."""
from dataclasses import dataclass

import numpy.typing as _npt
import pandas as _pd


@dataclass
class SingleROIMetadata:
    name: str
    mask: _npt.NDArray[bool]
    description: str

    def __repr__(self) -> str:
        return f"SingleROIMetadata(name='{self.name}', ...)"


@dataclass
class ROISetMetadata:
    transform: _npt.NDArray
    rois: tuple[SingleROIMetadata]

    def transform_as_table(self) -> _pd.DataFrame:
        tab = _pd.DataFrame(data={
            'x_in': self.transform[:, 0],
            'y_in': self.transform[:, 1],
            'c_in': self.transform[:, 2],
        })
        tab.index = ['x_out', 'y_out']
        return tab
