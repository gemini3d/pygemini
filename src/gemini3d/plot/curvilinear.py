from __future__ import annotations
from datetime import datetime

import numpy as np
import xarray
from matplotlib.figure import Figure


def curv3d_long(
    time: datetime,
    grid: dict[str, np.ndarray],
    parm: xarray.DataArray,
    name: str,
    fg: Figure = None,
    **kwargs,
):
    raise NotImplementedError


def curv2d(
    time: datetime,
    grid: dict[str, np.ndarray],
    parm: xarray.DataArray,
    name: str,
    fg: Figure = None,
    **kwargs,
):
    raise NotImplementedError
