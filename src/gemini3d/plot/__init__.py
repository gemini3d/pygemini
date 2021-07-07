from __future__ import annotations
from pathlib import Path
import typing as T
from datetime import datetime

import numpy as np
import matplotlib as mpl

from .. import read
from ..utils import to_datetime
from .core import save_fig
from .glow import glow
from .constants import PARAMS
from . import cartesian
from . import curvilinear

mpl.rcParams["axes.formatter.limits"] = (-3, 4)
mpl.rcParams["axes.formatter.useoffset"] = False
mpl.rcParams["axes.formatter.min_exponent"] = 4


def grid2plotfun(xg: dict[str, np.ndarray]) -> T.Callable:
    plotfun = None
    h1 = xg.get("h1")

    lxs = read.get_lxs(xg)

    if h1 is not None:
        minh1 = h1.min()
        maxh1 = h1.max()
        if (abs(minh1 - 1) > 1e-4) or (abs(maxh1 - 1) > 1e-4):  # curvilinear grid
            if (lxs[1] > 1) and (lxs[2] > 1):
                plotfun = curvilinear.curv3d_long  # type: ignore
            else:
                plotfun = curvilinear.curv2d  # type: ignore

    if plotfun is None:
        if (lxs[1] > 1) and (lxs[2] > 1):
            plotfun = cartesian.cart3d_long_ENU  # type: ignore
        else:
            plotfun = cartesian.cart2d  # type: ignore

    return plotfun  # type: ignore


def plot_all(direc: Path, var: set[str] = None, saveplot_fmt: str = ""):

    direc = Path(direc).expanduser().resolve(strict=True)

    if not var:
        var = PARAMS

    if {"png", "pdf", "eps"} & var:
        raise ValueError("please use saveplot_fmt='png' or similar for plot format")

    xg = read.grid(direc)
    plotfun = grid2plotfun(xg)
    cfg = read.config(direc)

    aurmap_dir = None
    if cfg.get("aurmap_dir"):
        # handle relative or absolute path to GLOW data
        if cfg["aurmap_dir"].is_absolute():
            aurmap_dir = cfg["aurmap_dir"]
        else:
            aurmap_dir = direc / cfg["aurmap_dir"]

    # %% loop over files / time
    for t in cfg["time"]:
        frame(direc, time=t, var=var, saveplot_fmt=saveplot_fmt, xg=xg, cfg=cfg, plotfun=plotfun)
        glow(aurmap_dir, t, saveplot_fmt, xg=xg)


def frame(
    direc: Path,
    time: datetime = None,
    *,
    plotfun: T.Callable = None,
    saveplot_fmt: str = None,
    var: set[str] = None,
    xg: dict[str, T.Any] = None,
    cfg: dict[str, T.Any] = None,
):
    """
    if save_dir, plots will not be visible while generating to speed plot writing
    """

    if not var:
        var = PARAMS

    if time is None:
        dat = read.data(direc, var)
        direc = direc.parent
    else:
        dat = read.frame(direc, time, var=var)

    if not xg:
        xg = read.grid(direc)

    if plotfun is None:
        plotfun = grid2plotfun(xg)

    for k, v in dat.items():
        if k == "Phitop":
            continue
            # FIXME: fix the reason why this plotting sometimes fails
        if any(s in k for s in var):
            if plotfun.__name__.startswith("curv"):
                fg = plotfun(cfg, xg, v)
            else:
                fg = plotfun(
                    to_datetime(dat.time), xg, v.squeeze(), wavelength=dat.get("wavelength")
                )
            save_fig(fg, direc, name=k, fmt=saveplot_fmt, time=time)
