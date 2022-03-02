from __future__ import annotations
from pathlib import Path
import typing as T
from datetime import datetime
import logging
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

    # %% loop over files / time
    for t in cfg["time"]:
        frame(direc, time=t, var=var, saveplot_fmt=saveplot_fmt, xg=xg, cfg=cfg, plotfun=plotfun)


def frame(
    path: Path,
    time: datetime = None,
    *,
    plotfun: T.Callable = None,
    saveplot_fmt: str = None,
    var: set[str] = None,
    xg: dict[str, T.Any] = None,
    cfg: dict[str, T.Any] = None,
):
    """
    Parameters
    ----------

    path: pathlib.Path
        filename or directory + time to plot
    time: datetime.datetime, optional
        if path is a directory, time is required
    """

    if not var:
        var = PARAMS
        var.add("aurora")

    if not cfg:
        cfg = read.config(path)

    if time is None:
        # read a specific filename
        dat = read.data(path, var)
        path = path.parent
    else:
        dat = read.frame(path, time, var=var)

    if not xg:
        xg = read.grid(path)

    if plotfun is None:
        plotfun = grid2plotfun(xg)

    t0 = to_datetime(dat.time)

    for k, v in dat.items():
        try:
            if any(s in k for s in var):
                if plotfun.__name__.startswith("curv"):
                    fg, ax = plotfun(cfg, xg, v)
                else:
                    fg, ax = plotfun(t0, xg, v.squeeze(), wavelength=dat.get("wavelength"))
                save_fig(fg, path, name=k, fmt=saveplot_fmt, time=t0)
        except ValueError as e:
            logging.error(f"SKIP: plot {k} at {t0} due to {e}")

    if "aurora" in var:
        aurmap_dir = cfg.get("aurmap_dir")
        if not aurmap_dir:
            return

        # handle relative or absolute path to GLOW data
        if not aurmap_dir.is_absolute():
            aurmap_dir = path / cfg["aurmap_dir"]

        glow(aurmap_dir, t0, saveplot_fmt, xg=xg)
