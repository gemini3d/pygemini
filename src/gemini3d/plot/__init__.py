"""
gemini3d.plot: functions to plot Gemini3D output

Command line usage:

    python -m gemini3d.plot path/to/data
"""

from __future__ import annotations
from pathlib import Path
import typing as T
from datetime import datetime
import logging
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


def grid2plotfun(xg: dict[str, T.Any]) -> T.Callable:
    h1 = xg.get("h1")

    lxs = read.get_lxs(xg)

    if h1 is not None:
        minh1 = h1.min()
        maxh1 = h1.max()
        if (abs(minh1 - 1) > 1e-4) or (abs(maxh1 - 1) > 1e-4):  # curvilinear grid
            if (lxs[1] > 1) and (lxs[2] > 1):
                return curvilinear.curv3d_long  # type: ignore
            else:
                return curvilinear.curv2d  # type: ignore

    if (lxs[1] > 1) and (lxs[2] > 1):
        return cartesian.cart3d_long_ENU  # type: ignore

    return cartesian.cart2d  # type: ignore


def plot_all(direc: Path, var: set[str] | None = None, saveplot_fmt: str | None = None):
    """
    gemini3d.plot.plot_all(direc, var=None, saveplot_fmt="")

    Parameters
    ---------

    direc: pathlib.Path
        directory of simulation output to plot
    var: set of str, optional
        variables to plot, default is all
    saveplot_fmt: str, optional
        format to save plots, default is n
    """
    direc = Path(direc).expanduser().resolve(strict=True)

    if not var:
        var = PARAMS

    xg = read.grid(direc)
    plotfun = grid2plotfun(xg)
    cfg = read.config(direc)

    #    fg = mpl.figure.Figure(constrained_layout=True)
    fg = mpl.figure.Figure(constrained_layout=True, dpi=150, figsize=(18, 4.5))

    # %% loop over files / time
    for t in cfg["time"]:
        fg.clf()
        frame(
            fg,
            direc,
            time=t,
            var=var,
            saveplot_fmt=saveplot_fmt,
            xg=xg,
            cfg=cfg,
            plotfun=plotfun,
        )


def frame(
    fg: mpl.figure.Figure,
    path: Path,
    time: datetime | None = None,
    *,
    plotfun: T.Callable | None = None,
    saveplot_fmt: str | None = None,
    var: set[str] | None = None,
    xg: dict[str, T.Any] | None = None,
    cfg: dict[str, T.Any] | None = None,
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
        var = var.union({"aurora", "Phitop"})

    if not cfg:
        cfg = read.config(path)

    if time is None:
        # read a specific filename
        dat = read.frame(path, var=var)
        path = path.parent
    else:
        dat = read.frame(path, time, var=var)

    if not xg:
        xg = read.grid(path)

    if plotfun is None:
        plotfun = grid2plotfun(xg)

    t0 = to_datetime(dat.time)

    for k in var.intersection(dat.data_vars):
        fg.clf()
        try:
            if plotfun.__name__.startswith("curv"):
                plotfun(fg, t0, xg, dat[k], cfg)
            else:
                plotfun(fg, t0, xg, dat[k].squeeze(), wavelength=dat.get("wavelength"))
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

        glow(aurmap_dir, t0, fg=fg, saveplot_fmt=saveplot_fmt, xg=xg)
