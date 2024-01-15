from __future__ import annotations
from pathlib import Path
import typing as T
import logging

import matplotlib as mpl

from . import grid2plotfun
from .core import save_fig

from .. import find, read
from ..efield import get_times as efield_times
from ..particles import get_times as precip_times
from ..utils import to_datetime


def plot_all(
    direc: Path,
    var: set[str] | None = None,
    saveplot_fmt: str | None = None,
    xg: dict[str, T.Any] | None = None,
) -> None:
    """
    plot simulation inputs, under "direc/inputs"

    if save_dir defined, plots will not be visible while generating to speed plot writing
    """

    fg = mpl.figure.Figure(figsize=mpl.figure.figaspect(1 / 4), tight_layout=True)
    # tight_layout works better with suptitle

    direc = Path(direc).expanduser().resolve(strict=True)

    if not var:
        var = {"ns", "Ts", "vs1"}

    cfg = read.config(direc)
    init_file = direc / cfg["indat_file"]

    if not xg:
        xg = read.grid(direc)

    dat = read.frame(init_file, var={"ns", "Ts", "vs1"})

    if not dat:
        raise ValueError(f"No data in {init_file}")

    plotfun = grid2plotfun(xg)

    time = to_datetime(dat.time)

    for k in var.intersection(dat.data_vars):
        fg.clf()
        # FIXME: for now we just look at electrons dat[k][-1, ...]
        cmap_name = {"ns": "ne", "Ts": "Te", "vs1": "v1"}

        plotfun(
            fg,
            time=time,
            xg=xg,
            parm=dat[k][-1, :, :, :].squeeze(),
            name=cmap_name[k],
            wavelength=dat.get("wavelength"),
        )
        save_fig(fg, direc, name=k, fmt=saveplot_fmt, time=time)


def Efield(direc: Path) -> None:
    """plot input E-field

    Parameters
    ----------

    direc: pathlib.Path
        top-level simulation directory
    """

    fg = mpl.figure.Figure()

    direc = Path(direc).expanduser()

    cfg = read.config(direc)
    path = find.inputs(direc, cfg.get("E0dir"))

    for t in efield_times(cfg):
        try:
            file = find.frame(path, t)
        except FileNotFoundError:
            logging.error(f"no E-field data found at {t} in {path}")
            continue

        dat = read.Efield(file)

        for k in {"Exit", "Eyit", "Vminx1it", "Vmaxx1it", "Vminx2ist", "Vmaxx2ist"}:
            fg.clf()
            ax = fg.gca()
            if dat[k].ndim == 1:
                plot2d_input(ax, dat[k], cfg)
            else:
                plot3d_input(ax, dat[k])

            ax.set_title(f"{k}: {t}")

            save_fig(fg, direc, name=f"Efield-{k}", time=t)


def precip(direc: Path) -> None:
    """
    plot input precipitation

    Parameters
    ----------

    direc: pathlib.Path
        top-level simulation directory
    """

    fg = mpl.figure.Figure(tight_layout=True)

    direc = Path(direc).expanduser()

    cfg = read.config(direc)
    precip_path = find.inputs(direc, cfg.get("precdir"))

    for t in precip_times(cfg):
        try:
            file = find.frame(precip_path, t)
        except FileNotFoundError:
            logging.error(f"no precipitation data found at {t} in {precip_path}")
            continue

        dat = read.precip(file)

        for k in {"E0", "Q"}:
            fg.clf()
            ax = fg.gca()
            if dat[k].ndim == 1:
                plot2d_input(ax, dat[k], cfg)
            else:
                plot3d_input(ax, dat[k])

            ax.set_title(f"{k}: {t}")

            save_fig(fg, direc, name=f"precip-{k}", time=t)


def plot2d_input(ax: mpl.axes.Axes, A, cfg: dict[str, T.Any]) -> None:
    if cfg["lyp"] == 1:
        x = A["mlon"]
        ax.set_xlabel("magnetic longitude")
    else:
        x = A["mlat"]
        ax.set_xlabel("magnetic latitude")

    ax.plot(x, A)


def plot3d_input(ax: mpl.axes.Axes, A) -> None:
    h0 = ax.pcolormesh(A["mlon"], A["mlat"], A, shading="nearest")
    ax.get_figure().colorbar(h0, ax=ax)  # type: ignore
    ax.set_ylabel("magnetic latitude")
    ax.set_xlabel("magnetic longitude")
