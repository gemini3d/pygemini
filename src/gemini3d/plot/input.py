from __future__ import annotations
from pathlib import Path
import typing as T
import logging

from matplotlib.figure import Figure

from . import grid2plotfun
from .core import save_fig

from .. import find, read
from ..utils import to_datetime
from ..config import datetime_range


def plot_all(
    direc: Path, var: set[str] = None, saveplot_fmt: str = None, xg: dict[str, T.Any] = None
):
    """
    plot simulation inputs, under "direc/inputs"

    if save_dir defined, plots will not be visible while generating to speed plot writing
    """

    direc = Path(direc).expanduser().resolve(strict=True)

    if not var:
        var = {"ns", "Ts", "vs1"}

    cfg = read.config(direc)
    init_file = direc / cfg["indat_file"]

    if not xg:
        xg = read.grid(direc)

    dat = read.data(init_file, var={"ns", "Ts", "vs1"})

    if not dat:
        raise ValueError(f"No data in {init_file}")

    plotfun = grid2plotfun(xg)

    for k, v in dat.items():
        if any(s in k for s in var):
            # FIXME: for now we just look at electrons v[-1, ...]
            cmap_name = {"ns": "ne", "Ts": "Te", "vs1": "v1"}
            fg = plotfun(
                to_datetime(dat.time),
                xg,
                v[-1, :, :, :].squeeze(),
                cmap_name[k],
                wavelength=dat.get("wavelength"),
            )
            save_fig(fg, direc, name=k, fmt=saveplot_fmt)


def Efield(direc: Path):
    """plot input E-field

    Parameters
    ----------

    direc: pathlib.Path
        top-level simulation directory
    """

    direc = Path(direc).expanduser()

    cfg = read.config(direc)
    path = find.inputs(direc, cfg.get("E0dir"))

    time = datetime_range(cfg["time"][0], cfg["time"][0] + cfg["tdur"], cfg["dtE0"])
    for t in time:
        try:
            file = find.frame(path, t)
        except FileNotFoundError:
            logging.error(f"no E-field data found at {t} in {path}")
            continue

        dat = read.Efield(file)

        for k in {"Exit", "Eyit", "Vminx1it", "Vmaxx1it", "Vminx2ist", "Vmaxx2ist"}:
            if dat[k].ndim == 1:
                fg = plot2d_input(dat[k], cfg)
            else:
                fg = plot3d_input(dat[k], cfg)

            fg.suptitle(f"{k}: {t}")

            save_fig(fg, direc, name=f"Efield-{k}", time=t)


def precip(direc: Path):
    """plot input precipitation

    Parameters
    ----------

    direc: pathlib.Path
        top-level simulation directory
    """

    direc = Path(direc).expanduser()

    cfg = read.config(direc)
    precip_path = find.inputs(direc, cfg.get("precdir"))

    time = datetime_range(cfg["time"][0], cfg["time"][0] + cfg["tdur"], cfg["dtprec"])

    for t in time:
        try:
            file = find.frame(precip_path, t)
        except FileNotFoundError:
            logging.error(f"no precipitation data found at {t} in {precip_path}")
            continue

        dat = read.precip(file)

        for k in {"E0", "Q"}:
            if dat[k].ndim == 1:
                fg = plot2d_input(dat[k], cfg)
            else:
                fg = plot3d_input(dat[k], cfg)

            fg.suptitle(f"{k}: {t}")

            save_fig(fg, direc, name=f"precip-{k}", time=t)


def plot2d_input(A, cfg: dict[str, T.Any]) -> Figure:
    fg = Figure()
    ax = fg.gca()

    if cfg["lyp"] == 1:
        x = A["mlon"]
        ax.set_xlabel("magnetic longitude")
    else:
        x = A["mlat"]
        ax.set_xlabel("magnetic latitude")

    ax.plot(x, A)

    return fg


def plot3d_input(A, cfg: dict[str, T.Any]) -> Figure:
    fg = Figure()
    ax = fg.gca()

    h0 = ax.pcolormesh(A["mlon"], A["mlat"], A, shading="nearest")
    fg.colorbar(h0, ax=ax)
    ax.set_ylabel("magnetic latitude")
    ax.set_xlabel("magnetic longitude")

    return fg
