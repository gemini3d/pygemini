from __future__ import annotations
from pathlib import Path
from matplotlib.figure import Figure
import typing as T
from datetime import datetime
import logging
import numpy as np
import xarray

from .. import read
from .. import find
from .vis import grid2plotfun
from ..config import datetime_range


PARAMS = ["ne", "v1", "Ti", "Te", "J1", "v2", "v3", "J2", "J3", "Phi"]


def grid(direc: Path, only: list[str] = None, saveplot_fmt: str = None):
    """plot 3D grid

    Parameters
    ----------

    direc: pathlib.Path
        top-level path of simulation grid
    """

    direc = Path(direc).expanduser()

    xg = read.grid(direc)

    if only is None:
        only = ["basic", "alt", "geog"]
        if xg["lxs"].prod() < 10000:
            # gets extremely slow if too many points
            only.append("ecef")

    # %% x1, x2, x3
    if "basic" in only:
        fg = basic(xg)
        stitle(fg, xg)
        save_fig(fg, direc, "grid-basic")

    # %% detailed altitude plot
    if "alt" in only:
        fg = altitude_grid(xg)
        save_fig(fg, direc, "grid-altitude")

    # %% ECEF surface
    if "ecef" in only:
        fg = Figure()
        ax = fg.gca(projection="3d")
        ax.scatter(xg["x"], xg["y"], xg["z"])

        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.set_zlabel("z [m]")

        stitle(fg, xg, "ECEF")
        save_fig(fg, direc, "grid-ecef")

    # %% lat lon map
    if "geog" in only:
        fg = grid_geog(xg)
        save_fig(fg, direc, "grid-geog")


def grid_geog(xg: dict[str, T.Any]) -> Figure:
    """
    plots grid in geographic map
    """

    fig = Figure()

    glon = xg["glon"]
    glat = xg["glat"]

    try:
        import cartopy

        proj = cartopy.crs.PlateCarree()  # arbitrary

        ax = fig.gca(projection=proj)
        ax.add_feature(cartopy.feature.LAND)
        ax.add_feature(cartopy.feature.OCEAN)
        ax.add_feature(cartopy.feature.COASTLINE)
        ax.add_feature(cartopy.feature.BORDERS, linestyle=":")
        ax.gridlines(draw_labels=True, x_inline=False, y_inline=False)
        ax.scatter(glon, glat, transform=proj)
    except ImportError as e:
        logging.error(e)
        ax = fig.gca()
        ax.scatter(glon, glat)

    ax.set_xlabel("geographic longitude")
    ax.set_ylabel("geographic latitude")
    stitle(fig, xg, "glat, glon")

    return fig


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
    if not path:
        raise FileNotFoundError(f"{direc} does not contain E-field data")

    time = datetime_range(cfg["time"][0], cfg["time"][0] + cfg["tdur"], cfg["dtE0"])
    for t in time:
        file = find.frame(path, t)
        if not file:
            logging.error(f"no E-field data found at {t} in {path}")
            continue

        dat = read.Efield(file)

        for k in {"Exit", "Eyit", "Vminx1it", "Vmaxx1it", "Vminx2ist", "Vmaxx2ist"}:
            if dat[k].ndim == 1:
                fg = plot2d_input(dat[k], cfg)
            else:
                fg = plot3d_input(dat[k], cfg)

            fg.suptitle(f"{k}: {t}")

            save_fig(fg, direc, f"Efield-{k}", time=t)


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
    if not precip_path:
        raise FileNotFoundError(f"{direc} does not contain precipitation data")

    time = datetime_range(cfg["time"][0], cfg["time"][0] + cfg["tdur"], cfg["dtprec"])

    for t in time:
        file = find.frame(precip_path, t)
        if not file:
            logging.error(f"no precipitation data found at {t} in {precip_path}")
            continue

        dat = read.precip(file)

        for k in {"E0", "Q"}:
            if dat[k].ndim == 1:
                fg = plot2d_input(dat[k], cfg)
            else:
                fg = plot3d_input(dat[k], cfg)

            fg.suptitle(f"{k}: {t}")

            save_fig(fg, direc, f"precip-{k}", time=t)


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


def basic(xg: dict[str, T.Any]) -> Figure:
    fig = Figure()
    axs = fig.subplots(1, 3)
    # %% x1
    lx1 = xg["x1"].size
    ax = axs[0]
    ax.plot(range(lx1), xg["x1"] / 1000, marker=".")
    ax.set_ylabel("x1 [km]")
    ax.set_xlabel("index (dimensionless)")
    ax.set_title(f"x1 (upward) lx1 = {lx1}")

    # %% x2
    lx2 = xg["x2"].size
    ax = axs[1]
    ax.plot(xg["x2"] / 1e3, range(lx2), marker=".")
    ax.set_xlabel("x2 [km]")
    ax.set_ylabel("index (dimensionless)")
    ax.set_title(f"x2 (eastward) lx2 = {lx2}")

    # %% x3
    lx3 = xg["x3"].size
    ax = axs[2]
    ax.plot(range(lx3), xg["x3"] / 1e3, marker=".")
    ax.set_ylabel("x3 [km]")
    ax.set_xlabel("index (dimensionless)")
    ax.set_title(f"x3 (northward) lx3 = {lx3}")

    return fig


def stitle(fig: Figure, xg: dict[str, T.Any], ttxt: str = ""):
    """suptitle"""
    if "time" in xg:
        ttxt += f" {xg['time']}"

    if "filename" in xg:
        ttxt += f" {xg['filename']}"

    fig.suptitle(ttxt)


def altitude_grid(xg: dict[str, T.Any]) -> Figure:
    """
    plot altitude x1 grid

    Parameters
    ----------

    xg: pathlib.Path or dict of numpy.ndarray
        simulation grid: filename or dict
    """

    if isinstance(xg, Path):
        xg = read.grid(xg)

    x1_km = xg["x1"] / 1000

    fig = Figure()
    ax = fig.gca()

    ax.plot(x1_km, marker="*")
    ax.set_ylabel("x1 [km]")
    ax.set_xlabel("index (dimensionless)")

    file = xg.get("filename", "")

    ax.set_title(
        f"{file}  min. alt: {x1_km.min():0.1f} [km]  max. alt: {x1_km.max():0.1f} [km]  lx1: {x1_km.size}"
    )

    return fig


def plot_3d(direc: Path, var: list[str], saveplot_fmt: str = None):
    from . import vis3d

    direc = Path(direc).expanduser().resolve(strict=True)

    cfg = read.config(direc)
    for t in cfg["time"]:
        vis3d.frame(direc, time=t, var=var, saveplot_fmt=saveplot_fmt)


def plot_all(direc: Path, var: list[str] = None, saveplot_fmt: str = None):

    direc = Path(direc).expanduser().resolve(strict=True)

    if not var:
        var = PARAMS
    if isinstance(var, str):
        var = [var]

    if set(var).intersection({"png", "pdf", "eps"}):
        raise ValueError("please use saveplot_fmt='png' or similar for plot format")

    cfg = read.config(direc)
    # %% loop over files / time
    for t in cfg["time"]:
        frame(direc, time=t, var=var, saveplot_fmt=saveplot_fmt)


def frame(
    direc: Path,
    time: datetime = None,
    *,
    saveplot_fmt: str = None,
    var: list[str] = None,
    xg: dict[str, T.Any] = None,
):
    """
    if save_dir, plots will not be visible while generating to speed plot writing
    """

    if not var:
        var = PARAMS

    file = None
    if time is None:
        if not direc.is_file():
            raise ValueError("must either specify directory and time, or single file")
        file = direc
        direc = direc.parent

    if not xg:
        xg = read.grid(direc)

    if file is None:
        dat = read.frame(direc, time, var=var)
    else:
        dat = read.data(file, var)

    if not dat:
        raise ValueError(f"No data in {direc} at {time}")

    plotfun = grid2plotfun(xg)

    for k, v in dat.items():
        if any(s in k for s in var):
            fg = plotfun(dat.time, xg, v.squeeze(), k, wavelength=dat.get("wavelength"))
            save_fig(fg, direc, k, saveplot_fmt, time)


def plot_input(
    direc: Path, var: list[str] = None, saveplot_fmt: str = None, xg: dict[str, T.Any] = None
):
    """
    plot simulation inputs, under "direc/inputs"

    if save_dir defined, plots will not be visible while generating to speed plot writing
    """

    direc = Path(direc).expanduser().resolve(strict=True)

    if not var:
        var = ["ns", "Ts", "vs1"]

    cfg = read.config(direc)
    init_file = direc / cfg["indat_file"]

    if not xg:
        xg = read.grid(direc)

    dat = read.data(init_file, var=["ns", "Ts", "vs1"])

    if not dat:
        raise ValueError(f"No data in {init_file}")

    plotfun = grid2plotfun(xg)

    for k, v in dat.items():
        if any(s in k for s in var):
            # FIXME: for now we just look at electrons v[-1, ...]
            cmap_name = {"ns": "ne", "Ts": "Te", "vs1": "v1"}
            fg = plotfun(
                dat.time,
                xg,
                v[-1, :, :, :].squeeze(),
                cmap_name[k],
                wavelength=dat.get("wavelength"),
            )
            save_fig(fg, direc, k, saveplot_fmt)


def save_fig(fg: Figure, direc: Path, name: str, fmt: str = "png", time: datetime = None):
    if not fmt:
        fmt = "png"

    if time is None:
        tstr = ""
    else:
        tstr = f"-{time.isoformat().replace(':','')}"

    plot_fn = direc / f"plots/{name}{tstr}.{fmt}"
    plot_fn.parent.mkdir(exist_ok=True)
    print(f"{time} => {plot_fn}")
    fg.savefig(plot_fn)
