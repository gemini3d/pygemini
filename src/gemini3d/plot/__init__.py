from pathlib import Path
from matplotlib.pyplot import close, figure
import typing as T
from datetime import datetime
import logging
import numpy as np
import xarray

from .. import read
from .. import find
from .vis import grid2plotfun


PARAMS = ["ne", "v1", "Ti", "Te", "J1", "v2", "v3", "J2", "J3", "Phitop"]


def grid(direc: Path, only: T.Sequence[str] = None, saveplot_fmt: str = None):
    """plot 3D grid

    Parameters
    ----------

    direc: pathlib.Path
        top-level path of simulation grid
    """

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

    # %% detailed altitude plot
    if "alt" in only:
        altitude_grid(xg)

    # %% ECEF surface
    if "ecef" in only:
        fig3 = figure()
        ax = fig3.gca(projection="3d")
        ax.scatter(xg["x"], xg["y"], xg["z"])

        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.set_zlabel("z [m]")

        stitle(fig3, xg, "ECEF")

    # %% lat lon map
    if "geog" in only:
        grid_geog(xg)


def grid_geog(xg: T.Dict[str, T.Any]):
    """
    plots grid in geographic map
    """

    fig = figure()

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


def precip(direc: Path):
    """plot input precipitation

    Parameters
    ----------

    direc: pathlib.Path
        top-level simulation directory
    """

    cfg = read.config(direc)
    precip_path = find.precip(direc, cfg.get("precdir"))
    if not precip_path:
        raise FileNotFoundError(f"{direc} does not contain precipitation data")

    for t in cfg["time"]:
        file = find.frame(precip_path, t)
        dat = read.precip(file)

        E0 = dat["E0"].squeeze()
        Q = dat["Q"].squeeze()

        fg = figure()
        axs = fg.subplots(1, 2, sharex=True)

        if E0.ndim == 1:
            # 2D sim
            if cfg["lyp"] == 1:
                x = dat["mlon"]
                axs[0].set_xlabel("magnetic longitude")
            else:
                x = dat["mlat"]
                axs[0].set_xlabel("magnetic latitude")

            axs[0].plot(x, E0)
            axs[1].plot(x, Q)
            axs[0].set_ylabel("ev")
            axs[1].set_ylabel("flux")
        else:
            # 3D sim
            h0 = axs[0].pcolormesh(dat["mlon"], dat["mlat"], E0, shading="nearest")
            h1 = axs[1].pcolormesh(dat["mlon"], dat["mlat"], Q, shading="nearest")
            fg.colorbar(h0, ax=axs[0])
            fg.colorbar(h1, ax=axs[1])
            axs[0].set_ylabel("magnetic latitude")
            axs[0].set_xlabel("magnetic longitude")

        fg.suptitle(f"particle precipitation input: {t}")
        axs[0].set_title("$E_0$ characteristic energy")
        axs[1].set_title("$Q$")


def basic(xg: T.Dict[str, T.Any]):
    fig = figure()
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


def stitle(fig, xg, ttxt: str = ""):
    """suptitle"""
    if "time" in xg:
        ttxt += f" {xg['time']}"

    if "filename" in xg:
        ttxt += f" {xg['filename']}"

    fig.suptitle(ttxt)


def altitude_grid(xg: T.Dict[str, T.Any]):
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

    fig = figure()
    ax = fig.gca()

    ax.plot(x1_km, marker="*")
    ax.set_ylabel("x1 [km]")
    ax.set_xlabel("index (dimensionless)")

    file = xg.get("filename", "")

    ax.set_title(
        f"{file}  min. alt: {x1_km.min():0.1f} [km]  max. alt: {x1_km.max():0.1f} [km]  lx1: {x1_km.size}"
    )

    return fig


def plot_3d(direc: Path, var: T.Sequence[str], saveplot_fmt: str = None):
    from . import vis3d

    direc = Path(direc).expanduser().resolve(strict=True)

    cfg = read.config(direc)
    for t in cfg["time"]:
        vis3d.frame(direc, time=t, var=var, saveplot_fmt=saveplot_fmt)


def plot_all(direc: Path, var: T.Sequence[str] = None, saveplot_fmt: str = None):

    direc = Path(direc).expanduser().resolve(strict=True)

    cfg = read.config(direc)
    # %% loop over files / time
    for t in cfg["time"]:
        frame(direc, time=t, var=var, saveplot_fmt=saveplot_fmt)


def frame(
    direc: Path,
    time: datetime,
    saveplot_fmt: str = None,
    var: T.Sequence[str] = None,
    xg: T.Dict[str, T.Any] = None,
):
    """
    if save_dir, plots will not be visible while generating to speed plot writing
    """
    if not var:
        var = PARAMS

    if not xg:
        xg = read.grid(direc)

    dat = read.frame(direc, time)

    for k in var:
        if k not in dat:  # not present at this time step, often just the first time step
            continue

        plotfun = grid2plotfun(xg)
        time = to_datetime(dat["time"])

        fg = plotfun(time, xg, dat[k].squeeze(), k, wavelength=dat.get("wavelength"))
        save_fig(fg, direc, k, time, saveplot_fmt)


def save_fig(fg, direc: Path, name: str, time: datetime, fmt: str):
    if not fmt or not fg:
        return

    plot_fn = direc / "plots" / f"{name}-{time.isoformat().replace(':','')}.{fmt}"
    plot_fn.parent.mkdir(exist_ok=True)
    print(f"{time} => {plot_fn}")
    fg.savefig(plot_fn)
    close(fg)


def to_datetime(times: xarray.DataArray) -> datetime:
    """
    xarray time to python datetime.datetime
    """

    if isinstance(times, xarray.DataArray):
        times = times.values  # numpy.datetime64
    if isinstance(times, np.datetime64):
        times = times.astype("datetime64[us]").astype(datetime)
    if isinstance(times, np.ndarray):
        times = times.squeeze()[()]  # might still be array, but squeezed at least

    return times
