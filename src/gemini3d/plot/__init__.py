from pathlib import Path
from matplotlib.pyplot import close, figure
import typing as T
from datetime import datetime

from .. import read
from .vis import grid2plotfun


PARAMS = ["ne", "v1", "Ti", "Te", "J1", "v2", "v3", "J2", "J3", "Phitop"]


def grid(xg: T.Dict[str, T.Any], only: T.Sequence[str] = None, saveplot_fmt: str = None):
    """plot 3D grid

    Parameters
    ----------

    xg: pathlib.Path or dict of numpy.ndarray
        simulation grid: filename or dict
    """

    if isinstance(xg, (str, Path)):
        xg = read.grid(xg)

    assert xg, "not a simulation grid"

    if only is None:
        only = ("basic", "alt", "ecef", "geog")

    # %% x1, x2, x3
    if "basic" in only:
        basic(xg)

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
    # FIXME: use cartopy
    if "geog" in only:
        fig = figure()
        ax = fig.gca()
        ax.scatter(xg["glat"], xg["glon"])
        ax.set_xlabel("geographic longitude")
        ax.set_ylabel("geographic latitude")
        stitle(fig, xg, "glat, glon")


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

    stitle(fig, xg)


def stitle(fig, xg, ttxt: str = ""):
    """suptitle"""
    if "time" in xg:
        ttxt += f" {xg['time']}"

    if "filename" in xg:
        ttxt += xg.filename

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
    plotfun = grid2plotfun(xg)

    dat = read.frame(direc, time)

    for k in var:
        if k not in dat:  # not present at this time step, often just the first time step
            continue

        fg = plotfun(time, xg, dat[k][1].squeeze(), k, wavelength=dat.get("wavelength"))

        if saveplot_fmt:
            plot_fn = direc / "plots" / f"{k}-{time.isoformat().replace(':','')}.png"
            plot_fn.parent.mkdir(exist_ok=True)
            print(f"{dat['time']} => {plot_fn}")
            fg.savefig(plot_fn)
            close(fg)
