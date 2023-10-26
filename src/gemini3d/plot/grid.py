from __future__ import annotations
from pathlib import Path
import logging
import typing as T

import matplotlib as mpl

from .. import read
from .core import basic, stitle, save_fig


def grid(
    direc: Path, only: list[str] | None = None, saveplot_fmt: str | None = None
) -> None:
    """plot 3D grid

    Parameters
    ----------

    direc: pathlib.Path
        top-level path of simulation grid
    """

    fg3 = mpl.figure.Figure(tight_layout=True, figsize=mpl.figure.figaspect(1 / 3))
    fg = mpl.figure.Figure()

    direc = Path(direc).expanduser()

    xg = read.grid(direc)

    if only is None:
        only = ["basic", "alt", "geog"]
        if xg["lx"].prod() < 10000:
            # gets extremely slow if too many points
            only.append("ecef")

    # %% x1, x2, x3
    if "basic" in only:
        basic(fg3, xg)
        stitle(fg3, xg)
        save_fig(fg3, direc, "grid-basic")

    # %% detailed altitude plot
    if "alt" in only:
        ax = fg.gca()
        altitude(ax, xg)
        save_fig(fg, direc, "grid-altitude")

    # %% ECEF surface
    if "ecef" in only:
        fg.clf()  # clear figure for 3D axes when 2D previously
        ax = fg.gca(projection="3d")  # type: ignore

        ax.scatter(xg["x"], xg["y"], xg["z"])

        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.set_zlabel("z [m]")  # type: ignore

        stitle(fg, xg, "ECEF")
        save_fig(fg, direc, "grid-ecef")

    # %% lat lon map
    if "geog" in only:
        fg.clf()
        geographic(fg, xg)
        save_fig(fg, direc, name="grid-geog", fmt=saveplot_fmt)


def geographic(fig: mpl.figure.Figure, xg: dict[str, T.Any]) -> None:
    """
    plots grid in geographic map
    """

    glon = xg["glon"]
    glat = xg["glat"]

    try:
        import cartopy

        proj = cartopy.crs.PlateCarree()  # arbitrary

        ax = fig.gca(projection=proj)  # type: ignore
        ax.add_feature(cartopy.feature.LAND)  # type: ignore
        ax.add_feature(cartopy.feature.OCEAN)  # type: ignore
        ax.add_feature(cartopy.feature.COASTLINE)  # type: ignore
        ax.add_feature(cartopy.feature.BORDERS, linestyle=":")  # type: ignore
        ax.gridlines(draw_labels=True, x_inline=False, y_inline=False)  # type: ignore
        ax.scatter(glon, glat, transform=proj)
    except ImportError as e:
        logging.error(e)
        ax = fig.gca()
        ax.scatter(glon, glat)

    ax.set_xlabel("geographic longitude")
    ax.set_ylabel("geographic latitude")
    stitle(fig, xg, "glat, glon")


def altitude(ax: mpl.axes.Axes, xg: dict[str, T.Any]) -> None:
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

    ax.plot(x1_km, marker="*")
    ax.set_ylabel("x1 [km]")
    ax.set_xlabel("index (dimensionless)")

    file = xg.get("filename", "")

    ax.set_title(
        f"{file}  min. alt: {x1_km.min():0.1f} [km]  max. alt: {x1_km.max():0.1f} [km]  lx1: {x1_km.size}"
    )
