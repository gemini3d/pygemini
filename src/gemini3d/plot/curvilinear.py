"""
function names in this module must start with "curv" so that "if" statements in plot/__init__.py work
"""

from __future__ import annotations
import typing as T
from datetime import datetime

import matplotlib as mpl

from ..grid.gridmodeldata import model2magcoords


def curv3d_long(
    fg: mpl.figure.Figure,
    time: datetime,
    xg: dict[str, T.Any],
    parm,
    cfg: dict[str, T.Any],
    *,
    lalt: int = 256,
    llon: int = 256,
    llat: int = 256,
) -> None:
    """plot dipole data vs. alt,lon,lat"""

    altref = 300e3

    # grid data; wasteful and should only do a slice at a time???
    alti, mloni, mlati, parmi = model2magcoords(xg, parm, lalt, llon, llat)

    # define slices indices
    ialt = abs(alti - altref).argmin()
    lonavg = cfg.get("sourcemlon", mloni.mean())
    ilon = abs(mloni - lonavg).argmin()
    latavg = cfg.get("sourcemlat", mlati.mean())
    ilat = abs(mlati - latavg).argmin()

    # plot various slices through the 3D domain

    axs = fg.subplots(1, 3)  # type: T.Any

    ax = axs[0]
    h = ax.pcolormesh(mloni, alti / 1e3, parmi[:, :, ilat], shading="nearest")
    ax.set_xlabel("mlon")
    ax.set_ylabel("alt")
    fg.colorbar(h, ax=ax)

    ax = axs[1]
    h = ax.pcolormesh(mloni, mlati, parmi[ialt, :, :].transpose(), shading="nearest")
    ax.set_xlabel("mlon")
    ax.set_ylabel("mlat")
    fg.colorbar(h, ax=ax)

    ax = axs[2]
    ax.pcolormesh(mlati, alti / 1e3, parmi[:, ilon, :], shading="nearest")
    ax.set_xlabel("mlat")
    ax.set_ylabel("alt")
    fg.colorbar(h, ax=ax)

    fg.suptitle(f"{time}")


def curv2d(
    fg: mpl.figure.Figure,
    time: datetime,
    xg: dict[str, T.Any],
    parm,
    cfg: dict[str, T.Any],
    *,
    lalt: int = 256,
    llat: int = 256,
) -> None:
    """
    Note: leave "cfg" argument so that the polymorphic function call
    is the same as other cases.
    """

    # grid data
    alti, mloni, mlati, parmi = model2magcoords(xg, parm, lalt, 1, llat)

    # define slices indices, for 2D there is only one longitude index
    ilon = 0

    # plot the meridional slice
    ax = fg.gca()
    h = ax.pcolormesh(mlati, alti / 1e3, parmi[:, ilon, :], shading="nearest")
    ax.set_xlabel("mlat")
    ax.set_ylabel("alt")
    fg.colorbar(h, ax=ax)

    ax.set_title(f"{time}")
