from __future__ import annotations
from datetime import datetime
import numpy as np
import scipy.interpolate as interp

from matplotlib.ticker import MultipleLocator
from matplotlib.figure import Figure
from matplotlib.axes import Axes

from .constants import CB_LBL
from ..utils import git_meta


def plot12(
    x,
    z,
    parm,
    ax: Axes,
    clim: tuple,
    *,
    name: str,
    ref_alt: float,
    cmap: str | None = None,
) -> None:
    if parm.ndim != 2:
        raise ValueError(f"data must have 2 dimensions, you have {parm.shape}")

    hi = ax.pcolormesh(
        x / 1e3, z / 1e3, parm, cmap=cmap, vmin=clim[0], vmax=clim[1], shading="nearest"
    )
    ax.yaxis.set_major_locator(MultipleLocator(100))
    ax.set_xlabel("eastward dist. (km)")
    ax.set_ylabel("upward dist. (km)")
    ax.axhline(ref_alt, color="w", linestyle="--", linewidth=2)
    ax.get_figure().colorbar(hi, ax=ax, label=CB_LBL[name])  # type: ignore


def plot13(
    y,
    z,
    parm,
    ax: Axes,
    clim: tuple,
    *,
    name: str,
    cmap: str | None = None,
) -> None:
    if parm.ndim != 2:
        raise ValueError(f"data must have 2 dimensions, you have {parm.shape}")

    hi = ax.pcolormesh(
        y / 1e3, z / 1e3, parm, cmap=cmap, vmin=clim[0], vmax=clim[1], shading="nearest"
    )
    ax.yaxis.set_major_locator(MultipleLocator(100))
    ax.set_xlabel("northward dist. (km)")
    ax.set_ylabel("upward dist. (km)")
    ax.get_figure().colorbar(hi, ax=ax, label=CB_LBL[name])  # type: ignore


def plot23(
    x,
    y,
    parm,
    name: str,
    ax: Axes,
    clim: tuple,
    *,
    cmap: str | None = None,
) -> None:
    if parm.ndim != 2:
        raise ValueError(f"data must have 2 dimensions, you have {parm.shape}")

    hi = ax.pcolormesh(
        x / 1e3, y / 1e3, parm, cmap=cmap, vmin=clim[0], vmax=clim[1], shading="nearest"
    )
    ax.set_xlabel("eastward dist. (km)")
    ax.set_ylabel("northward dist. (km)")
    ax.get_figure().colorbar(hi, ax=ax, label=CB_LBL[name])  # type: ignore


def plot1d2(x, parm, name: str, ax: Axes) -> None:
    if parm.ndim != 1:
        raise ValueError("expecting 1-D data oriented east-west (along latitude)")

    ax.plot(x / 1e3, parm)
    ax.set_xlabel("eastward dist. (km)")
    ax.set_ylabel(CB_LBL[name])


def plot1d3(y, parm, name: str, ax: Axes) -> None:
    if parm.ndim != 1:
        raise ValueError("expecting 1-D data oriented east-west (along latitude)")

    ax.plot(y / 1e3, parm)
    ax.set_xlabel("northward dist. (km)")
    ax.set_ylabel(CB_LBL[name])


def bright_east_north(
    fg: Figure,
    grid,
    parm,
    xp,
    yp,
    inds2: slice,
    inds3: slice,
    cmap: str | None,
    clim: tuple[float, float],
    name: str,
    time: datetime,
    wavelength,
) -> None:
    if parm.ndim != 3:
        raise ValueError(f"Expected 3D data but got {parm.ndim}D data.")

    meta = git_meta()

    axs = fg.subplots(2, 2, sharey=True, sharex=True).ravel()  # type: ignore
    fg.suptitle(f"{name}: {time.isoformat()}  {meta['commit']}", y=0.99)
    # arbitrary pick of which emission lines to plot lat/lon slices
    Xp, Yp = np.meshgrid(xp, yp, indexing="ij")
    for j, i in enumerate([1, 3, 4, 8]):
        f = interp.RegularGridInterpolator(
            (grid["x2"][inds2], grid["x3"][inds3]),
            parm[i, :, :].data.astype(np.float64),
            bounds_error=False,
        )
        hi = axs[j].pcolormesh(
            xp / 1e3,
            yp / 1e3,
            f((Yp, Xp)),
            shading="nearest",
            cmap=cmap,
            vmin=clim[0],
            vmax=clim[1],
        )
        axs[j].set_title(wavelength[i] + r"$\AA$")
        fg.colorbar(hi, ax=axs[j], label="Rayleighs")
    axs[2].set_xlabel("eastward dist. (km)")
    axs[2].set_ylabel("northward dist. (km)")


def east_north(
    ax: Axes,
    grid,
    parm,
    xp,
    yp,
    inds2,
    inds3,
    cmap: str | None,
    clim: tuple[float, float],
    name: str,
    time: datetime,
) -> None:
    if parm.ndim != 2:
        raise ValueError(f"Expected 2D data but got {parm.ndim}D data.")

    meta = git_meta()

    Xp, Yp = np.meshgrid(xp, yp, indexing="ij")
    f = interp.RegularGridInterpolator(
        (grid["x2"][inds2], grid["x3"][inds3]),
        parm.data.astype(np.float64),
        bounds_error=False,
    )
    hi = ax.pcolormesh(
        xp / 1e3,
        yp / 1e3,
        f((Yp, Xp)),
        cmap=cmap,
        vmin=clim[0],
        vmax=clim[1],
        shading="nearest",
    )
    ax.set_xlabel("eastward dist. (km)")
    ax.set_ylabel("northward dist. (km)")
    ax.set_title(f"{name}: {time.isoformat()}  {meta['commit']}")
    ax.get_figure().colorbar(hi, ax=ax, label=CB_LBL[name])  # type: ignore


def mag_lonlat(
    ax: Axes,
    grid,
    parm,
    cmap: str | None,
    clim: tuple[float, float],
    name: str,
    time: datetime,
) -> None:
    if parm.ndim != 2:
        raise ValueError(f"Expected 2D data but got {parm.ndim}D data.")

    meta = git_meta()

    hi = ax.pcolormesh(
        grid["mlon"],
        grid["mlat"],
        parm,
        cmap=cmap,
        vmin=clim[0],
        vmax=clim[1],
        shading="nearest",
    )
    ax.set_xlabel("magnetic longitude (deg.)")
    ax.set_ylabel("magnetic latitude (deg.)")
    ax.set_title(f"{name}: {time.isoformat()}  {meta['commit']}")
    ax.get_figure().colorbar(hi, ax=ax, label=CB_LBL[name])  # type: ignore
