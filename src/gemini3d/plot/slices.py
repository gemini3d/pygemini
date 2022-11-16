from __future__ import annotations

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
    *,
    name: str,
    ref_alt: float,
    cmap: str | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
) -> None:

    if parm.ndim != 2:
        raise ValueError(f"data must have 2 dimensions, you have {parm.shape}")

    hi = ax.pcolormesh(x / 1e3, z / 1e3, parm, cmap=cmap, vmin=vmin, vmax=vmax, shading="nearest")
    ax.yaxis.set_major_locator(MultipleLocator(100))
    ax.set_xlabel("eastward dist. (km)")
    ax.set_ylabel("upward dist. (km)")
    ax.axhline(ref_alt, color="w", linestyle="--", linewidth=2)
    ax.figure.colorbar(hi, ax=ax, label=CB_LBL[name])


def plot13(
    y,
    z,
    parm,
    ax: Axes,
    *,
    name: str,
    cmap: str | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
) -> None:

    if parm.ndim != 2:
        raise ValueError(f"data must have 2 dimensions, you have {parm.shape}")

    hi = ax.pcolormesh(y / 1e3, z / 1e3, parm, cmap=cmap, vmin=vmin, vmax=vmax, shading="nearest")
    ax.yaxis.set_major_locator(MultipleLocator(100))
    ax.set_xlabel("northward dist. (km)")
    ax.set_ylabel("upward dist. (km)")
    ax.figure.colorbar(hi, ax=ax, label=CB_LBL[name])


def plot23(
    x,
    y,
    parm,
    name: str,
    ax: Axes,
    *,
    cmap: str | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
) -> None:

    if parm.ndim != 2:
        raise ValueError(f"data must have 2 dimensions, you have {parm.shape}")

    hi = ax.pcolormesh(x / 1e3, y / 1e3, parm, cmap=cmap, vmin=vmin, vmax=vmax, shading="nearest")
    ax.set_xlabel("eastward dist. (km)")
    ax.set_ylabel("northward dist. (km)")
    ax.figure.colorbar(hi, ax=ax, label=CB_LBL[name])


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
    fg: Figure, grid, parm, xp, yp, inds2, inds3, cmap, vmin, vmax, name, time, wavelength
) -> None:

    if parm.ndim != 3:
        raise ValueError(f"Expected 3D data but got {parm.ndim}D data.")

    meta = git_meta()

    axs = fg.subplots(2, 2, sharey=True, sharex=True).ravel()
    fg.suptitle(f"{name}: {time.isoformat()}  {meta['commit']}", y=0.99)
    # arbitrary pick of which emission lines to plot lat/lon slices
    for j, i in enumerate([1, 3, 4, 8]):
        f = interp.interp2d(grid["x3"][inds3], grid["x2"][inds2], parm[i, :, :], bounds_error=False)
        hi = axs[j].pcolormesh(xp / 1e3, yp / 1e3, f(yp, xp), shading="nearest")
        axs[j].set_title(wavelength[i] + r"$\AA$")
        fg.colorbar(hi, ax=axs[j], label="Rayleighs")
    axs[2].set_xlabel("eastward dist. (km)")
    axs[2].set_ylabel("northward dist. (km)")


def east_north(ax: Axes, grid, parm, xp, yp, inds2, inds3, cmap, vmin, vmax, name, time) -> None:

    if parm.ndim != 2:
        raise ValueError(f"Expected 2D data but got {parm.ndim}D data.")

    meta = git_meta()

    f = interp.interp2d(grid["x3"][inds3], grid["x2"][inds2], parm, bounds_error=False)
    hi = ax.pcolormesh(
        xp / 1e3, yp / 1e3, f(yp, xp), cmap=cmap, vmin=vmin, vmax=vmax, shading="nearest"
    )
    ax.set_xlabel("eastward dist. (km)")
    ax.set_ylabel("northward dist. (km)")
    ax.set_title(f"{name}: {time.isoformat()}  {meta['commit']}")
    ax.figure.colorbar(hi, ax=ax, label=CB_LBL[name])


def mag_lonlat(ax: Axes, grid, parm, cmap, vmin, vmax, name, time) -> None:

    if parm.ndim != 2:
        raise ValueError(f"Expected 2D data but got {parm.ndim}D data.")

    meta = git_meta()

    hi = ax.pcolormesh(
        grid["mlon"], grid["mlat"], parm, cmap=cmap, vmin=vmin, vmax=vmax, shading="nearest"
    )
    ax.set_xlabel("magnetic longitude (deg.)")
    ax.set_ylabel("magnetic latitude (deg.)")
    ax.set_title(f"{name}: {time.isoformat()}  {meta['commit']}")
    ax.figure.colorbar(hi, ax=ax, label=CB_LBL[name])
