from __future__ import annotations
import typing as T

import numpy as np
import xarray
from matplotlib.ticker import MultipleLocator
from matplotlib.figure import Figure
import scipy.interpolate as interp

from .constants import CB_LBL, REF_ALT
from ..utils import git_meta

if T.TYPE_CHECKING:
    import matplotlib.axes as mpla


def plot12(
    x: np.ndarray | xarray.DataArray,
    z: np.ndarray | xarray.DataArray,
    parm: np.ndarray | xarray.DataArray,
    *,
    name: str,
    cmap: str = None,
    vmin: float = None,
    vmax: float = None,
    fg: Figure = None,
    ax: "mpla.Axes" = None,
) -> Figure:

    if parm.ndim != 2:
        raise ValueError(f"data must have 2 dimensions, you have {parm.shape}")

    if fg is None:
        fg = Figure(constrained_layout=True)
    if ax is None:
        ax = fg.gca()

    hi = ax.pcolormesh(x / 1e3, z / 1e3, parm, cmap=cmap, vmin=vmin, vmax=vmax, shading="nearest")
    ax.yaxis.set_major_locator(MultipleLocator(100))
    ax.set_xlabel("eastward dist. (km)")
    ax.set_ylabel("upward dist. (km)")
    ax.axhline(REF_ALT, color="w", linestyle="--", linewidth=2)
    fg.colorbar(hi, ax=ax, label=CB_LBL[name])

    return fg


def plot13(
    y: np.ndarray | xarray.DataArray,
    z: np.ndarray | xarray.DataArray,
    parm: np.ndarray | xarray.DataArray,
    *,
    name: str,
    cmap: str = None,
    vmin: float = None,
    vmax: float = None,
    fg: Figure = None,
    ax: "mpla.Axes" = None,
) -> Figure:

    if parm.ndim != 2:
        raise ValueError(f"data must have 2 dimensions, you have {parm.shape}")

    if fg is None:
        fg = Figure(constrained_layout=True)
    if ax is None:
        ax = fg.gca()

    hi = ax.pcolormesh(y / 1e3, z / 1e3, parm, cmap=cmap, vmin=vmin, vmax=vmax, shading="nearest")
    ax.yaxis.set_major_locator(MultipleLocator(100))
    ax.set_xlabel("northward dist. (km)")
    ax.set_ylabel("upward dist. (km)")
    fg.colorbar(hi, ax=ax, label=CB_LBL[name])

    return fg


def plot23(
    x: np.ndarray | xarray.DataArray,
    y: np.ndarray | xarray.DataArray,
    parm: np.ndarray | xarray.DataArray,
    name: str,
    *,
    cmap: str = None,
    vmin: float = None,
    vmax: float = None,
    fg: Figure = None,
    ax: "mpla.Axes" = None,
) -> Figure:

    if parm.ndim != 2:
        raise ValueError(f"data must have 2 dimensions, you have {parm.shape}")

    if fg is None:
        fg = Figure(constrained_layout=True)
    if ax is None:
        ax = fg.gca()

    hi = ax.pcolormesh(x / 1e3, y / 1e3, parm, cmap=cmap, vmin=vmin, vmax=vmax, shading="nearest")
    ax.set_xlabel("eastward dist. (km)")
    ax.set_ylabel("northward dist. (km)")
    fg.colorbar(hi, ax=ax, label=CB_LBL[name])

    return fg


def plot1d2(
    x: np.ndarray, parm: np.ndarray, name: str, fg: Figure = None, ax: "mpla.Axes" = None
) -> Figure:

    if parm.ndim != 1:
        raise ValueError("expecting 1-D data oriented east-west (along latitude)")

    if fg is None:
        fg = Figure(constrained_layout=True)
    if ax is None:
        ax = fg.gca()

    ax.plot(x / 1e3, parm)
    ax.set_xlabel("eastward dist. (km)")
    ax.set_ylabel(CB_LBL[name])

    return fg


def plot1d3(
    y: np.ndarray, parm: np.ndarray, name: str, fg: Figure = None, ax: "mpla.Axes" = None
) -> Figure:

    if parm.ndim != 1:
        raise ValueError("expecting 1-D data oriented east-west (along latitude)")

    if fg is None:
        fg = Figure(constrained_layout=True)
    if ax is None:
        ax = fg.gca()

    ax.plot(y / 1e3, parm)
    ax.set_xlabel("northward dist. (km)")
    ax.set_ylabel(CB_LBL[name])

    return fg


def bright_east_north(
    fg, grid, parm, xp, yp, inds2, inds3, cmap, vmin, vmax, name, time, wavelength
):

    if parm.ndim != 3:
        raise ValueError(f"Expected 3D data, you gave {parm.ndim}D data.")

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


def east_north(fg, grid, parm, xp, yp, inds2, inds3, cmap, vmin, vmax, name, time):

    if parm.ndim != 2:
        raise ValueError(f"Expected 2D data, you gave {parm.ndim}D data.")

    meta = git_meta()

    ax = fg.gca()

    f = interp.interp2d(grid["x3"][inds3], grid["x2"][inds2], parm, bounds_error=False)
    hi = ax.pcolormesh(
        xp / 1e3, yp / 1e3, f(yp, xp), cmap=cmap, vmin=vmin, vmax=vmax, shading="nearest"
    )
    ax.set_xlabel("eastward dist. (km)")
    ax.set_ylabel("northward dist. (km)")
    ax.set_title(f"{name}: {time.isoformat()}  {meta['commit']}")
    fg.colorbar(hi, ax=ax, label=CB_LBL[name])


def mag_lonlat(fg, grid, parm, cmap, vmin, vmax, name, time):

    if parm.ndim != 2:
        raise ValueError(f"Expected 2D data, you gave {parm.ndim}D data.")

    meta = git_meta()

    ax = fg.gca()
    hi = ax.pcolormesh(
        grid["mlon"], grid["mlat"], parm, cmap=cmap, vmin=vmin, vmax=vmax, shading="nearest"
    )
    ax.set_xlabel("magnetic longitude (deg.)")
    ax.set_ylabel("magnetic latitude (deg.)")
    ax.set_title(f"{name}: {time.isoformat()}  {meta['commit']}")
    fg.colorbar(hi, ax=ax, label=CB_LBL[name])
