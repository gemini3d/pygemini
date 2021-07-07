from __future__ import annotations

from datetime import datetime
import numpy as np
import math
import xarray
import scipy.interpolate as interp
from matplotlib.figure import Figure

from ..utils import git_meta
from ..read import get_lxs

from .constants import R_EARTH, REF_ALT
from .slices import (
    plot12,
    plot13,
    plot23,
    plot1d2,
    plot1d3,
    bright_east_north,
    east_north,
    mag_lonlat,
)


def plot_interp(
    time: datetime,
    xg: dict[str, np.ndarray],
    parm: xarray.DataArray,
    fg: Figure = None,
    **kwargs,
) -> Figure:

    """

    Parameters
    ----------

    xp:  eastward distance (rads.)
        should be interpreted as northward distance (in rads.).
        Irrespective of ordering of xg.theta, this will be monotonic increasing!!!

    zp: altitude (meters)

    y:  this is a mag colat. coordinate and is only used for defining
        grid in linspaces below, runs backward from north distance,
        hence the negative sign
    """

    name = parm.name
    assert isinstance(name, str)

    if fg is None:
        fg = Figure(constrained_layout=True)

    meta = git_meta()

    cmap = None
    is_Efield = False
    vmin = None
    vmax = None
    if name.startswith("J") or name == "Phitop":
        cmap = "bwr"
        vmax = abs(parm).max()
        vmin = -vmax
    elif name.startswith("v"):
        cmap = "bwr"
        vmax = 80.0
        vmin = -vmax
    elif name.startswith(("V", "E")):
        is_Efield = True
        cmap = "bwr"
        vmax = abs(parm).max()
        vmin = -vmax
    elif name.startswith("T"):
        vmin = 0.0
        vmax = parm.max()
    elif name.startswith("n"):
        vmin = 1e-7

    # %% SIZE OF SIMULATION
    lxs = get_lxs(xg)

    lx1, lx2, lx3 = lxs
    inds1 = slice(2, lx1 + 2)
    inds2 = slice(2, lx2 + 2)
    inds3 = slice(2, lx3 + 2)
    # %% SIZE OF PLOT GRID THAT WE ARE INTERPOLATING ONTO
    meantheta = xg["theta"].mean()
    # this is a mag colat. coordinate and is only used for defining grid in linspaces below
    # runs backward from north distance, hence the negative sign
    # [radians]
    y = -1 * (xg["theta"] - meantheta)
    # eastward distance [radians]
    x = xg["x2"][inds2] / R_EARTH / math.sin(meantheta)
    # altitude [meters]
    z = xg["alt"] / 1e3

    # arbitrary output plot resolution
    lxp = 500
    lyp = 500
    lzp = 500

    # eastward distance [meters]
    xp = np.linspace(x.min(), x.max(), lxp) * R_EARTH * math.sin(meantheta)
    # northward distance [meters]
    yp = np.linspace(y.min(), y.max(), lyp) * R_EARTH
    # upward distance [meters]
    zp = np.linspace(z.min(), z.max(), lzp) * 1e3

    # %% INTERPOLATE ONTO PLOTTING GRID
    if lxs[2] == 1:  # alt./lon. slice
        ax = fg.gca()
        ax.set_title(f"{name}: {time.isoformat()}  {meta['commit']}")
        # meridional meshgrid, this defines the grid for plotting
        # slice expects the first dim. to be "y" ("z" in the 2D case)
        # %% CONVERT ANGULAR COORDINATES TO MLAT,MLON
        i = np.argsort(xp)  # FIXME: this was in Matlab code--what is its purpose?

        if name == "rayleighs":
            f = interp.interp1d(xg["x2"][inds2], parm, axis=1, bounds_error=False)
            # hack for pcolormesh to put labels in center of pixel
            wl = kwargs["wavelength"] + [""]
            hi = ax.pcolormesh(xp / 1e3, np.arange(len(wl)), f(xp)[:, i], shading="nearest")
            ax.set_yticks(np.arange(len(wl)) + 0.5)
            ax.set_yticklabels(wl)
            ax.set_ylim(0, len(wl) - 1)
            fg.colorbar(hi, ax=ax, aspect=60, pad=0.01)
            # end hack
            ax.set_ylabel(r"wavelength $\AA$")
            ax.set_xlabel("eastward dist. (km)")
        elif parm.ndim == 2:
            f = interp.interp2d(xg["x2"][inds2], xg["x1"][inds1], parm, bounds_error=False)
            plot12(
                xp[i], zp, f(xp, zp)[:, i], name=name, cmap=cmap, vmin=vmin, vmax=vmax, fg=fg, ax=ax
            )
        elif parm.ndim == 1:  # phitop
            f = interp.interp1d(xg["x2"][inds2], parm, bounds_error=False)
            plot1d2(xp, f(xp), name, fg, ax)
        else:
            raise ValueError(f"{name}: only 2D and 1D data are expected--squeeze data")
    elif lxs[1] == 1:  # alt./lat. slice
        ax = fg.gca()
        ax.set_title(f"{name}: {time.isoformat()}  {meta['commit']}")
        # so north dist, east dist., alt.
        # slice expects the first dim. to be "y"
        # %% CONVERT ANGULAR COORDINATES TO MLAT,MLON
        i = np.argsort(yp)  # FIXME: this was in Matlab code--what is its purpose?

        if name == "rayleighs":
            # FIXME: this needs to be tested
            f = interp.interp1d(xg["x3"][inds3], parm, axis=1, bounds_error=False)
            # hack for pcolormesh to put labels in center of pixel
            wl = kwargs["wavelength"] + [""]
            hi = ax.pcolormesh(np.arange(len(wl)), yp / 1e3, f(yp)[:, i].T, shading="nearest")
            ax.set_xticks(np.arange(len(wl)) + 0.5)
            ax.set_xticklabels(wl)
            ax.set_xlim(0, len(wl) - 1)
            fg.colorbar(hi, ax=ax, aspect=60, pad=0.01)
            # end hack
            ax.set_xlabel(r"wavelength $\AA$")
            ax.set_ylabel("northward dist. (km)")
        elif parm.ndim == 2:
            f = interp.interp2d(xg["x3"][inds3], xg["x1"][inds1], parm, bounds_error=False)
            parmp = f(yp, zp).reshape((lzp, lyp))
            plot13(yp[i], zp, parmp[:, i], name=name, cmap=cmap, vmin=vmin, vmax=vmax, fg=fg, ax=ax)
        elif parm.ndim == 1:  # phitop
            f = interp.interp1d(xg["x3"][inds3], parm, bounds_error=False)
            plot1d3(yp, f(yp), name, fg, ax)
        else:
            raise ValueError(f"{name}: only 2D and 1D data are expected--squeeze data")
    elif parm.ndim == 3:
        plot3d_slice(
            fg,
            name,
            time,
            meta,
            inds1,
            inds2,
            inds3,
            lx2,
            lx3,
            lxp,
            lyp,
            xp,
            yp,
            zp,
            cmap,
            vmin,
            vmax,
            parm,
            xg,
        )
    elif name == "rayleighs":
        bright_east_north(
            fg,
            xg,
            parm,
            xp,
            yp,
            inds2,
            inds3,
            cmap,
            vmin,
            vmax,
            name,
            time,
            kwargs["wavelength"],
        )
    elif is_Efield:
        # single 2D plot
        mag_lonlat(fg, xg, parm, cmap, vmin, vmax, name, time)
    else:
        # single 2D plot
        east_north(fg, xg, parm, xp, yp, inds2, inds3, cmap, vmin, vmax, name, time)

    return fg


def plot3d_slice(
    fg,
    name,
    time,
    meta,
    inds1,
    inds2,
    inds3,
    lx2,
    lx3,
    lxp,
    lyp,
    xp,
    yp,
    zp,
    cmap,
    vmin,
    vmax,
    parm,
    xg,
):

    fg.set_size_inches((18, 5))
    axs = fg.subplots(1, 3, sharey=False, sharex=False)
    fg.suptitle(f"{name}: {time.isoformat()}  {meta['commit']}", y=0.99)
    # %% CONVERT TO DISTANCE UP, EAST, NORTH (left panel)
    # JUST PICK AN X3 LOCATION FOR THE MERIDIONAL SLICE PLOT,
    # AND AN ALTITUDE FOR THE LAT./LON. SLICE
    ix3 = lx3 // 2 - 1  # arbitrary slice, to match Matlab
    f = interp.interp2d(xg["x2"][inds2], xg["x1"][inds1], parm[:, :, ix3], bounds_error=False)
    # CONVERT ANGULAR COORDINATES TO MLAT,MLON
    ix = np.argsort(xp)
    iy = np.argsort(yp)
    plot12(
        xp[ix], zp, f(xp, zp)[:, ix], name=name, cmap=cmap, vmin=vmin, vmax=vmax, fg=fg, ax=axs[0]
    )
    # %% LAT./LONG. SLICE COORDINATES (center panel)
    zp2 = REF_ALT
    X3, Y3, Z3 = np.meshgrid(xp, yp, zp2 * 1e3)
    # transpose: so north dist, east dist., alt.
    parmp = interp.interpn(
        points=(xg["x1"][inds1], xg["x2"][inds2], xg["x3"][inds3]),
        values=parm.data,
        xi=np.column_stack((Z3.ravel(), X3.ravel(), Y3.ravel())),
        bounds_error=False,
    ).reshape((1, lxp, lyp))

    parmp = parmp[:, :, iy]  # must be indexed in two steps
    plot23(
        xp[ix],
        yp[iy],
        parmp[0, ix, :],
        name=name,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        fg=fg,
        ax=axs[1],
    )
    # %% ALT/LAT SLICE (right panel)
    ix2 = lx2 // 2 - 1  # arbitrary slice, to match Matlab
    f = interp.interp2d(xg["x3"][inds3], xg["x1"][inds1], parm[:, ix2, :], bounds_error=False)
    plot13(
        yp[iy], zp, f(yp, zp)[:, iy], name=name, cmap=cmap, vmin=vmin, vmax=vmax, fg=fg, ax=axs[2]
    )

    return fg


cart3d_long_ENU = plot_interp
cart2d = plot_interp
