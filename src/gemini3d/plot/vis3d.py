"""
3-D visualizations using Mayavi / VTK

suggested install:

    conda install mayavi
"""

from __future__ import annotations
from pathlib import Path
import numpy as np
from datetime import datetime
import typing as T

# import scipy.interpolate as interp

from mayavi import mlab

# mlab.options.backend = 'envisage'  # for GUI

from . import PARAMS
from .. import read

PLOTFUN = {"scalar": ("ne", "Ti", "Te", "J1", "J2", "J3"), "vector": ("v1", "v2", "v3")}
R_EARTH = 6370e3


def frame(
    direc: Path,
    time: datetime,
    var: list[str] = None,
    saveplot_fmt: str = None,
    xg: dict[str, T.Any] = None,
):
    """
    plot plasma quantities in 3D

    if save_dir, plots will not be visible while generating to speed plot writing
    """
    if not var:
        var = PARAMS

    if not xg:
        xg = read.grid(direc)

    dat = read.frame(direc, time)
    time = dat["time"]

    for k in var:
        if k not in dat:  # not present at this time step, often just the first time step
            continue

        if k in PLOTFUN["scalar"]:
            fg = scalar(time, xg, dat[k].squeeze(), name=k)
        elif k in PLOTFUN["vector"]:
            print("TODO: vector plot", k)

        if saveplot_fmt:
            plot_fn = direc / "plots" / f"{k}-{time.isoformat().replace(':','')}.{saveplot_fmt}"
            plot_fn.parent.mkdir(exist_ok=True)
            print(f"{dat['time']} => {plot_fn}")
            fg.savefig(plot_fn)


def scalar(time: datetime, xg: dict[str, np.ndarray], parm: np.ndarray, name: str):
    """
    plot scalar field data in transparent 3D volume
    """

    if parm.ndim != 3:
        raise ValueError("Mayavi scalar field plots are for 3-D data")

    # %% arbitrary output plot resolution
    # lxp = 100
    # lyp = 100
    # lzp = 100

    # %% SIZE OF SIMULATION
    lxs = read.get_lxs(xg)

    lx2 = lxs[1]
    # inds1 = slice(2, lx1 + 2)
    inds2 = slice(2, lx2 + 2)
    # inds3 = slice(2, lx3 + 2)

    # %% SIZE OF PLOT GRID THAT WE ARE INTERPOLATING ONTO
    meantheta = xg["theta"].mean()
    # this is a mag colat. coordinate and is only used for defining grid in linspaces below
    # runs backward from north distance, hence the negative sign

    # northward distance [m]
    y = -(xg["theta"] - meantheta) * R_EARTH
    # eastward distance [m]
    x = xg["x2"][inds2]
    # upward distance [m]
    z = xg["alt"]

    # Mayavi requires a grid like so:
    # interpolatedz
    # eastward distance [meters]
    # xp = np.linspace(x.min(), x.max(), lxp)
    # northward distance [meters]
    # yp = np.linspace(y.min(), y.max(), lyp)
    # upward distance [meters]
    # zp = np.linspace(z.min(), z.max(), lzp)
    # x3, y3, z3 = np.mgrid[xp[0]: xp[-1]: lxp * 1j,
    #   yp[0]: yp[-1]: lyp * 1j, zp[0]: zp[-1]: lzp * 1j]  # type: ignore

    # non-interpolated
    parm = parm.transpose(1, 2, 0)
    xp = np.linspace(x.min(), x.max(), parm.shape[0])
    yp = np.linspace(y.min(), y.max(), parm.shape[1])
    zp = np.linspace(z.min(), z.max(), parm.shape[2])
    x3, y3, z3 = np.mgrid[
        xp[0] : xp[-1] : xp.size * 1j, yp[0] : yp[-1] : yp.size * 1j, zp[0] : zp[-1] : zp.size * 1j  # type: ignore
    ]

    # %% 3-D interpolation for plot
    if ~np.isfinite(parm).all():
        raise ValueError("Mayavi requires finite data values")

    # TODO: check order of interpolated axes (1,2,0) or ?
    # parmp = interp.interpn(
    #     points=(xg["x1"][inds1], xg["x2"][inds2], xg["x3"][inds3]),
    #     values=parm.values,
    #     xi=np.column_stack((x3.ravel(), y3.ravel(), z3.ravel())),
    #     bounds_error=False,
    # ).reshape((lxp, lyp, lzp))

    # if ~np.isfinite(parmp).all():
    #     raise ValueError('Interpolation issue: Mayavi requires finite data values')

    fig = mlab.figure()
    scf = mlab.pipeline.scalar_field(x3 / 1e3, y3 / 1e3, z3 / 1e3, parm, figure=fig)
    vol = mlab.pipeline.volume(scf, figure=fig)
    mlab.colorbar(vol, title=name)
    mlab.axes(figure=fig, xlabel="x (km)", ylabel="y (km)", zlabel="z (km)")

    return fig
