"""
HDF5 file read
"""

from __future__ import annotations
from pathlib import Path
import typing as T
import logging
from datetime import datetime, timedelta

import xarray
import numpy as np
import h5py

from gemini3d.utils import filename2datetime

from .. import find
from .. import WAVELEN


def simsize(path: Path) -> tuple[int, ...]:
    """
    get simulation size

    Parameters
    ----------
    fn: pathlib.Path
        filepath to simsize.dat

    Returns
    -------
    size: tuple of int, int, int
        3 integers telling simulation grid size
    """

    path = find.simsize(path, ".h5")

    with h5py.File(path, "r") as f:
        if "lxs" in f:
            lx = f["lxs"][:]
        elif "lx" in f:
            lx = f["lx"][:]
        elif "lx1" in f:
            if f["lx1"].ndim > 0:
                lx = np.array(
                    [
                        f["lx1"][:].squeeze()[()],
                        f["lx2"][:].squeeze()[()],
                        f["lx3"][:].squeeze()[()],
                    ]
                )
            else:
                lx = np.array([f["lx1"][()], f["lx2"][()], f["lx3"][()]])
        else:
            raise KeyError(f"could not find '/lxs', '/lx' or '/lx1' in {path.as_posix()}")

    return lx


def flagoutput(file: Path, cfg: dict[str, T.Any]) -> int:
    """detect output type"""

    with h5py.File(file, "r") as f:
        if "nsall" in f:
            # milestone or full
            flag = 1
        elif "flagoutput" in f:
            flag = f["/flagoutput"][()]
        elif "ne" in f and f["/ne"].ndim == 3:
            flag = 3
        elif "neall" in f:
            flag = 2
        else:
            flag = cfg["flagoutput"]

    return flag


def grid(file: Path, *, var: set[str] = None, shape: bool = False) -> dict[str, T.Any]:
    """
    get simulation grid

    Parameters
    ----------
    file: pathlib.Path
        filepath to simgrid
    var: set of str, optional
        read only these grid variables
    shape: bool, optional
        read only the shape of the grid instead of the data iteslf

    Returns
    -------
    grid: dict
        grid parameters

    Transpose on read to undo the transpose operation we had to do in write_grid C => Fortran order.
    """

    xg: dict[str, T.Any] = {}

    if not file.is_file():
        file = find.grid(file, suffix=".h5")

    if shape:
        with h5py.File(file, "r") as f:
            for k in f.keys():
                if f[k].ndim >= 2:
                    xg[k] = f[k].shape[::-1]
                else:
                    xg[k] = f[k].shape

        xg["lx"] = np.array([xg["x1"], xg["x2"], xg["x3"]])
        return xg

    if isinstance(var, str):
        var = [var]

    with h5py.File(file, "r") as f:
        var = set(var) if var else f.keys()

        for k in var:
            if f[k].ndim >= 2:
                xg[k] = f[k][:].transpose()
            else:
                if f[k].size > 1:
                    xg[k] = f[k][:]
                else:
                    xg[k] = f[k]

    xg["lx"] = simsize(file.with_name("simsize.h5"))

    return xg


def Efield(file: Path) -> xarray.Dataset:
    """
    load electric field
    """

    with h5py.File(file.with_name("simgrid.h5"), "r") as f:
        E = xarray.Dataset(coords={"mlon": f["/mlon"][:], "mlat": f["/mlat"][:]})

    with h5py.File(file, "r") as f:
        E["flagdirich"] = f["flagdirich"][()].item()
        for p in ("Exit", "Eyit", "Vminx1it", "Vmaxx1it"):
            E[p] = (("mlat", "mlon"), f[p][:])
        for p in ("Vminx2ist", "Vmaxx2ist"):
            E[p] = (("mlat",), f[p][:])
        for p in ("Vminx3ist", "Vmaxx3ist"):
            E[p] = (("mlon",), f[p][:])

    return E


def precip(file: Path) -> xarray.Dataset:
    """
    load precipitation
    """

    with h5py.File(file.with_name("simgrid.h5"), "r") as f:
        dat = xarray.Dataset(coords={"mlon": f["/mlon"][:], "mlat": f["/mlat"][:]})

    with h5py.File(file, "r") as f:
        for k in ("Q", "E0"):
            dat[k] = (("mlat", "mlon"), f[f"/{k}p"][:])

    return dat


def frame3d_curvne(file: Path, xg: dict[str, T.Any] = None) -> xarray.Dataset:
    """
    just Ne
    """

    if not xg:
        xg = grid(file.parent, var={"x1", "x2", "x3"})

    dat = xarray.Dataset(coords={"x1": xg["x1"][2:-2], "x2": xg["x2"][2:-2], "x3": xg["x3"][2:-2]})

    p3 = (2, 1, 0)

    with h5py.File(file, "r") as f:
        dat["ne"] = (("x1", "x2", "x3"), f["/ne"][:].transpose(p3))

    return dat


def frame3d_curv(file: Path, var: set[str], xg: dict[str, T.Any] = None) -> xarray.Dataset:
    """
    curvilinear

    Parameters
    ----------

    file: pathlib.Path
        filename to read
    var: set of str
        variable(s) to read
    """

    if isinstance(var, str):
        var = [var]
    var = set(var)

    if not xg:
        xg = grid(file.parent, var={"x1", "x2", "x3"})

    dat = xarray.Dataset(coords={"x1": xg["x1"][2:-2], "x2": xg["x2"][2:-2], "x3": xg["x3"][2:-2]})

    lx = xg["lx"]

    p4n = (0, 3, 2, 1)
    p3n = (2, 1, 0)

    with h5py.File(file, "r") as f:

        p4 = p4n
        p3 = p3n

        if {"ne", "ns", "v1", "Ti"} & var:
            dat["ns"] = (("species", "x1", "x2", "x3"), f["/nsall"][:].transpose(p4))

        if {"v1", "vs1"} & var:
            dat["vs1"] = (("species", "x1", "x2", "x3"), f["/vs1all"][:].transpose(p4))

        if {"Te", "Ti", "Ts"} & var:
            dat["Ts"] = (("species", "x1", "x2", "x3"), f["/Tsall"][:].transpose(p4))

        for k in {"J1", "J2", "J3"} & var:
            dat[k] = (("x1", "x2", "x3"), f[f"/{k}all"][:].transpose(p3))

        for k in {"v2", "v3"} & var:
            dat[k] = (("x1", "x2", "x3"), f[f"/{k}avgall"][:].transpose(p3))

        if "Phi" in var:
            Phiall = f["/Phiall"][:]

            if Phiall.ndim == 1:
                if lx[1] == 1:
                    Phiall = Phiall[None, :]
                else:
                    Phiall = Phiall[:, None]

            dat["Phitop"] = (("x2", "x3"), Phiall.transpose())

    return dat


def frame3d_curvavg(file: Path, var: set[str], xg: dict[str, T.Any] = None) -> xarray.Dataset:
    """

    Parameters
    ----------
    file: pathlib.Path
        filename of this timestep of simulation output
    var: set of str
        variable(s) to read
    """

    if not xg:
        xg = grid(file.parent, var={"x1", "x2", "x3"})

    dat = xarray.Dataset(coords={"x1": xg["x1"][2:-2], "x2": xg["x2"][2:-2], "x3": xg["x3"][2:-2]})

    p3 = (2, 1, 0)

    v2n = {
        "ne": "neall",
        "v1": "v1avgall",
        "Ti": "Tavgall",
        "Te": "TEall",
        "J1": "J1all",
        "J2": "J2all",
        "J3": "J3all",
        "v2": "v2avgall",
        "v3": "v3avgall",
        "Phi": "Phiall",
    }

    if isinstance(var, str):
        var = [var]

    with h5py.File(file, "r") as f:
        var = set(var) if var else f.keys()

        for k in var:
            if k == "Phi":
                dat["Phitop"] = (("x2", "x3"), f[f"/{v2n[k]}"][:].transpose())
            else:
                dat[k] = (("x1", "x2", "x3"), f[f"/{v2n[k]}"][:].transpose(p3))

    return dat


def glow_aurmap(file: Path, xg: dict[str, T.Any] = None) -> xarray.Dataset:
    """
    read the auroral output from GLOW

    Parameters
    ----------
    file: pathlib.Path
        filename of this timestep of simulation output
    """

    if not xg:
        xg = grid(file.parents[1], var={"x2", "x3"})

    dat = xarray.Dataset(coords={"wavelength": WAVELEN, "x2": xg["x2"][2:-2], "x3": xg["x3"][2:-2]})

    p3 = (0, 2, 1)

    with h5py.File(file, "r") as h:
        dat["rayleighs"] = (("wavelength", "x2", "x3"), h["/aurora/iverout"][:].transpose(p3))

    return dat


def time(file: Path) -> datetime:
    """
    reads simulation time
    """

    try:
        with h5py.File(file, "r") as f:
            ymd = datetime(*f["/time/ymd"][:3])

            try:
                hour = f["/time/UThour"][()].item()
            except KeyError:
                hour = f["/time/UTsec"][()].item() / 3600

        t = ymd + timedelta(hours=hour)
    except KeyError:
        logging.error(f"/time group missing from {file}, getting time from filename pattern.")
        t = filename2datetime(file)

    return t
