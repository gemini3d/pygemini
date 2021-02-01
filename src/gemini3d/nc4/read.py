"""
NetCDF4 file reading
"""

from __future__ import annotations
import xarray
from pathlib import Path
import typing as T
import numpy as np
import logging
from datetime import datetime, timedelta

from .. import find
from .. import WAVELEN

try:
    from netCDF4 import Dataset
except ImportError:
    # must be ImportError not ModuleNotFoundError for botched NetCDF4 linkage
    Dataset = None


def simsize(path: Path) -> tuple[int, ...]:
    """
    get simulation size
    """

    if Dataset is None:
        raise ImportError("netcdf missing or broken")

    path = find.simsize(path, ".nc")

    with Dataset(path, "r") as f:
        if "lxs" in f.variables:
            lxs = f["lxs"][:]
        elif "lx" in f.variables:
            lxs = f["lx"][:]
        elif "lx1" in f.variables:
            if f["lx1"].ndim > 0:
                lxs = np.array(
                    [
                        f["lx1"][:].squeeze()[()],
                        f["lx2"][:].squeeze()[()],
                        f["lx3"][:].squeeze()[()],
                    ]
                )
            else:
                lxs = np.array([f["lx1"][()], f["lx2"][()], f["lx3"][()]])
        else:
            raise KeyError(f"could not find 'lxs', 'lx' or 'lx1' in {path.as_posix()}")

    return lxs


def flagoutput(file: Path, cfg: dict[str, T.Any]) -> int:
    """ detect output type """

    if Dataset is None:
        raise ImportError("netcdf missing or broken")

    flag = None
    with Dataset(file, "r") as f:
        if "nsall" in f.variables:
            # milestone or full
            flag = 1
        elif "flagoutput" in f.variables:
            flag = f["flagoutput"][()]
        elif "ne" in f.variables and f["ne"].ndim == 3:
            flag = 0
        elif "Tavgall" in f.variables:
            flag = 2
        elif "neall" in f.variables:
            flag = 3
    if flag is None:
        flag = cfg.get("flagoutput")

    return flag


def grid(
    file: Path, *, var: tuple[str, ...] | list[str] = None, shape: bool = False
) -> dict[str, np.ndarray]:
    """
    get simulation grid

    Parameters
    ----------
    file: pathlib.Path
        filepath to simgrid
    var: list of str, optional
        read only these grid variables
    shape: bool, optional
        read only the shape of the grid instead of the data iteslf

    Returns
    -------
    grid: dict
        grid parameters
    """

    if Dataset is None:
        raise ImportError("netcdf missing or broken")

    grid: dict[str, T.Any] = {}

    if not file.is_file():
        file2 = find.grid(file)
        if file2 and file2.is_file():
            file = file2
        else:
            logging.error(f"{file} grid file is not present.")
            return grid

    if shape:
        with Dataset(file, "r") as f:
            for key in f.variables:
                grid[key] = f[key].shape

        grid["lxs"] = np.array([grid["x1"], grid["x2"], grid["x3"]])
        return grid

    with Dataset(file, "r") as f:
        if not var:
            var = f.variables
        for k in var:
            if f[k].ndim >= 2:
                grid[k] = f[k][:].transpose()
            else:
                grid[k] = f[k][:]

    grid["lxs"] = simsize(file.with_name("simsize.nc"))

    return grid


def Efield(file: Path) -> xarray.Dataset:
    """
    load electric field
    """

    if Dataset is None:
        raise ImportError("netcdf missing or broken")

    with Dataset(file.with_name("simgrid.nc"), "r") as f:
        E = xarray.Dataset(coords={"mlon": f["/mlon"][:], "mlat": f["/mlat"][:]})

    with Dataset(file, "r") as f:
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

    if Dataset is None:
        raise ImportError("netcdf missing or broken")

    with Dataset(file.with_name("simgrid.nc"), "r") as f:
        dat = xarray.Dataset(coords={"mlon": f["/mlon"][:], "mlat": f["/mlat"][:]})

    with Dataset(file, "r") as f:
        for k in ("Q", "E0"):
            dat[k] = (("mlat", "mlon"), f[f"/{k}p"][:])

    return dat


def frame3d_curvne(file: Path) -> xarray.Dataset:
    """
    just Ne
    """

    if Dataset is None:
        raise ImportError("netcdf missing or broken")

    xg = grid(file.parent, var=("x1", "x2", "x3"))
    dat = xarray.Dataset(coords={"x1": xg["x1"][2:-2], "x2": xg["x2"][2:-2], "x3": xg["x3"][2:-2]})

    with Dataset(file, "r") as f:
        dat["ne"] = (("x1", "x2", "x3"), f["/ne"][:])

    return dat


def frame3d_curv(file: Path, var: tuple[str, ...] | list[str]) -> xarray.Dataset:
    """
    curvilinear

    Parameters
    ----------

    file: pathlib.Path
        filename to read
    var: list of str
        variable(s) to read
    """

    xg = grid(file.parent, var=("x1", "x2", "x3"))
    dat = xarray.Dataset(coords={"x1": xg["x1"][2:-2], "x2": xg["x2"][2:-2], "x3": xg["x3"][2:-2]})

    if Dataset is None:
        raise ImportError("netcdf missing or broken")

    lxs = simsize(file.parent)

    p4 = (0, 3, 2, 1)
    if lxs[2] == 1:  # east-west
        p3 = (2, 0, 1)
    else:  # 3D or north-south, no swap
        p3 = (2, 1, 0)

    with Dataset(file, "r") as f:
        if {"ne", "ns", "v1", "Ti"}.intersection(var):
            dat["ns"] = (("species", "x1", "x2", "x3"), f["nsall"][:].transpose(p4))

        if {"v1", "vs1"}.intersection(var):
            dat["vs1"] = (("species", "x1", "x2", "x3"), f["vs1all"][:].transpose(p4))

        if {"Te", "Ti", "Ts"}.intersection(var):
            dat["Ts"] = (("species", "x1", "x2", "x3"), f["Tsall"][:].transpose(p4))

        for k in {"J1", "J2", "J3"}.intersection(var):
            dat[k] = (("x1", "x2", "x3"), f[f"{k}all"][:].transpose(p3))

        for k in {"v2", "v3"}.intersection(var):
            dat[k] = (("x1", "x2", "x3"), f[f"/{k}avgall"][:].transpose(p3))

        if "Phi" in var:
            Phiall = f["Phiall"][:]
            if Phiall.ndim == 2:
                Phiall = Phiall.transpose()
            elif Phiall.ndim == 1:
                if dat.x2.size == 1:
                    Phiall = Phiall[:, None]
                else:
                    Phiall = Phiall[None, :]

            dat["Phitop"] = (("x2", "x3"), Phiall)

    return dat


def frame3d_curvavg(file: Path, var: tuple[str, ...] | list[str]) -> xarray.Dataset:
    """

    Parameters
    ----------
    file: pathlib.Path
        filename of this timestep of simulation output
    var: list of str
        variable(s) to read
    """

    xg = grid(file.parent, var=("x1", "x2", "x3"))
    dat = xarray.Dataset(coords={"x1": xg["x1"][2:-2], "x2": xg["x2"][2:-2], "x3": xg["x3"][2:-2]})

    if Dataset is None:
        raise ImportError("netcdf missing or broken")

    lxs = simsize(file.parent)

    if lxs[2] == 1:  # east-west
        p3 = (2, 0, 1)
    else:  # 3D or north-south, no swap
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

    with Dataset(file, "r") as f:
        for k in var:
            if k == "Phi":
                dat["Phitop"] = (("x2", "x3"), f[v2n[k]][:].transpose())
            else:
                dat[k] = (("x1", "x2", "x3"), f[v2n[k]][:].transpose(p3))

    return dat


def glow_aurmap(file: Path) -> xarray.Dataset:
    """
    read the auroral output from GLOW

    Parameters
    ----------
    file: pathlib.Path
        filename of this timestep of simulation output
    """

    xg = grid(file.parents[1], var=("x2", "x3"))
    dat = xarray.Dataset(coords={"wavelength": WAVELEN, "x2": xg["x2"][2:-2], "x3": xg["x3"][2:-2]})

    if Dataset is None:
        raise ImportError("netcdf missing or broken")

    with Dataset(file, "r") as h:
        dat["rayleighs"] = (("wavelength", "x2", "x3"), h["iverout"][:])

    return dat


def time(file: Path) -> datetime:
    """
    reads simulation time
    """

    if Dataset is None:
        raise ImportError("netcdf missing or broken")

    with Dataset(file, "r") as f:
        ymd = datetime(*f["ymd"][:3])

        if "UThour" in f:
            hour = f["UThour"][()].item()
        elif "UTsec" in f:
            hour = f["UTsec"][()].item() / 3600
        else:
            raise KeyError(f"did not find time of day in {file}")

    t = ymd + timedelta(hours=hour)

    return t
