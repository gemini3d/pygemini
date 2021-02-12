"""
HDF5 file read
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
    import h5py
except (ImportError, AttributeError):
    # must be ImportError not ModuleNotFoundError for botched HDF5 linkage
    h5py = None


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

    if h5py is None:
        raise ImportError("h5py missing or broken")

    path = find.simsize(path, ".h5")

    with h5py.File(path, "r") as f:
        if "lxs" in f:
            lxs = f["lxs"][:]
        elif "lx" in f:
            lxs = f["lx"][:]
        elif "lx1" in f:
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
            raise KeyError(f"could not find '/lxs', '/lx' or '/lx1' in {path.as_posix()}")

    return lxs


def flagoutput(file: Path, cfg: dict[str, T.Any]) -> int:
    """ detect output type """

    if h5py is None:
        raise ImportError("h5py missing or broken")

    flag = None
    with h5py.File(file, "r") as f:
        if "nsall" in f:
            # milestone or full
            flag = 1
        elif "flagoutput" in f:
            flag = f["/flagoutput"][()]
        elif "ne" in f and f["/ne"].ndim == 3:
            flag = 0
        elif "Tavgall" in f:
            flag = 2
        elif "neall" in f:
            flag = 3
    if flag is None:
        flag = cfg.get("flagoutput")

    return flag


def grid(
    file: Path, *, var: tuple[str, ...] | list[str] = None, shape: bool = False
) -> dict[str, T.Any]:
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

    Transpose on read to undo the transpose operation we had to do in write_grid C => Fortran order.
    """

    if h5py is None:
        raise ImportError("h5py missing or broken")

    xg: dict[str, T.Any] = {}

    if not file.is_file():
        file2 = find.grid(file)
        if file2 and file2.is_file():
            file = file2
        else:
            logging.error(f"{file} grid file is not present.")
            return xg

    if shape:
        with h5py.File(file, "r") as f:
            for k in f.keys():
                if f[k].ndim >= 2:
                    xg[k] = f[k].shape[::-1]
                else:
                    xg[k] = f[k].shape

        xg["lxs"] = np.array([xg["x1"], xg["x2"], xg["x3"]])
        return xg

    with h5py.File(file, "r") as f:
        if not var:
            var = f.keys()
        for k in var:
            if f[k].ndim >= 2:
                xg[k] = f[k][:].transpose()
            else:
                xg[k] = f[k][:]

    xg["lxs"] = simsize(file.with_name("simsize.h5"))

    return xg


def Efield(file: Path) -> xarray.Dataset:
    """
    load electric field
    """

    if h5py is None:
        raise ImportError("h5py missing or broken")

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

    if h5py is None:
        raise ImportError("h5py missing or broken")

    with h5py.File(file.with_name("simgrid.h5"), "r") as f:
        dat = xarray.Dataset(coords={"mlon": f["/mlon"][:], "mlat": f["/mlat"][:]})

    with h5py.File(file, "r") as f:
        for k in ("Q", "E0"):
            dat[k] = (("mlat", "mlon"), f[f"/{k}p"][:])

    return dat


def frame3d_curvne(file: Path) -> xarray.Dataset:
    """
    just Ne
    """

    if h5py is None:
        raise ImportError("h5py missing or broken")

    xg = grid(file.parent, var=("x1", "x2", "x3"))
    dat = xarray.Dataset(coords={"x1": xg["x1"][2:-2], "x2": xg["x2"][2:-2], "x3": xg["x3"][2:-2]})

    lxs = simsize(file.parent)

    if lxs[2] == 1:  # east-west
        p3 = (2, 0, 1)
    else:  # 3D or north-south, no swap
        p3 = (2, 1, 0)

    with h5py.File(file, "r") as f:
        dat["ne"] = (("x1", "x2", "x3"), f["/ne"][:].transpose(p3))

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

    if h5py is None:
        raise ImportError("h5py missing or broken")

    lxs = simsize(file.parent)

    p4s = (0, 3, 1, 2)  # only for Gemini < 0.10.0
    p3s = (2, 0, 1)
    p4n = (0, 3, 2, 1)
    p3n = (2, 1, 0)

    with h5py.File(file, "r") as f:

        if lxs[1] == 1 or file.name == "initial_conditions.h5":
            p4 = p4n
            p3 = p3n
        elif lxs[2] == 1:  # east-west
            if f["/nsall"].shape[2] == 1:  # old, Gemini < 0.10.0 data
                p4 = p4s
                p3 = p3s
            else:  # Gemini >= 0.10.0
                p4 = p4n
                p3 = p3n
        else:  # 3D
            p4 = p4n
            p3 = p3n

        if {"ne", "ns", "v1", "Ti"}.intersection(var):
            dat["ns"] = (("species", "x1", "x2", "x3"), f["/nsall"][:].transpose(p4))

        if {"v1", "vs1"}.intersection(var):
            dat["vs1"] = (("species", "x1", "x2", "x3"), f["/vs1all"][:].transpose(p4))

        if {"Te", "Ti", "Ts"}.intersection(var):
            dat["Ts"] = (("species", "x1", "x2", "x3"), f["/Tsall"][:].transpose(p4))

        for k in {"J1", "J2", "J3"}.intersection(var):
            dat[k] = (("x1", "x2", "x3"), f[f"/{k}all"][:].transpose(p3))

        for k in {"v2", "v3"}.intersection(var):
            dat[k] = (("x1", "x2", "x3"), f[f"/{k}avgall"][:].transpose(p3))

        if "Phi" in var:
            Phiall = f["/Phiall"][:]

            if Phiall.ndim == 1:
                if lxs[1] == 1:
                    Phiall = Phiall[None, :]
                else:
                    Phiall = Phiall[:, None]

            if all(Phiall.shape == lxs[1:][::-1]):
                Phiall = Phiall.transpose()  # Gemini < 0.10.0
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

    if h5py is None:
        raise ImportError("h5py missing or broken")

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

    with h5py.File(file, "r") as f:
        for k in var:
            if k == "Phi":
                dat["Phitop"] = (("x2", "x3"), f[f"/{v2n[k]}"][:].transpose())
            else:
                dat[k] = (("x1", "x2", "x3"), f[f"/{v2n[k]}"][:].transpose(p3))

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

    lxs = simsize(file.parents[1])

    if h5py is None:
        raise ImportError("h5py missing or broken")

    if lxs[2] == 1:  # east-west
        p3 = (0, 2, 1)
    else:  # 3D or north-south, no swap
        p3 = (0, 2, 1)

    with h5py.File(file, "r") as h:
        dat["rayleighs"] = (("wavelength", "x2", "x3"), h["/aurora/iverout"][:].transpose(p3))

    return dat


def time(file: Path) -> datetime:
    """
    reads simulation time
    """

    if h5py is None:
        raise ImportError("h5py missing or broken")

    with h5py.File(file, "r") as f:
        ymd = datetime(*f["/time/ymd"][:3])

        if "/time/UThour" in f:
            hour = f["/time/UThour"][()].item()
        elif "/time/UTsec" in f:
            hour = f["/time/UTsec"][()].item() / 3600
        else:
            raise KeyError(f"did not find time of day in {file}")

    t = ymd + timedelta(hours=hour)

    return t
