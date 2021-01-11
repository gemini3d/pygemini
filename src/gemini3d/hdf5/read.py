"""
HDF5 file read
"""

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


def simsize(path: Path) -> T.Tuple[int, ...]:
    """
    get simulation size
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


def flagoutput(file: Path, cfg: T.Dict[str, T.Any]) -> int:
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
    file: Path, *, var: T.Sequence[str] = None, shape: bool = False
) -> T.Dict[str, np.ndarray]:
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

    grid: T.Dict[str, T.Any] = {}

    if not file.is_file():
        file2 = find.grid(file)
        if file2 and file2.is_file():
            file = file2
        else:
            logging.error(f"{file} grid file is not present.")
            return grid

    if shape:
        with h5py.File(file, "r") as f:
            for k in f.keys():
                if f[k].ndim >= 2:
                    grid[k] = f[k].shape[::-1]
                else:
                    grid[k] = f[k].shape

        grid["lxs"] = np.array([grid["x1"], grid["x2"], grid["x3"]])
        return grid

    with h5py.File(file, "r") as f:
        if not var:
            var = f.keys()
        for k in var:
            if f[k].ndim >= 2:
                grid[k] = f[k][:].transpose()
            else:
                grid[k] = f[k][:]

    grid["lxs"] = simsize(file.with_name("simsize.h5"))

    return grid


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


def frame3d_curv(file: Path, var: T.Sequence[str]) -> xarray.Dataset:
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

    p4s = (0, 3, 1, 2)
    p3s = (2, 0, 1)
    p4n = (0, 3, 2, 1)
    p3n = (2, 1, 0)

    if lxs[1] == 1 or file.name == "initial_conditions.h5":
        p4 = p4n
        p3 = p3n
    elif lxs[2] == 1:  # east-west
        p4 = p4s
        p3 = p3s
    else:  # 3D
        p4 = p4n
        p3 = p3n

    with h5py.File(file, "r") as f:
        if {"ne", "ns", "v1", "Ti"}.intersection(var):
            dat["ns"] = (("lsp", "x1", "x2", "x3"), f["/nsall"][:].transpose(p4))

        if {"v1", "vs1"}.intersection(var):
            dat["vs1"] = (("lsp", "x1", "x2", "x3"), f["/vs1all"][:].transpose(p4))

        if {"Te", "Ti", "Ts"}.intersection(var):
            dat["Ts"] = (("lsp", "x1", "x2", "x3"), f["/Tsall"][:].transpose(p4))

        for k in {"J1", "J2", "J3"}.intersection(var):
            dat[k] = (("x1", "x2", "x3"), f[f"/{k}all"][:].transpose(p3))

        for k in {"v2", "v3"}.intersection(var):
            dat[k] = (("x1", "x2", "x3"), f[f"/{k}avgall"][:].transpose(p3))

        if "Phiall" in f:
            Phiall = f["/Phiall"][:]

            if Phiall.ndim == 1:
                if lxs[1] == 1:
                    Phiall = Phiall[None, :]
                else:
                    Phiall = Phiall[:, None]

            if (lxs[1] == 1 and not file.name == "initial_conditions.h5") or (
                lxs[1] != 1 and lxs[2] != 1
            ):
                Phiall = Phiall.transpose()
            dat["Phitop"] = (("x2", "x3"), Phiall)

    return dat


def frame3d_curvavg(file: Path, var: T.Sequence[str]) -> xarray.Dataset:
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

    with h5py.File(file, "r") as f:
        for j, k in zip(
            ("ne", "v1", "Ti", "Te", "J1", "J2", "J3", "v2", "v3"),
            (
                "neall",
                "v1avgall",
                "Tavgall",
                "TEall",
                "J1all",
                "J2all",
                "J3all",
                "v2avgall",
                "v3avgall",
            ),
        ):

            dat[j] = (("x1", "x2", "x3"), f[f"/{k}"][:].transpose(p3))

            if not np.array_equal(dat[j].shape, lxs):
                raise ValueError(f"simsize {lxs} does not match {k} {j} shape {dat[j].shape}")

        dat["Phitop"] = (("x2", "x3"), f["/Phiall"][:].transpose())

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


def time(file: Path) -> np.ndarray:
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
