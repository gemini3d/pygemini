"""
NetCDF4 file reading
"""

from pathlib import Path
import typing as T
import logging
import numpy as np
from datetime import datetime, timedelta

from .. import find


try:
    from netCDF4 import Dataset
except ImportError:
    # must be ImportError not ModuleNotFoundError for botched NetCDF4 linkage
    Dataset = None


def simsize(path: Path) -> T.Tuple[int, ...]:
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


def flagoutput(file: Path, cfg: T.Dict[str, T.Any]) -> int:

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


def grid(file: Path, shape: bool = False) -> T.Dict[str, np.ndarray]:
    """
    get simulation grid

    Parameters
    ----------
    file: pathlib.Path
        filepath to simgrid
    shape: bool, optional
        read only the shape of the grid instead of the data iteslf

    Returns
    -------
    grid: dict
        grid parameters
    """

    if Dataset is None:
        raise ImportError("netcdf missing or broken")

    grid: T.Dict[str, T.Any] = {}

    if not file.is_file():
        logging.error(f"{file} grid file is not present.")
        return grid

    if shape:
        with Dataset(file, "r") as f:
            for key in f.variables:
                grid[key] = f[key].shpae

        grid["lxs"] = np.array([grid["x1"], grid["x2"], grid["x3"]])

        return grid

    with Dataset(file, "r") as f:
        for k in f.variables:
            if f[k].ndim >= 2:
                grid[k] = f[k][:].transpose()
            else:
                grid[k] = f[k][:]

    grid["lxs"] = simsize(file.with_name("simsize.nc"))
    # FIXME: line below not always work. Why not?
    # grid["lxs"] = np.array([grid["x1"].size, grid["x2"].size, grid["x3"].size])

    return grid


def Efield(file: Path) -> T.Dict[str, T.Any]:
    """
    load electric field
    """

    # NOT the whole sim simsize
    # with Dataset(file.with_name("simsize.nc") , "r") as f:
    #     E["llon"] = f["/llon"][()]
    #     E["llat"] = f["/llat"][()]

    if Dataset is None:
        raise ImportError("netcdf missing or broken")

    with Dataset(file.with_name("simgrid.nc"), "r") as f:
        E = {"mlon": f["/mlon"][:], "mlat": f["/mlat"][:]}

    with Dataset(file, "r") as f:
        E["flagdirich"] = f["flagdirich"]
        for p in ("Exit", "Eyit", "Vminx1it", "Vmaxx1it"):
            E[p] = (("x2", "x3"), f[p][:])
        for p in ("Vminx2ist", "Vmaxx2ist"):
            E[p] = (("x2",), f[p][:])
        for p in ("Vminx3ist", "Vmaxx3ist"):
            E[p] = (("x3",), f[p][:])

    return E


def precip(file: Path) -> T.Dict[str, T.Any]:
    # with Dataset(file.with_name("simsize.nc"), "r") as f:
    #     dat["llon"] = f["/llon"][()]
    #     dat["llat"] = f["/llat"][()]

    if Dataset is None:
        raise ImportError("netcdf missing or broken")

    with Dataset(file.with_name("simgrid.nc"), "r") as f:
        dat = {"mlon": f["/mlon"][:], "mlat": f["/mlat"][:]}

    with Dataset(file, "r") as f:
        for k in ("Q", "E0"):
            dat[k] = f[f"/{k}p"][:]

    return dat


def frame3d_curvne(file: Path) -> T.Dict[str, T.Any]:
    """
    just Ne
    """

    if Dataset is None:
        raise ImportError("netcdf missing or broken")

    dat: T.Dict[str, T.Any] = {}

    with Dataset(file, "r") as f:
        dat["ne"] = (("x1", "x2", "x3"), f["/ne"][:])

    return dat


def frame3d_curv(file: Path, var: T.Sequence[str]) -> T.Dict[str, T.Any]:
    """

    Parameters
    ----------

    file: pathlib.Path
        filename of this timestep of simulation output
    var: list of str
        variable(s) to read
    """

    #    grid = readgrid(file.parent)
    #    dat = xarray.Dataset(
    #        coords={"x1": grid["x1"][2:-2], "x2": grid["x2"][2:-2], "x3": grid["x3"][2:-2]}
    #    )

    if Dataset is None:
        raise ImportError("netcdf missing or broken")

    lxs = simsize(file.parent)

    dat: T.Dict[str, T.Any] = {}

    if lxs[2] == 1:  # east-west
        p4 = (0, 3, 1, 2)
        p3 = (2, 0, 1)
    else:  # 3D or north-south, no swap
        p4 = (0, 3, 2, 1)
        p3 = (2, 1, 0)

    with Dataset(file, "r") as f:
        if {"ne", "ns", "v1", "Ti"}.intersection(var):
            dat["ns"] = (("lsp", "x1", "x2", "x3"), f["nsall"][:].transpose(p4))

        if {"v1", "vs1"}.intersection(var):
            dat["vs1"] = (("lsp", "x1", "x2", "x3"), f["vs1all"][:].transpose(p4))

        if {"Te", "Ti", "Ts"}.intersection(var):
            dat["Ts"] = (("lsp", "x1", "x2", "x3"), f["Tsall"][:].transpose(p4))

        for k in {"J1", "J2", "J3"}.intersection(var):
            dat[k] = (("x1", "x2", "x3"), f[f"{k}all"][:].transpose(p3))

        for k in {"v2", "v3"}.intersection(var):
            dat[k] = (("x1", "x2", "x3"), f[f"/{k}avgall"][:].transpose(p3))

        if "Phiall" in f:
            dat["Phitop"] = (("x2", "x3"), f["Phiall"][:].transpose())

    return dat


def frame3d_curvavg(file: Path, var: T.Sequence[str]) -> T.Dict[str, T.Any]:
    """

    Parameters
    ----------
    file: pathlib.Path
        filename of this timestep of simulation output
    var: list of str
        variable(s) to read
    """
    #    grid = readgrid(file.parent)
    #    dat = xarray.Dataset(
    #        coords={"x1": grid["x1"][2:-2], "x2": grid["x2"][2:-2], "x3": grid["x3"][2:-2]}
    #    )

    if Dataset is None:
        raise ImportError("netcdf missing or broken")

    lxs = simsize(file.parent)

    dat: T.Dict[str, T.Any] = {}

    if lxs[2] == 1:  # east-west
        p3 = (2, 0, 1)
    else:  # 3D or north-south, no swap
        p3 = (2, 1, 0)

    with Dataset(file, "r") as f:
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

            dat[j] = (("x1", "x2", "x3"), f[k][:].transpose(p3))

            if not np.array_equal(dat[j][1].shape, lxs):
                raise ValueError(f"simsize {lxs} does not match {k} {j} shape {dat[j][1].shape}")

        dat["Phitop"] = (("x2", "x3"), f["Phiall"][:].transpose())

    return dat


def glow_aurmap(file: Path) -> T.Dict[str, T.Any]:
    """
    read the auroral output from GLOW

    Parameters
    ----------
    file: pathlib.Path
        filename of this timestep of simulation output
    """

    if Dataset is None:
        raise ImportError("netcdf missing or broken")

    with Dataset(file, "r") as h:
        dat = {"rayleighs": (("wavelength", "x2", "x3"), h["iverout"][:])}

    return dat


def time(file: Path) -> np.ndarray:
    """
    reads simulation time
    """

    if Dataset is None:
        raise ImportError("netcdf missing or broken")

    with Dataset(file, "r") as f:
        ymd = datetime(*f["ymd"][:3])

        if "UThour" in f:
            hour = f["UThour"][()]
        elif "UTsec" in f:
            hour = f["UTsec"][()] / 3600
        else:
            raise KeyError(f"did not find time of day in {file}")

    t = ymd + timedelta(hours=hour)

    return t
