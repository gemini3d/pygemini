"""
NetCDF4 file reading
"""

from pathlib import Path
import typing as T
import logging
import numpy as np
from datetime import datetime, timedelta

from .. import LSP
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


def flagoutput(fn: Path, cfg: T.Dict[str, T.Any]) -> int:

    with Dataset(fn, "r") as f:
        if "flagoutput" in f.variables:
            flag = f["flagoutput"][()]
        elif "flagoutput" in cfg:
            flag = cfg["flagoutput"]
        else:
            if "ne" in f.variables and f["ne"].ndim == 3:
                flag = 0
            elif "nsall" in f.variables and f["nsall"].ndim == 4:
                flag = 1
            elif "neall" in f.variables and f["neall"].ndim == 3:
                flag = 2
            else:
                raise ValueError(f"please specify flagoutput in config.nml or {fn}")

    return flag


def grid(fn: Path, shape: bool = False) -> T.Dict[str, np.ndarray]:
    """
    get simulation grid

    Parameters
    ----------
    fn: pathlib.Path
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

    if not fn.is_file():
        logging.error(f"{fn} grid file is not present.")
        return grid

    if shape:
        with Dataset(fn, "r") as f:
            for key in f.variables:
                grid[key] = f[key].shpae

        grid["lxs"] = np.array([grid["x1"], grid["x2"], grid["x3"]])

        return grid

    with Dataset(fn, "r") as f:
        for key in f.variables:
            grid[key] = f[key][:]

    grid["lxs"] = simsize(fn.with_name("simsize.nc"))
    # FIXME: line below not always work. Why not?
    # grid["lxs"] = np.array([grid["x1"].size, grid["x2"].size, grid["x3"].size])

    return grid


def state(fn: Path) -> T.Dict[str, T.Any]:
    """
    load initial condition data
    """

    if Dataset is None:
        raise ImportError("netcdf missing or broken")

    with Dataset(fn, "r") as f:
        return {"ns": f["/nsall"][:], "vs": f["/vs1all"][:], "Ts": f["/Tsall"][:]}


def Efield(fn: Path) -> T.Dict[str, T.Any]:
    """
    load electric field
    """

    # NOT the whole sim simsize
    # with Dataset(fn.with_name("simsize.nc") , "r") as f:
    #     E["llon"] = f["/llon"][()]
    #     E["llat"] = f["/llat"][()]

    if Dataset is None:
        raise ImportError("netcdf missing or broken")

    with Dataset(fn.with_name("simgrid.nc"), "r") as f:
        E = {"mlon": f["/mlon"][:], "mlat": f["/mlat"][:]}

    with Dataset(fn, "r") as f:
        E["flagdirich"] = f["flagdirich"]
        for p in ("Exit", "Eyit", "Vminx1it", "Vmaxx1it"):
            E[p] = (("x2", "x3"), f[p][:])
        for p in ("Vminx2ist", "Vmaxx2ist"):
            E[p] = (("x2",), f[p][:])
        for p in ("Vminx3ist", "Vmaxx3ist"):
            E[p] = (("x3",), f[p][:])

    return E


def precip(fn: Path) -> T.Dict[str, T.Any]:
    # with Dataset(fn.with_name("simsize.nc"), "r") as f:
    #     dat["llon"] = f["/llon"][()]
    #     dat["llat"] = f["/llat"][()]

    if Dataset is None:
        raise ImportError("netcdf missing or broken")

    with Dataset(fn.with_name("simgrid.nc"), "r") as f:
        dat = {"mlon": f["/mlon"][:], "mlat": f["/mlat"][:]}

    with Dataset(fn, "r") as f:
        for k in ("Q", "E0"):
            dat[k] = f[f"/{k}p"][:]

    return dat


def frame3d_curvne(fn: Path) -> T.Dict[str, T.Any]:
    """
    just Ne
    """

    if Dataset is None:
        raise ImportError("netcdf missing or broken")

    dat: T.Dict[str, T.Any] = {}

    with Dataset(fn, "r") as f:
        dat["ne"] = (("x1", "x2", "x3"), f["/ne"][:])

    return dat


def frame3d_curv(fn: Path, vars: T.Sequence[str]) -> T.Dict[str, T.Any]:
    """

    Parameters
    ----------

    fn: pathlib.Path
        filename of this timestep of simulation output
    """

    #    grid = readgrid(fn.parent)
    #    dat = xarray.Dataset(
    #        coords={"x1": grid["x1"][2:-2], "x2": grid["x2"][2:-2], "x3": grid["x3"][2:-2]}
    #    )

    if Dataset is None:
        raise ImportError("netcdf missing or broken")

    lxs = simsize(fn.parent)

    dat: T.Dict[str, T.Any] = {}

    if lxs[2] == 1:  # east-west
        p4 = (0, 3, 1, 2)
        p3 = (2, 0, 1)
    else:  # 3D or north-south, no swap
        p4 = (0, 3, 2, 1)
        p3 = (2, 1, 0)

    with Dataset(fn, "r") as f:
        ns = f["nsall"][:].transpose(p4)
        # np.any() in case neither is an np.ndarray
        if ns.shape[0] != 7 or np.any(ns.shape[1:] != lxs):
            raise ValueError(
                f"may have wrong permutation on read. lxs: {lxs}  ns x1,x2,x3: {ns.shape}"
            )
        dat["ns"] = (("lsp", "x1", "x2", "x3"), ns)
        vs = f["vs1all"][:].transpose(p4)
        dat["vs"] = (("lsp", "x1", "x2", "x3"), vs)
        Ts = f["Tsall"][:].transpose(p4)
        dat["Ts"] = (("lsp", "x1", "x2", "x3"), Ts)

        dat["Te"] = (("x1", "x2", "x3"), Ts[LSP - 1, :, :, :])

        dat["J1"] = (("x1", "x2", "x3"), f["J1all"][:].transpose(p3))
        # np.any() in case neither is an np.ndarray
        if np.any(dat["J1"][1].shape != lxs):
            raise ValueError("may have wrong permutation on read")
        dat["J2"] = (("x1", "x2", "x3"), f["J2all"][:].transpose(p3))
        dat["J3"] = (("x1", "x2", "x3"), f["J3all"][:].transpose(p3))

        dat["v2"] = (("x1", "x2", "x3"), f["v2avgall"][:].transpose(p3))
        dat["v3"] = (("x1", "x2", "x3"), f["v3avgall"][:].transpose(p3))

        dat["Phitop"] = (("x2", "x3"), f["Phiall"][:].transpose())

    return dat


def frame3d_curvavg(fn: Path, vars: T.Sequence[str]) -> T.Dict[str, T.Any]:
    """

    Parameters
    ----------
    fn: pathlib.Path
        filename of this timestep of simulation output
    """
    #    grid = readgrid(fn.parent)
    #    dat = xarray.Dataset(
    #        coords={"x1": grid["x1"][2:-2], "x2": grid["x2"][2:-2], "x3": grid["x3"][2:-2]}
    #    )

    if Dataset is None:
        raise ImportError("netcdf missing or broken")

    lxs = simsize(fn.parent)

    dat: T.Dict[str, T.Any] = {}
    p3 = (2, 0, 1)

    with Dataset(fn, "r") as f:
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

            if dat[j][1].shape != lxs:
                raise ValueError(f"simsize {lxs} does not match {k} {j} shape {dat[j][1].shape}")

    return dat


def glow_aurmap(fn: Path) -> T.Dict[str, T.Any]:
    """
    read the auroral output from GLOW

    Parameters
    ----------
    fn: pathlib.Path
        filename of this timestep of simulation output
    """

    if Dataset is None:
        raise ImportError("netcdf missing or broken")

    with Dataset(fn, "r") as h:
        dat = {"rayleighs": (("wavelength", "x2", "x3"), h["iverout"][:])}

    return dat


def time(file: Path) -> np.ndarray:
    """
    reads simulation time
    """

    if Dataset is None:
        raise ImportError("netcdf missing or broken")

    with Dataset(file, "r") as f:
        ymd = datetime(*f["ymd"][:2])

        if "UThour" in f:
            hour = f["UThour"][()]
        elif "UTsec" in f:
            hour = f["UTsec"][()] / 3600
        else:
            raise KeyError(f"did not find time of day in {file}")

    t = ymd + timedelta(hours=hour)

    return t
