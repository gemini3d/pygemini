"""
struct manpage:
https://docs.python.org/3/library/struct.html#struct-format-strings
"""

from __future__ import annotations
from pathlib import Path
from datetime import datetime
import typing as T

import numpy as np
import xarray

from .config import read_ini, read_nml
from . import find
from . import matlab
from . import LSP

from .raw import read as raw_read
from .hdf5 import read as h5read
from .nc4 import read as ncread


# do NOT use lru_cache--can have weird unexpected effects with complicated setups
def config(path: Path) -> dict[str, T.Any]:
    """
    read simulation input configuration

    .nml is strongly preferred, .ini is legacy.

    Parameters
    ----------
    path: pathlib.Path
        config file path

    Returns
    -------
    params: dict
        simulation parameters from config file
    """

    file = find.config(path)

    if file.suffix == ".ini":
        P = read_ini(file)
    else:
        P = read_nml(file)

    return P


def simsize(path: Path, suffix: str = None) -> tuple[int, ...]:
    """get simulation dimensions"""

    fn = find.simsize(path, suffix=suffix)

    if fn.suffix.endswith("h5"):
        return h5read.simsize(fn)
    elif fn.suffix.endswith("nc"):
        return ncread.simsize(fn)
    elif fn.suffix.endswith("dat"):
        return raw_read.simsize(fn)
    elif fn.suffix.endswith("mat"):
        return matlab.simsize(fn)
    else:
        raise ValueError("unknown simsize file type")


def grid(
    path: Path, *, var: set[str] = None, file_format: str = None, shape: bool = False
) -> dict[str, T.Any]:
    """
    get simulation grid

    Parameters
    ----------

    path: pathlib.Path
        path to simgrid.*
    var: set of str
        read only these grid variables
    file_format: str, optional
        force .h5, .nc, .dat (debugging)
    shape: bool, optional
        read only the shape of the grid instead of the data iteslf
    """

    fn = find.grid(path, suffix=file_format)

    if not file_format:
        file_format = fn.suffix

    if file_format.endswith("dat"):
        xg = raw_read.grid(fn.with_suffix(".dat"), shape=shape)
    elif file_format.endswith("h5"):
        xg = h5read.grid(fn.with_suffix(".h5"), var=var, shape=shape)
    elif file_format.endswith("nc"):
        xg = ncread.grid(fn.with_suffix(".nc"), var=var, shape=shape)
    elif file_format.endswith("mat"):
        xg = matlab.grid(fn.with_suffix(".mat"), shape=shape)
    else:
        raise ValueError(f"Unknown file type {fn}")

    xg["filename"] = fn

    return xg


def data(
    fn: Path,
    var: set[str] = None,
    *,
    file_format: str = "",
    cfg: dict[str, T.Any] = None,
    xg: dict[str, T.Any] = None,
) -> xarray.Dataset:
    """
    knowing the filename for a simulation time step, read the data for that time step

    Parameters
    ----------
    fn: pathlib.Path
        filename for this timestep
    var: set of set
        variables to use
    file_format: str
        specify file extension of data files
    cfg: dict
        to avoid reading config.nml
    xg: dict
        to avoid reading simgrid.*, useful to save time when reading data files in a loop

    Returns
    -------
    dat: xarray.Dataset
        simulation outputs
    """

    if not var:
        var = {"ne", "Ti", "Te", "v1", "v2", "v3", "J1", "J2", "J3", "Phi"}

    if isinstance(var, str):
        var = [var]
    var = set(var)

    fn = Path(fn).expanduser()

    if not cfg:
        cfg = config(fn.parent)

    if not file_format:
        file_format = cfg.get("file_format", fn.suffix)

    if file_format.endswith("dat"):
        flag = cfg.get("flagoutput")
        if flag == 3:
            dat = raw_read.frame3d_curvne(fn, xg)
        elif flag == 1:
            dat = raw_read.frame3d_curv(fn, xg)
        elif flag == 2:
            dat = raw_read.frame3d_curvavg(fn, xg)
        else:
            raise ValueError(f"Unsure how to read {fn} with flagoutput {flag}")
    elif file_format.endswith("h5"):
        flag = h5read.flagoutput(fn, cfg)

        if flag == 3:
            dat = h5read.frame3d_curvne(fn, xg)
        elif flag == 1:
            dat = h5read.frame3d_curv(fn, var, xg)
        elif flag == 2:
            dat = h5read.frame3d_curvavg(fn, var, xg)
        else:
            raise ValueError(f"Unsure how to read {fn} with flagoutput {flag}")
    elif file_format.endswith("nc"):
        flag = ncread.flagoutput(fn, cfg)

        if flag == 3:
            dat = ncread.frame3d_curvne(fn, xg)
        elif flag == 1:
            dat = ncread.frame3d_curv(fn, var, xg)
        elif flag == 2:
            dat = ncread.frame3d_curvavg(fn, var, xg)
        else:
            raise ValueError(f"Unsure how to read {fn} with flagoutput {flag}")
    else:
        raise ValueError(f"Unknown file type {fn}")

    lx = (dat.dims["x1"], dat.dims["x2"], dat.dims["x3"])

    # %% Derived variables
    if flag == 1:
        if {"ne", "v1", "Ti", "Te"} & var:
            dat["ne"] = (("x1", "x2", "x3"), dat["ns"][LSP - 1, :, :, :].data)
            # np.any() in case neither is an np.ndarray
            if dat["ns"].shape[0] != LSP or not np.array_equal(dat["ns"].shape[1:], lx):
                raise ValueError(
                    f"may have wrong permutation on read. lx: {lx}  ns x1,x2,x3: {dat['ns'].shape}"
                )
        if "v1" in var:
            dat["v1"] = (
                ("x1", "x2", "x3"),
                (dat["ns"][:6, :, :, :] * dat["vs1"][:6, :, :, :]).sum(axis=0).data
                / dat["ne"].data,
            )
        if "Ti" in var:
            dat["Ti"] = (
                ("x1", "x2", "x3"),
                (dat["ns"][:6, :, :, :] * dat["Ts"][:6, :, :, :]).sum(axis=0).data / dat["ne"].data,
            )
        if "Te" in var:
            dat["Te"] = (("x1", "x2", "x3"), dat["Ts"][LSP - 1, :, :, :].data)

        if "J1" in var:
            # np.any() in case neither is an np.ndarray
            if np.any(dat["J1"].shape != lx):
                raise ValueError("J1 may have wrong permutation on read")

    if "time" not in dat:
        dat = dat.assign_coords({"time": time(fn)})

    return dat


def glow(fn: Path) -> xarray.Dataset:

    fmt = fn.suffix

    if fmt.endswith("h5"):
        dat = h5read.glow_aurmap(fn)
    elif fmt.endswith("nc"):
        dat = ncread.glow_aurmap(fn)
    elif fmt.endswith("dat"):
        dat = raw_read.glow_aurmap(fn)
    else:
        raise ValueError(f"Unknown file type {fn}")

    return dat


def Efield(fn: Path, *, file_format: str = None) -> xarray.Dataset:
    """load Efield data "Efield_inputs"

    Parameters
    ----------
    fn: pathlib.Path
        filename for this timestep

    Returns
    -------
    dat: dict of np.ndarray
        electric field
    """

    fn = Path(fn).expanduser().resolve(strict=True)

    if not file_format:
        file_format = fn.suffix

    if file_format.endswith("h5"):
        E = h5read.Efield(fn)
    elif file_format.endswith("nc"):
        E = ncread.Efield(fn)
    elif file_format.endswith("dat"):
        E = raw_read.Efield(fn)
    else:
        raise ValueError(f"Unknown file type {fn}")

    return E


def precip(fn: Path, *, file_format: str = None) -> xarray.Dataset:
    """load precipitation to disk

    Parameters
    ----------
    fn: pathlib.Path
        path to precipitation file
    file_format: str
        file format to read

    Returns
    -------
    dat: dict
        precipitation
    """

    fn = Path(fn).expanduser().resolve(strict=True)

    if not file_format:
        file_format = fn.suffix

    if file_format.endswith("h5"):
        dat = h5read.precip(fn)
    elif file_format.endswith("nc"):
        dat = ncread.precip(fn)
    else:
        raise ValueError(f"unknown file format {file_format}")

    return dat


def frame(
    simdir: Path, time: datetime, *, var: set[str] = None, file_format: str = ""
) -> xarray.Dataset:
    """
    load a frame of simulation data, automatically selecting the correct
    functions based on simulation parameters

    Parameters
    ----------
    simdir: pathlib.Path
        top-level directory of simulation output
    time: datetime.datetime
        time to load from simulation output
    var: set of str
        variable(s) to read
    file_format: str, optional
        "hdf5", "nc" for hdf5 or netcdf4 respectively

    Returns
    -------
    dat: xarray.Dataset
        simulation output for this time step
    """

    return data(
        find.frame(simdir, time, file_format=file_format),
        var=var,
        file_format=file_format,
    )


def time(file: Path) -> datetime:
    """
    read simulation time of a file
    """

    if file.suffix.endswith("h5"):
        t = h5read.time(file)
    elif file.suffix.endswith("nc"):
        t = ncread.time(file)
    else:
        raise ValueError(f"unknown file format {file.suffix}")

    return t


def get_lxs(xg: dict[str, T.Any]) -> tuple[int, int, int]:

    lx = None
    for k in ("lx", "lxs", "lx1"):
        if k in xg:
            if k == "lx1":
                lx = [xg["lx1"], xg["lx2"], xg["lx3"]]
                break
            else:
                lx = xg[k]

    if lx is None:
        raise IndexError("Did not find grid size")

    return lx[0], lx[1], lx[2]
