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

from .config import read_nml
from . import find
from . import LSP

from .hdf5 import read as h5read


# do NOT use lru_cache--can have weird unexpected effects with complicated setups
def config(path: Path) -> dict[str, T.Any]:
    """
    read simulation input configuration from .nml Fortran namelist file

    Parameters
    ----------
    path: pathlib.Path
        config file path

    Returns
    -------
    params: dict
        simulation parameters from config file
    """

    return read_nml(find.config(path))


def simsize(path: Path) -> tuple[int, ...]:
    """get simulation dimensions"""

    return h5read.simsize(find.simsize(path))


def grid(
    path: Path, *, var: set[str] | None = None, shape: bool = False
) -> dict[str, T.Any]:
    """
    get simulation grid

    Parameters
    ----------

    path: pathlib.Path
        path to simgrid.*
    var: set of str
        read only these grid variables
    shape: bool, optional
        read only the shape of the grid instead of the data iteslf
    """

    fn = find.grid(path)

    xg = h5read.grid(fn, var=var, shape=shape)

    xg["filename"] = fn

    return xg


def frame(
    path: Path,
    time: datetime | None = None,
    var: set[str] | None = None,
    *,
    cfg: dict[str, T.Any] | None = None,
    xg: dict[str, T.Any] | None = None,
) -> xarray.Dataset:
    """
    load a frame of simulation data, automatically selecting the correct
    functions based on simulation parameters

    Parameters
    ----------
    file: pathlib.Path
        filename for this timestep
    time: datetime.datetime
        time to load from simulation output
    var: set of str
        variable(s) to read
    cfg: dict
        to avoid reading config.nml
    xg: dict
        to avoid reading simgrid.*, useful to save time when reading data files in a loop
    """

    # %% default variables
    if not var:
        var = {"ne", "Ti", "Te", "v1", "v2", "v3", "J1", "J2", "J3", "Phi"}

    if isinstance(var, str):
        var = [var]
    var = set(var)

    # %% file or directory
    path = Path(path).expanduser()
    if path.is_dir():
        if time is None:
            raise ValueError(
                f"must specify time when giving directory {path} instead of file"
            )
        path = find.frame(path, time)
    # %% config file needed
    if not cfg:
        cfg = config(path.parent)

    flag = h5read.flagoutput(path, cfg)

    if flag == 3:
        dat = h5read.frame3d_curvne(path, xg)
    elif flag == 1:
        dat = h5read.frame3d_curv(path, var, xg)
    elif flag == 2:
        dat = h5read.frame3d_curvavg(path, var, xg)
    else:
        raise ValueError(f"Unsure how to read {path} with flagoutput {flag}")

    dat.attrs["filename"] = path

    dat.update(derive(dat, var, flag))

    return dat


def derive(dat: xarray.Dataset, var: set[str], flag: int) -> xarray.Dataset:
    lx = (dat.dims["x1"], dat.dims["x2"], dat.dims["x3"])

    # %% Derived variables
    if flag == 1:
        if {"ne", "v1", "Ti"} & var:
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
                (dat["ns"][:6, :, :, :] * dat["Ts"][:6, :, :, :]).sum(axis=0).data
                / dat["ne"].data,
            )
        if "Te" in var:
            dat["Te"] = (("x1", "x2", "x3"), dat["Ts"][LSP - 1, :, :, :].data)

        if "J1" in var:
            # np.any() in case neither is an np.ndarray
            if np.any(dat["J1"].shape != lx):
                raise ValueError("J1 may have wrong permutation on read")

    if "time" not in dat:
        dat = dat.assign_coords({"time": time(dat.filename)})

    return dat


def glow(fn: Path) -> xarray.Dataset:
    """read GLOW data"""
    return h5read.glow_aurmap(fn)


def Efield(fn: Path) -> xarray.Dataset:
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

    return h5read.Efield(fn)


def precip(fn: Path) -> xarray.Dataset:
    """load precipitation to disk

    Parameters
    ----------
    fn: pathlib.Path
        path to precipitation file

    Returns
    -------
    dat: dict
        precipitation
    """

    fn = Path(fn).expanduser().resolve(strict=True)

    return h5read.precip(fn)


def time(file: Path) -> datetime:
    """
    read simulation time of a file
    """

    return h5read.time(file)


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
