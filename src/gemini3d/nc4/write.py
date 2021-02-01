"""
NetCDF4 file writing
"""

from __future__ import annotations
import xarray
import typing as T
import numpy as np
from datetime import datetime
from pathlib import Path
import logging

from ..utils import datetime2ymd_hourdec, to_datetime
from .. import LSP

try:
    from netCDF4 import Dataset
except ImportError:
    # must be ImportError not ModuleNotFoundError for botched NetCDF4 linkage
    Dataset = None


def state(fn: Path, dat: xarray.Dataset):
    """
     WRITE STATE VARIABLE DATA.
    NOTE THAT WE don't write ANY OF THE ELECTRODYNAMIC
    VARIABLES SINCE THEY ARE NOT NEEDED TO START THINGS
    UP IN THE FORTRAN CODE.

    INPUT ARRAYS SHOULD BE TRIMMED TO THE CORRECT SIZE
    I.E. THEY SHOULD NOT INCLUDE GHOST CELLS
    """

    if Dataset is None:
        raise ImportError("pip install netcdf4")

    time = dat.time

    logging.info(f"state: {fn} {time}")

    with Dataset(fn, "w") as f:
        f.createDimension("ymd", 3)
        g = f.createVariable("ymd", np.int32, "ymd")
        g[:] = [time.year, time.month, time.day]

        g = f.createVariable("UTsec", np.float32)
        g[:] = time.hour * 3600 + time.minute * 60 + time.second + time.microsecond / 1e6
        f.createDimension("species", 7)
        f.createDimension("x1", dat.x1.size)
        f.createDimension("x2", dat.x2.size)
        f.createDimension("x3", dat.x3.size)

        for k in {"ns", "vs1", "Ts"}:
            if k in dat.data_vars:
                _write_var(f, f"{k}all", dat[k])

        if "Phitop" in dat.data_vars:
            _write_var(f, "Phiall", dat["Phitop"])


def data(fn: Path, dat: xarray.Dataset, xg: dict[str, T.Any]):
    """
    write simulation data
    e.g. for converting a file format from a simulation
    """

    if Dataset is None:
        raise ImportError("pip install netcdf4")

    logging.info(f"write_data: {fn}")

    if "ns" in dat:
        shape = dat["ns"].shape
    elif "ne" in dat:
        shape = dat["ne"].shape
    else:
        raise ValueError("what variable should I use to determine dimensions?")

    with Dataset(fn, "w") as f:

        if len(shape) == 4:
            dims = ["species", "x1", "x2", "x3"]
            f.createDimension(dims[0], LSP)
            f.createDimension(dims[1], shape[1])
            f.createDimension(dims[2], shape[2])
            f.createDimension(dims[3], shape[3])
        elif len(shape) == 3:
            dims = ["x1", "x2", "x3"]
            f.createDimension(dims[0], shape[0])
            f.createDimension(dims[1], shape[1])
            f.createDimension(dims[2], shape[2])
        else:
            raise ValueError("unknown how to handle non 3-D or 4-D array")

        # set dimension values
        if xg:
            Ng = 4  # number of ghost cells
            for k in dims[-3:]:
                if xg[k].size == f.dimensions[k].size + Ng:
                    # omit ghost cells
                    x = xg[k][2:-2]
                elif xg[k].size == f.dimensions[k].size:
                    x = xg[k]
                else:
                    raise ValueError(f"{k}:  {xg[k].size} != {f.dimensions[k].size}")
                _write_var(f, name=k, value=x, dims=k)

        for k in ["ns", "vs1", "Ts"]:
            if k not in dat:
                continue
            _write_var(f, k, dat[k], dims)

        for k in ["ne", "v1", "Ti", "Te", "J1", "J2", "J3", "v2", "v3"]:
            if k not in dat:
                continue
            _write_var(f, k, dat[k], dims)

        if "Phitop" in dat:
            _write_var(f, "Phiall", dat["Phitop"], dims[-2:])


def grid(size_fn: Path, grid_fn: Path, xg: dict[str, T.Any]):
    """writes grid to disk

    Parameters
    ----------

    size_fn: pathlib.Path
        file to write
    grid_fn: pathlib.Path
        file to write
    xg: dict
        grid values
    """

    if Dataset is None:
        raise ImportError("pip install netcdf4")

    logging.info(f"grid: {size_fn}")

    with Dataset(size_fn, "w") as f:
        f.createDimension("length", len(xg["lx"]))
        g = f.createVariable("lx", np.int32, ("length",))
        g[:] = xg["lx"]

    logging.info(f"write_grid: {grid_fn}")
    Ng = 4  # number of ghost cells

    with Dataset(grid_fn, "w") as f:
        f.createDimension("x1ghost", xg["lx"][0] + Ng)
        f.createDimension("x1d", xg["lx"][0] + Ng - 1)
        f.createDimension("x1i", xg["lx"][0] + 1)
        f.createDimension("x1", xg["lx"][0])

        f.createDimension("x2ghost", xg["lx"][1] + Ng)
        f.createDimension("x2d", xg["lx"][1] + Ng - 1)
        f.createDimension("x2i", xg["lx"][1] + 1)
        f.createDimension("x2", xg["lx"][1])

        f.createDimension("x3ghost", xg["lx"][2] + Ng)
        f.createDimension("x3d", xg["lx"][2] + Ng - 1)
        f.createDimension("x3i", xg["lx"][2] + 1)
        f.createDimension("x3", xg["lx"][2])

        f.createDimension("ecef", 3)

        for i in (1, 2, 3):
            _write_var(f, f"x{i}", xg[f"x{i}"], dims=f"x{i}ghost")
            _write_var(f, f"x{i}i", xg[f"x{i}i"], dims=f"x{i}i")
            _write_var(f, f"dx{i}b", xg[f"dx{i}b"], dims=f"x{i}d")
            _write_var(f, f"dx{i}h", xg[f"dx{i}h"], dims=f"x{i}")
            _write_var(f, f"h{i}", xg[f"h{i}"], ("x1ghost", "x2ghost", "x3ghost"))
            _write_var(f, f"h{i}x1i", xg[f"h{i}x1i"], ("x1i", "x2", "x3"))
            _write_var(f, f"h{i}x2i", xg[f"h{i}x2i"], ("x1", "x2i", "x3"))
            _write_var(f, f"h{i}x3i", xg[f"h{i}x3i"], ("x1", "x2", "x3i"))
            _write_var(f, f"gx{i}", xg[f"gx{i}"], ("x1", "x2", "x3"))
            _write_var(f, f"e{i}", xg[f"e{i}"], ("x1", "x2", "x3", "ecef"))

        for k in ("alt", "glat", "glon", "Bmag", "nullpts", "r", "theta", "phi", "x", "y", "z"):
            _write_var(f, k, xg[k], ("x1", "x2", "x3"))

        for k in ("er", "etheta", "ephi"):
            _write_var(f, k, xg[k], ("x1", "x2", "x3", "ecef"))

        _write_var(f, "I", xg["I"], ("x2", "x3"))


def _write_var(
    f,
    name: str,
    value: np.ndarray | xarray.DataArray,
    dims: str | tuple[str, ...] | list[str] = None,
):

    if dims is None and isinstance(value, xarray.DataArray):
        dims = value.dims
    elif isinstance(dims, str):
        dims = [dims]
    else:
        raise ValueError(f"Please specify dims for {name}")

    dims = dims[::-1]

    g = f.createVariable(
        name,
        np.float32,
        dims,
        zlib=True,
        complevel=1,
        shuffle=True,
        fletcher32=True,
        fill_value=np.nan,
    )

    if value.ndim > 1:
        g[:] = value.transpose()
    else:
        g[:] = value


def Efield(outdir: Path, E: xarray.Dataset):
    """
    write Efield to disk
    """

    if Dataset is None:
        raise ImportError("pip install netcdf4")

    with Dataset(outdir / "simsize.nc", "w") as f:
        g = f.createVariable("llon", np.int32)
        g[:] = E.mlon.size
        g = f.createVariable("llat", np.int32)
        g[:] = E.mlat.size

    with Dataset(outdir / "simgrid.nc", "w") as f:
        f.createDimension("mlon", E.mlon.size)
        f.createDimension("mlat", E.mlat.size)
        _write_var(f, "mlon", E.mlon)
        _write_var(f, "mlat", E.mlat)

    for t in E.time:
        time: datetime = to_datetime(t)
        fn = outdir / (datetime2ymd_hourdec(time) + ".nc")

        # FOR EACH FRAME WRITE A BC TYPE AND THEN OUTPUT BACKGROUND AND BCs
        with Dataset(fn, "w") as f:
            f.createDimension("mlon", E.mlon.size)
            f.createDimension("mlat", E.mlat.size)

            g = f.createVariable("flagdirich", np.int32)
            g[:] = E["flagdirich"].loc[time]

            f.createDimension("ymd", 3)
            g = f.createVariable("ymd", np.int32, "ymd")
            g[:] = [time.year, time.month, time.day]

            g = f.createVariable("UTsec", np.float32)
            g[:] = time.hour * 3600 + time.minute * 60 + time.second + time.microsecond / 1e6

            for k in {"Exit", "Eyit", "Vminx1it", "Vmaxx1it"}:
                _write_var(f, k, E[k].loc[time])

            for k in {"Vminx2ist", "Vmaxx2ist"}:
                _write_var(f, k, E[k].loc[time])

            for k in {"Vminx3ist", "Vmaxx3ist"}:
                _write_var(f, k, E[k].loc[time])


def precip(outdir: Path, P: xarray.Dataset):
    """
    write precipitation to disk
    """

    if Dataset is None:
        raise ImportError("pip install netcdf4")

    with Dataset(outdir / "simsize.nc", "w") as f:
        g = f.createVariable("llon", np.int32)
        g[:] = P.mlon.size
        g = f.createVariable("llat", np.int32)
        g[:] = P.mlat.size

    with Dataset(outdir / "simgrid.nc", "w") as f:
        f.createDimension("mlon", P.mlon.size)
        f.createDimension("mlat", P.mlat.size)
        _write_var(f, "mlon", P.mlon)
        _write_var(f, "mlat", P.mlat)

    for t in P.time:
        time: datetime = to_datetime(t)
        fn = outdir / (datetime2ymd_hourdec(time) + ".nc")

        with Dataset(fn, "w") as f:
            f.createDimension("mlon", P.mlon.size)
            f.createDimension("mlat", P.mlat.size)

            for k in {"Q", "E0"}:
                _write_var(f, f"{k}p", P[k].loc[time])
