"""
NetCDF4 file writing
"""

import typing as T
from datetime import datetime
import numpy as np
from pathlib import Path
import logging

from ..utils import datetime2ymd_hourdec
from .. import LSP

try:
    from netCDF4 import Dataset
except ImportError:
    # must be ImportError not ModuleNotFoundError for botched NetCDF4 linkage
    Dataset = None


def state(time: datetime, ns: np.ndarray, vs: np.ndarray, Ts: np.ndarray, fn: Path):
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

    logging.info(f"state: {fn}")

    with Dataset(fn, "w") as f:
        p4 = (0, 3, 2, 1)

        f.createDimension("ymd", 3)
        g = f.createVariable("ymd", np.int32, "ymd")
        g[:] = [time.year, time.month, time.day]

        g = f.createVariable("UTsec", np.float32)
        g[:] = time.hour * 3600 + time.minute * 60 + time.second + time.microsecond / 1e6

        f.createDimension("species", 7)
        f.createDimension("x1", ns.shape[1])
        f.createDimension("x2", ns.shape[2])
        f.createDimension("x3", ns.shape[3])

        _write_var(f, "ns", ("species", "x3", "x2", "x1"), ns.transpose(p4))
        _write_var(f, "vsx1", ("species", "x3", "x2", "x1"), vs.transpose(p4))
        _write_var(f, "Ts", ("species", "x3", "x2", "x1"), Ts.transpose(p4))


def data(dat: T.Dict[str, T.Any], xg: T.Dict[str, T.Any], fn: Path):
    """
    write simulation data
    e.g. for converting a file format from a simulation
    """

    if Dataset is None:
        raise ImportError("pip install netcdf4")

    logging.info(f"write_data: {fn}")

    lxs = dat["lxs"]

    if "ns" in dat:
        shape = dat["ns"][1].shape
    elif "ne" in dat:
        shape = dat["ne"][1].shape
    else:
        raise ValueError("what variable should I use to determine dimensions?")

    if not lxs == shape:
        raise ValueError(f"simsize {lxs} does not match data shape {shape}")

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
                _write_var(f, k, (k,), x)

        for k in ["ns", "vs1", "Ts"]:
            if k not in dat:
                continue
            _write_var(f, k, dims, dat[k][1])

        for k in ["ne", "v1", "Ti", "Te", "J1", "J2", "J3", "v2", "v3"]:
            if k not in dat:
                continue
            _write_var(f, k, dims, dat[k][1])

        if "Phitop" in dat:
            _write_var(f, "Phitop", dims[-2:], dat["Phitop"][1])


def grid(size_fn: Path, grid_fn: Path, xg: T.Dict[str, T.Any]):
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
            _write_var(f, f"x{i}", (f"x{i}ghost",), xg[f"x{i}"])
            _write_var(f, f"x{i}i", (f"x{i}i",), xg[f"x{i}i"])
            _write_var(f, f"dx{i}b", (f"x{i}d",), xg[f"dx{i}b"])
            _write_var(f, f"dx{i}h", (f"x{i}",), xg[f"dx{i}h"])
            _write_var(f, f"h{i}", ("x3ghost", "x2ghost", "x1ghost"), xg[f"h{i}"].transpose())
            _write_var(f, f"h{i}x1i", ("x3", "x2", "x1i"), xg[f"h{i}x1i"].transpose())
            _write_var(f, f"h{i}x2i", ("x3", "x2i", "x1"), xg[f"h{i}x2i"].transpose())
            _write_var(f, f"h{i}x3i", ("x3i", "x2", "x1"), xg[f"h{i}x3i"].transpose())
            _write_var(f, f"gx{i}", ("x3", "x2", "x1"), xg[f"gx{i}"].transpose())
            _write_var(f, f"e{i}", ("ecef", "x3", "x2", "x1"), xg[f"e{i}"].transpose())

        for k in ("alt", "glat", "glon", "Bmag", "nullpts", "r", "theta", "phi", "x", "y", "z"):
            _write_var(f, k, ("x3", "x2", "x1"), xg[k].transpose())

        for k in ("er", "etheta", "ephi"):
            _write_var(f, k, ("ecef", "x3", "x2", "x1"), xg[k].transpose())

        _write_var(f, "I", ("x3", "x2"), xg["I"].transpose())


def _write_var(f, name: str, dims: T.Sequence[str], value: np.ndarray):
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
    g[:] = value


def Efield(outdir: Path, E: T.Dict[str, np.ndarray]):
    """
    write Efield to disk

    TODO: verify dimensions vs. data vs. Fortran order
    """

    if Dataset is None:
        raise ImportError("pip install netcdf4")

    with Dataset(outdir / "simsize.nc", "w") as f:
        for k in ("llon", "llat"):
            g = f.createVariable(k, np.int32)
            g[:] = E[k]

    with Dataset(outdir / "simgrid.nc", "w") as f:
        f.createDimension("lon", E["mlon"].size)
        f.createDimension("lat", E["mlat"].size)
        _write_var(f, "mlon", ("lon",), E["mlon"])
        _write_var(f, "mlat", ("lat",), E["mlat"])

    for i, t in enumerate(E["time"]):
        fn = outdir / (datetime2ymd_hourdec(t) + ".nc")

        # FOR EACH FRAME WRITE A BC TYPE AND THEN OUTPUT BACKGROUND AND BCs
        with Dataset(fn, "w") as f:
            f.createDimension("lon", E["mlon"].size)
            f.createDimension("lat", E["mlat"].size)

            g = f.createVariable("flagdirich", np.int32)
            g[:] = E["flagdirich"][i]

            f.createDimension("ymd", 3)
            g = f.createVariable("ymd", np.int32, "ymd")
            g[:] = [t.year, t.month, t.day]

            g = f.createVariable("UTsec", np.float32)
            g[:] = t.hour * 3600 + t.minute * 60 + t.second + t.microsecond / 1e6

            for k in ("Exit", "Eyit", "Vminx1it", "Vmaxx1it"):
                _write_var(f, k, ("lat", "lon"), E[k][i, :, :].transpose())

            for k in ("Vminx2ist", "Vmaxx2ist"):
                _write_var(f, k, ("lat",), E[k][i, :])

            for k in ("Vminx3ist", "Vmaxx3ist"):
                _write_var(f, k, ("lon",), E[k][i, :])


def precip(outdir: Path, precip: T.Dict[str, np.ndarray]):
    """
    write precipitation to disk

    TODO: verify dimensions vs. data vs. Fortran order
    """

    if Dataset is None:
        raise ImportError("pip install netcdf4")

    with Dataset(outdir / "simsize.nc", "w") as f:
        for k in ("llon", "llat"):
            g = f.createVariable(k, np.int32)
            g[:] = precip[k]

    with Dataset(outdir / "simgrid.nc", "w") as f:
        f.createDimension("lon", precip["mlon"].size)
        f.createDimension("lat", precip["mlat"].size)
        _write_var(f, "mlon", ("lon",), precip["mlon"])
        _write_var(f, "mlat", ("lat",), precip["mlat"])

    for i, t in enumerate(precip["time"]):
        fn = outdir / (datetime2ymd_hourdec(t) + ".nc")

        with Dataset(fn, "w") as f:
            f.createDimension("lon", precip["mlon"].size)
            f.createDimension("lat", precip["mlat"].size)

            for k in ("Q", "E0"):
                _write_var(f, f"{k}p", ("lat", "lon"), precip[k][i, :, :].transpose())
