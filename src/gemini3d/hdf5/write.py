"""
HDF5 file writing
"""

from __future__ import annotations
import typing as T
from pathlib import Path
from datetime import datetime
import logging

import h5py
import numpy as np
import xarray

from ..utils import datetime2ymd_hourdec, to_datetime

CLVL = 3  # GZIP compression level: larger => better compression, slower to write


def state(fn: Path, dat: xarray.Dataset):
    """
    write STATE VARIABLE initial conditions

    NOTE THAT WE don't write ANY OF THE ELECTRODYNAMIC
    VARIABLES SINCE THEY ARE NOT NEEDED TO START THINGS
    UP IN THE FORTRAN CODE.

    INPUT ARRAYS SHOULD BE TRIMMED TO THE CORRECT SIZE
    I.E. THEY SHOULD NOT INCLUDE GHOST CELLS
    """

    logging.info(f"state: {fn}")

    with h5py.File(fn, "w") as f:
        write_time(f, to_datetime(dat.time))

        for k in {"ns", "vs1", "Ts"}:
            if k in dat.data_vars:
                _write_var(f, f"/{k}all", dat[k])

        if "Phitop" in dat.data_vars:
            _write_var(f, "/Phiall", dat["Phitop"])


def _write_var(fid, name: str, A: xarray.DataArray):
    """
    NOTE: The .transpose() reverses the dimension order.
    The HDF Group never implemented the intended H5T_array_create(..., perm)
    and it's deprecated.

    Fortran, including the HDF Group Fortran interfaces and h5fortran as well as
    Matlab read/write HDF5 in Fortran order. h5py read/write HDF5 in C order so we
    need the .transpose() for h5py
    """

    p4s = ("species", "x3", "x2", "x1")
    p3s = ("x3", "x2", "x1")
    p2s = ("x3", "x2")

    if A.ndim == 4:
        A = A.transpose(*p4s)
    elif A.ndim == 3:
        A = A.transpose(*p3s)
    elif A.ndim == 2:
        A = A.transpose(*p2s)
    elif A.ndim == 1:
        pass
    else:
        raise ValueError(
            f"write_hdf5: unexpected number of dimensions {A.ndim}. Please raise a GitHub Issue."
        )

    fid.create_dataset(
        name,
        data=A,
        dtype=np.float32,  # float32 saves disk space
        compression="gzip",  # GZIP is universally available with HDF5
        compression_opts=CLVL,
        shuffle=True,
        fletcher32=True,
    )


def data(outfn: Path, dat: xarray.Dataset):
    """
    write simulation data
    e.g. for converting a file format from a simulation
    """

    with h5py.File(outfn, "w") as f:
        write_time(f, to_datetime(dat.time))

        for k in {
            "ns",
            "vs1",
            "Ts",
            "ne",
            "v1",
            "Ti",
            "Te",
            "J1",
            "J2",
            "J3",
            "v2",
            "v3",
            "Phitop",
        }:
            if k in dat:
                _write_var(f, k, dat[k])


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

    NOTE: The .transpose() reverses the dimension order.
    The HDF Group never implemented the intended H5T_array_create(..., perm)
    and it's deprecated.
    Fortran, including the HDF Group Fortran interfaces and h5fortran as well as
    Matlab read/write HDF5 in Fortran order. h5py read/write HDF5 in C order so we
    need the .transpose() for h5py
    """

    if "lx" not in xg:
        xg["lx"] = np.array([xg["x1"].shape, xg["x2"].shape, xg["x3"].shape]).astype(np.int32)

    logging.info(f"write_grid: {size_fn}")
    with h5py.File(size_fn, "w") as h:
        h["/lx"] = np.asarray(xg["lx"]).astype(np.int32)

    logging.info(f"write_grid: {grid_fn}")
    with h5py.File(grid_fn, "w") as h:
        for i in {1, 2, 3}:
            for k in {
                f"x{i}",
                f"x{i}i",
                f"dx{i}b",
                f"dx{i}h",
                f"h{i}",
                f"h{i}x1i",
                f"h{i}x2i",
                f"h{i}x3i",
            }:
                if k not in xg:
                    logging.info(f"SKIP: {k}")
                    continue

                if xg[k].ndim >= 2:
                    h.create_dataset(
                        f"/{k}",
                        data=xg[k].transpose(),
                        dtype=np.float32,
                        compression="gzip",
                        compression_opts=CLVL,
                        shuffle=True,
                        fletcher32=True,
                    )
                else:
                    h[f"/{k}"] = xg[k].astype(np.float32)

        # 3-D same as grid
        for k in {
            "gx1",
            "gx2",
            "gx3",
            "alt",
            "glat",
            "glon",
            "Bmag",
            "nullpts",
            "r",
            "theta",
            "phi",
            "x",
            "y",
            "z",
        }:
            if k not in xg:
                logging.info(f"SKIP: {k}")
                continue

            h.create_dataset(
                f"/{k}",
                shape=xg["lx"][::-1],
                data=xg[k].transpose(),
                dtype=np.float32,
                compression="gzip",
                compression_opts=CLVL,
                shuffle=True,
                fletcher32=True,
            )

        # %% 2-D
        for k in {"I"}:
            if k not in xg:
                logging.info(f"SKIP: {k}")
                continue

            h.create_dataset(
                f"/{k}",
                shape=(xg["lx"][1], xg["lx"][2])[::-1],
                data=xg[k].transpose(),
                dtype=np.float32,
                compression="gzip",
                compression_opts=CLVL,
                shuffle=True,
                fletcher32=True,
            )

        # %% 4-D
        for k in {"e1", "e2", "e3", "er", "etheta", "ephi"}:
            if k not in xg:
                logging.info(f"SKIP: {k}")
                continue

            h.create_dataset(
                f"/{k}",
                shape=(*xg["lx"], 3)[::-1],
                data=xg[k].transpose(),
                dtype=np.float32,
                compression="gzip",
                compression_opts=CLVL,
                shuffle=True,
                fletcher32=True,
            )

        if "glonctr" in xg:
            h["/glonctr"] = xg["glonctr"]
            h["/glatctr"] = xg["glatctr"]


def Efield(outdir: Path, E: xarray.Dataset):
    """
    write Efield to disk
    """

    with h5py.File(outdir / "simsize.h5", "w") as f:
        f.create_dataset("/llon", data=E.mlon.size, dtype=np.int32)
        f.create_dataset("/llat", data=E.mlat.size, dtype=np.int32)

    with h5py.File(outdir / "simgrid.h5", "w") as f:
        f["/mlon"] = E.mlon.astype(np.float32)
        f["/mlat"] = E.mlat.astype(np.float32)

    for t in E.time:
        time: datetime = to_datetime(t)
        fn = outdir / (datetime2ymd_hourdec(time) + ".h5")

        # FOR EACH FRAME WRITE A BC TYPE AND THEN OUTPUT BACKGROUND AND BCs
        with h5py.File(fn, "w") as f:
            f["/flagdirich"] = E["flagdirich"].loc[time].astype(np.int32)
            write_time(f, time)

            for k in {"Exit", "Eyit", "Vminx1it", "Vmaxx1it"}:
                f.create_dataset(
                    f"/{k}",
                    data=E[k].loc[time].transpose(),
                    dtype=np.float32,
                    compression="gzip",
                    compression_opts=CLVL,
                    shuffle=True,
                    fletcher32=True,
                )
            for k in {"Vminx2ist", "Vmaxx2ist", "Vminx3ist", "Vmaxx3ist"}:
                f[f"/{k}"] = E[k].loc[time].astype(np.float32)


def precip(outdir: Path, P: xarray.Dataset):

    with h5py.File(outdir / "simsize.h5", "w") as f:
        f.create_dataset("/llon", data=P.mlon.size, dtype=np.int32)
        f.create_dataset("/llat", data=P.mlat.size, dtype=np.int32)

    with h5py.File(outdir / "simgrid.h5", "w") as f:
        f["/mlon"] = P.mlon.astype(np.float32)
        f["/mlat"] = P.mlat.astype(np.float32)

    for t in P.time:
        time: datetime = to_datetime(t)
        fn = outdir / (datetime2ymd_hourdec(time) + ".h5")

        with h5py.File(fn, "w") as f:
            write_time(f, to_datetime(time))

            for k in {"Q", "E0"}:
                f.create_dataset(
                    f"/{k}p",
                    data=P[k].loc[time].transpose(),
                    dtype=np.float32,
                    compression="gzip",
                    compression_opts=CLVL,
                    shuffle=True,
                    fletcher32=True,
                )


def maggrid(fn: Path, mag: dict[str, np.ndarray], gridsize: tuple[int, int, int]):
    """
    hdf5 files can optionally store a gridsize variable which tells readers how to
    reshape the data into 2D or 3D arrays.
    NOTE: the Fortran magcalc.f90 is looking for flat list.
    """

    print("write", fn)

    freal = np.float32

    with h5py.File(fn, "w") as f:
        f.create_dataset("/lpoints", data=mag["r"].size, dtype=np.int32)
        f["/r"] = mag["r"].ravel(order="F").astype(freal)
        f["/theta"] = mag["theta"].ravel(order="F").astype(freal)
        f["/phi"] = mag["phi"].ravel(order="F").astype(freal)
        f["/gridsize"] = np.asarray(gridsize).astype(np.int32)


def write_time(fid: h5py.File, time: datetime):
    """
    write time to HDF5 file as year-month-day, UTsec
    """

    fid["/time/ymd"] = np.array([time.year, time.month, time.day]).astype(np.int32)
    fid["/time/UTsec"] = time.hour * 3600 + time.minute * 60 + time.second + time.microsecond / 1e6
