"""
raw binary file I/O.
Raw files are deprecated and do not contain most features of Gemini
"""

from __future__ import annotations

from pathlib import Path
import typing as T
import logging
import struct
from datetime import datetime, timedelta

import numpy as np
import xarray

from .. import find
from .. import WAVELEN, LSP


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

    path = find.simsize(path, suffix=".dat")

    fsize = path.stat().st_size
    if fsize == 12:
        lx = struct.unpack("III", path.open("rb").read(12))
    elif fsize == 8:
        lx = struct.unpack("II", path.open("rb").read(8))
    else:
        raise ValueError(f"{path} is not expected 8 bytes (2-D) or 12 bytes (3-D) long")

    return lx


def grid(file: Path, shape: bool = False) -> dict[str, T.Any]:
    """
    get simulation grid

    Parameters
    ----------
    fn: pathlib.Path
        filepath to simgrid

    Returns
    -------
    grid: dict
        grid parameters
    """

    if shape:
        raise NotImplementedError("grid shape for raw would be straightforward.")

    lx = simsize(file)

    if not file.is_file():
        file = find.grid(file, suffix=".dat")

    if len(lx) == 2:
        return grid2(file, lx)
    elif len(lx) == 3:
        return grid3(file, lx)
    else:
        raise ValueError("lx must be 2-D or 3-D")


def grid2(fn: Path, lx: tuple[int, ...] | list[int]) -> dict[str, T.Any]:
    """for Efield"""

    ft = np.float64

    if not fn.is_file():
        raise FileNotFoundError(fn)

    with fn.open("rb") as f:
        xg = {"lx": lx, "mlon": np.fromfile(f, ft, lx[0]), "mlat": np.fromfile(f, ft, lx[1])}

    return xg


def grid3(fn: Path, lx: tuple[int, ...] | list[int]) -> dict[str, T.Any]:
    """
    load 3D grid
    """

    if not fn.is_file():
        raise FileNotFoundError(fn)

    lgridghost = (lx[0] + 4) * (lx[1] + 4) * (lx[2] + 4)
    gridsizeghost = [lx[0] + 4, lx[1] + 4, lx[2] + 4]

    ft = np.float64

    xg = {"lx": lx}

    read = np.fromfile

    with fn.open("rb") as f:
        for i in (1, 2, 3):
            xg[f"x{i}"] = read(f, ft, lx[i - 1] + 4)
            xg[f"x{i}i"] = read(f, ft, lx[i - 1] + 1)
            xg[f"dx{i}b"] = read(f, ft, lx[i - 1] + 3)
            xg[f"dx{i}h"] = read(f, ft, lx[i - 1])
        for i in (1, 2, 3):
            xg[f"h{i}"] = read(f, ft, lgridghost).reshape(gridsizeghost)
        L = [lx[0] + 1, lx[1], lx[2]]
        for i in (1, 2, 3):
            xg[f"h{i}x1i"] = read(f, ft, np.prod(L)).reshape(L)
        L = [lx[0], lx[1] + 1, lx[2]]
        for i in (1, 2, 3):
            xg[f"h{i}x2i"] = read(f, ft, np.prod(L)).reshape(L)
        L = [lx[0], lx[1], lx[2] + 1]
        for i in (1, 2, 3):
            xg[f"h{i}x3i"] = read(f, ft, np.prod(L)).reshape(L)
        for i in (1, 2, 3):
            xg[f"gx{i}"] = read(f, ft, np.prod(lx)).reshape(lx)
        for k in ("alt", "glat", "glon", "Bmag"):
            xg[k] = read(f, ft, np.prod(lx)).reshape(lx)
        xg["Bincl"] = read(f, ft, lx[1] * lx[2]).reshape(lx[1:])
        xg["nullpts"] = read(f, ft, np.prod(lx)).reshape(lx)
        if f.tell() == fn.stat().st_size:  # not EOF
            return xg

        L = [lx[0], lx[1], lx[2], 3]
        for i in (1, 2, 3):
            xg[f"e{i}"] = read(f, ft, np.prod(L)).reshape(L)
        for k in ("er", "etheta", "ephi"):
            xg[k] = read(f, ft, np.prod(L)).reshape(L)
        for k in ("r", "theta", "phi"):
            xg[k] = read(f, ft, np.prod(lx)).reshape(lx)
        if f.tell() == fn.stat().st_size:  # not EOF
            return xg

        for k in ("x", "y", "z"):
            xg[k] = read(f, ft, np.prod(lx)).reshape(lx)

    return xg


def Efield(file: Path) -> xarray.Dataset:
    """
    load Efield_inputs files that contain input electric field in V/m
    """

    ft = np.float64

    lx = simsize(file.parent)

    assert lx[0] > 0, "must have strictly positive number of longitude cells"
    assert lx[1] > 0, "must have strictly positive number of latitude cells"

    m = grid2(file.parent / "simgrid.dat", lx)

    if ((m["mlat"] < -90) | (m["mlat"] > 90)).any():
        raise ValueError(f"impossible latitude, was file read correctly? {file}")

    dat = xarray.Dataset(coords={"mlon": m["mlon"], "mlat": m["mlat"]})

    with file.open("rb") as f:
        """
        NOTE:
        this is mistakenly a float from Matlab
        to keep compatibility with old files, we left it as real64.
        New work should be using HDF5 instead of raw in any case.
        """
        dat["flagdirich"] = int(np.fromfile(f, ft, 1))
        for p in ("Exit", "Eyit", "Vminx1it", "Vmaxx1it"):
            dat[p] = (("x2", "x3"), read2D(f, lx))
        for p in ("Vminx2ist", "Vmaxx2ist"):
            dat[p] = (("x2",), np.fromfile(f, ft, lx[1]))
        for p in ("Vminx3ist", "Vmaxx3ist"):
            dat[p] = (("x3",), np.fromfile(f, ft, lx[0]))
        filesize = file.stat().st_size
        if f.tell() != filesize:
            logging.error(f"{file} size {filesize} != file read position {f.tell()}")

    return dat


def frame3d_curv(file: Path, xg: dict[str, T.Any] = None) -> xarray.Dataset:
    """
    curvilinear

    Parameters
    ----------

    file: pathlib.Path
        filename to read
    """

    if not file.is_file():
        raise FileNotFoundError(file)

    lx = simsize(file.parent)

    try:
        if not xg:
            xg = grid(file.parent)

        dat = xarray.Dataset(
            coords={"x1": xg["x1"][2:-2], "x2": xg["x2"][2:-2], "x3": xg["x3"][2:-2]}
        )
    except FileNotFoundError:
        # perhaps converting raw data, and didn't have the huge grid file
        logging.error("simgrid.dat missing, returning data without grid information")
        dat = xarray.Dataset(coords={"x1": range(lx[0]), "x2": range(lx[1]), "x3": range(lx[2])})

    with file.open("rb") as f:
        dat = dat.assign_coords({"time": time(f)})

        ns = read4D(f, LSP, lx)
        dat["ne"] = (("x1", "x2", "x3"), ns[:, :, :, LSP - 1])

        vs1 = read4D(f, LSP, lx)
        dat["v1"] = (
            ("x1", "x2", "x3"),
            (ns[:, :, :, :6] * vs1[:, :, :, :6]).sum(axis=3) / dat["ne"],
        )

        Ts = read4D(f, LSP, lx)
        dat["Ti"] = (
            ("x1", "x2", "x3"),
            (ns[:, :, :, :6] * Ts[:, :, :, :6]).sum(axis=3) / dat["ne"],
        )
        dat["Te"] = (("x1", "x2", "x3"), Ts[:, :, :, LSP - 1].squeeze())

        for p in ("J1", "J2", "J3", "v2", "v3"):
            dat[p] = (("x1", "x2", "x3"), read3D(f, lx))

        dat["Phitop"] = (("x2", "x3"), read2D(f, lx))

    return dat


def frame3d_curvavg(file: Path, xg: dict[str, T.Any] = None) -> xarray.Dataset:
    """

    Parameters
    ----------
    file: pathlib.Path
        filename of this timestep of simulation output
    """

    if not file.is_file():
        raise FileNotFoundError(file)

    lx = simsize(file.parent)

    try:
        if not xg:
            xg = grid(file.parent)

        dat = xarray.Dataset(
            coords={"x1": xg["x1"][2:-2], "x2": xg["x2"][2:-2], "x3": xg["x3"][2:-2]}
        )
    except FileNotFoundError:
        # perhaps converting raw data, and didn't have the huge grid file
        logging.error("simgrid.dat missing, returning data without grid information")
        dat = xarray.Dataset(coords={"x1": range(lx[0]), "x2": range(lx[1]), "x3": range(lx[2])})

    with file.open("rb") as f:
        dat = dat.assign_coords({"time": time(f)})

        for p in ("ne", "v1", "Ti", "Te", "J1", "J2", "J3", "v2", "v3"):
            dat[p] = (("x1", "x2", "x3"), read3D(f, lx))

        dat["Phitop"] = (("x2", "x3"), read2D(f, lx))

    return dat


def frame3d_curvne(file: Path, xg: dict[str, T.Any] = None) -> xarray.Dataset:

    if not file.is_file():
        raise FileNotFoundError(file)

    lx = simsize(file.parent)

    try:
        if not xg:
            xg = grid(file.parent)

        dat = xarray.Dataset(
            coords={"x1": xg["x1"][2:-2], "x2": xg["x2"][2:-2], "x3": xg["x3"][2:-2]}
        )
    except FileNotFoundError:
        # perhaps converting raw data, and didn't have the huge grid file
        logging.error("simgrid.dat missing, returning data without grid information")
        dat = xarray.Dataset(coords={"x1": range(lx[0]), "x2": range(lx[1]), "x3": range(lx[2])})

    with file.open("rb") as f:
        dat = dat.assign_coords({"time": time(f)})

        dat["ne"] = (("x1", "x2", "x3"), read3D(f, lx))

    return dat


def read4D(f: T.BinaryIO, lsp: int, lx: tuple[int, ...] | list[int]) -> np.ndarray:
    """
    read 4D array from raw file
    """

    if not len(lx) == 3:
        raise ValueError(f"lx must have 3 elements, you have lx={lx}")

    return np.fromfile(f, np.float64, np.prod(lx) * lsp).reshape((*lx, lsp), order="F")


def read3D(f: T.BinaryIO, lx: tuple[int, ...] | list[int]) -> np.ndarray:
    """
    read 3D array from raw file
    """

    if not len(lx) == 3:
        raise ValueError(f"lx must have 3 elements, you have lx={lx}")

    return np.fromfile(f, np.float64, np.prod(lx)).reshape(*lx, order="F")


def read2D(f: T.BinaryIO, lx: tuple[int, ...] | list[int]) -> np.ndarray:
    """
    read 2D array from raw file
    """

    if not len(lx) == 3:
        raise ValueError(f"lx must have 3 elements, you have lx={lx}")

    return np.fromfile(f, np.float64, np.prod(lx[1:])).reshape(*lx[1:], order="F")


def glow_aurmap(file: Path, xg: dict[str, T.Any] = None) -> xarray.Dataset:
    """
    read the auroral output from GLOW
    """

    lx = simsize(file.parent)
    if not xg:
        xg = grid(file.parent)

    dat = xarray.Dataset(coords={"wavelength": WAVELEN, "x2": xg["x2"][2:-2], "x3": xg["x3"][2:-2]})

    if not len(lx) == 3:
        raise ValueError(f"lx must have 3 elements, you have lx={lx}")

    with file.open("rb") as f:
        raw = np.fromfile(f, np.float64, np.prod(lx[1:]) * len(WAVELEN)).reshape(
            np.prod(lx[1:]) * len(WAVELEN), order="F"
        )

    dat["rayleighs"] = (("wavelength", "x2", "x3"), raw)

    return dat


def time(f: T.BinaryIO) -> datetime:

    t = np.fromfile(f, np.float64, 4)

    return datetime(int(t[0]), int(t[1]), int(t[2])) + timedelta(hours=t[3])
