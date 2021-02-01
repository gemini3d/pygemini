"""
raw binary file I/O.
Raw files are deprecated and do not contain most features of Gemini
"""

from __future__ import annotations
import xarray
from pathlib import Path
import typing as T
import numpy as np
import logging
import struct
from datetime import datetime, timedelta

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

    path = find.simsize(path, ".dat")

    fsize = path.stat().st_size
    if fsize == 12:
        lxs = struct.unpack("III", path.open("rb").read(12))
    elif fsize == 8:
        lxs = struct.unpack("II", path.open("rb").read(8))
    else:
        raise ValueError(f"{path} is not expected 8 bytes (2-D) or 12 bytes (3-D) long")

    return lxs


def grid(fn: Path, shape: bool = False) -> dict[str, T.Any]:
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

    lxs = simsize(fn.parent)
    if len(lxs) == 2:
        return grid2(fn, lxs)
    elif len(lxs) == 3:
        return grid3(fn, lxs)
    else:
        raise ValueError("lxs must be 2-D or 3-D")


def grid2(fn: Path, lxs: tuple[int, ...] | list[int]) -> dict[str, np.ndarray]:
    """ for Efield """
    if not fn.is_file():
        raise FileNotFoundError(fn)

    grid: dict[str, T.Any] = {"lx": lxs}
    with fn.open("r") as f:
        grid["mlon"] = np.fromfile(f, np.float64, lxs[0])
        grid["mlat"] = np.fromfile(f, np.float64, lxs[1])

    return grid


def grid3(fn: Path, lxs: tuple[int, ...] | list[int]) -> dict[str, np.ndarray]:

    lgridghost = (lxs[0] + 4) * (lxs[1] + 4) * (lxs[2] + 4)
    gridsizeghost = [lxs[0] + 4, lxs[1] + 4, lxs[2] + 4]

    grid: dict[str, T.Any] = {"lx": lxs}

    if not fn.is_file():
        logging.error(f"{fn} grid file is not present. Will try to load rest of data.")
        return grid

    read = np.fromfile

    with fn.open("r") as f:
        for i in (1, 2, 3):
            grid[f"x{i}"] = read(f, np.float64, lxs[i - 1] + 4)
            grid[f"x{i}i"] = read(f, np.float64, lxs[i - 1] + 1)
            grid[f"dx{i}b"] = read(f, np.float64, lxs[i - 1] + 3)
            grid[f"dx{i}h"] = read(f, np.float64, lxs[i - 1])
        for i in (1, 2, 3):
            grid[f"h{i}"] = read(f, np.float64, lgridghost).reshape(gridsizeghost)
        L = [lxs[0] + 1, lxs[1], lxs[2]]
        for i in (1, 2, 3):
            grid[f"h{i}x1i"] = read(f, np.float64, np.prod(L)).reshape(L)
        L = [lxs[0], lxs[1] + 1, lxs[2]]
        for i in (1, 2, 3):
            grid[f"h{i}x2i"] = read(f, np.float64, np.prod(L)).reshape(L)
        L = [lxs[0], lxs[1], lxs[2] + 1]
        for i in (1, 2, 3):
            grid[f"h{i}x3i"] = read(f, np.float64, np.prod(L)).reshape(L)
        for i in (1, 2, 3):
            grid[f"gx{i}"] = read(f, np.float64, np.prod(lxs)).reshape(lxs)
        for k in ("alt", "glat", "glon", "Bmag"):
            grid[k] = read(f, np.float64, np.prod(lxs)).reshape(lxs)
        grid["Bincl"] = read(f, np.float64, lxs[1] * lxs[2]).reshape(lxs[1:])
        grid["nullpts"] = read(f, np.float64, np.prod(lxs)).reshape(lxs)
        if f.tell() == fn.stat().st_size:  # not EOF
            return grid

        L = [lxs[0], lxs[1], lxs[2], 3]
        for i in (1, 2, 3):
            grid[f"e{i}"] = read(f, np.float64, np.prod(L)).reshape(L)
        for k in ("er", "etheta", "ephi"):
            grid[k] = read(f, np.float64, np.prod(L)).reshape(L)
        for k in ("r", "theta", "phi"):
            grid[k] = read(f, np.float64, np.prod(lxs)).reshape(lxs)
        if f.tell() == fn.stat().st_size:  # not EOF
            return grid

        for k in ("x", "y", "z"):
            grid[k] = read(f, np.float64, np.prod(lxs)).reshape(lxs)

    return grid


def Efield(file: Path) -> xarray.Dataset:
    """
    load Efield_inputs files that contain input electric field in V/m
    """

    lxs = simsize(file.parent)

    assert lxs[0] > 0, "must have strictly positive number of longitude cells"
    assert lxs[1] > 0, "must have strictly positive number of latitude cells"

    m = grid2(file.parent / "simgrid.dat", lxs)

    assert (
        (m["mlat"] >= -90) & (m["mlat"] <= 90)
    ).all(), f"impossible latitude, was file read correctly? {file}"

    dat = xarray.Dataset(coords=m)

    with file.open("r") as f:
        """
        NOTE:
        this is mistakenly a float from Matlab
        to keep compatibility with old files, we left it as real64.
        New work should be using HDF5 instead of raw in any case.
        """
        dat["flagdirich"] = int(np.fromfile(f, np.float64, 1))
        for p in ("Exit", "Eyit", "Vminx1it", "Vmaxx1it"):
            dat[p] = (("x2", "x3"), read2D(f, lxs))
        for p in ("Vminx2ist", "Vmaxx2ist"):
            dat[p] = (("x2",), np.fromfile(f, np.float64, lxs[1]))
        for p in ("Vminx3ist", "Vmaxx3ist"):
            dat[p] = (("x3",), np.fromfile(f, np.float64, lxs[0]))
        filesize = file.stat().st_size
        if f.tell() != filesize:
            logging.error(f"{file} size {filesize} != file read position {f.tell()}")

    return dat


def frame3d_curv(file: Path) -> xarray.Dataset:
    """
    curvilinear

    Parameters
    ----------

    file: pathlib.Path
        filename to read
    """

    xg = grid(file.parent)
    dat = xarray.Dataset(coords={"x1": xg["x1"][2:-2], "x2": xg["x2"][2:-2], "x3": xg["x3"][2:-2]})
    lxs = simsize(file.parent)

    with file.open("r") as f:
        time(f)

        ns = read4D(f, LSP, lxs)
        dat["ne"] = (("x1", "x2", "x3"), ns[:, :, :, LSP - 1])

        vs1 = read4D(f, LSP, lxs)
        dat["v1"] = (
            ("x1", "x2", "x3"),
            (ns[:, :, :, :6] * vs1[:, :, :, :6]).sum(axis=3) / dat["ne"],
        )

        Ts = read4D(f, LSP, lxs)
        dat["Ti"] = (
            ("x1", "x2", "x3"),
            (ns[:, :, :, :6] * Ts[:, :, :, :6]).sum(axis=3) / dat["ne"],
        )
        dat["Te"] = (("x1", "x2", "x3"), Ts[:, :, :, LSP - 1].squeeze())

        for p in ("J1", "J2", "J3", "v2", "v3"):
            dat[p] = (("x1", "x2", "x3"), read3D(f, lxs))

        dat["Phitop"] = (("x2", "x3"), read2D(f, lxs))

    return dat


def frame3d_curvavg(file: Path) -> xarray.Dataset:
    """

    Parameters
    ----------
    file: pathlib.Path
        filename of this timestep of simulation output
    """

    lxs = simsize(file.parent)
    xg = grid(file.parent)
    dat = xarray.Dataset(coords={"x1": xg["x1"][2:-2], "x2": xg["x2"][2:-2], "x3": xg["x3"][2:-2]})

    with file.open("r") as f:
        time(f)

        for p in ("ne", "v1", "Ti", "Te", "J1", "J2", "J3", "v2", "v3"):
            dat[p] = (("x1", "x2", "x3"), read3D(f, lxs))

        dat["Phitop"] = (("x2", "x3"), read2D(f, lxs))

    return dat


def frame3d_curvne(file: Path) -> xarray.Dataset:

    lxs = simsize(file.parent)
    xg = grid(file.parent)
    dat = xarray.Dataset(coords={"x1": xg["x1"][2:-2], "x2": xg["x2"][2:-2], "x3": xg["x3"][2:-2]})

    with file.open("r") as f:
        time(f)

        dat["ne"] = (("x1", "x2", "x3"), read3D(f, lxs))

    return dat


def read4D(f, lsp: int, lxs: tuple[int, ...] | list[int]) -> np.ndarray:
    """
    read 4D array from raw file
    """
    if not len(lxs) == 3:
        raise ValueError(f"lxs must have 3 elements, you have lxs={lxs}")

    return np.fromfile(f, np.float64, np.prod(lxs) * lsp).reshape((*lxs, lsp), order="F")


def read3D(f, lxs: tuple[int, ...] | list[int]) -> np.ndarray:
    """
    read 3D array from raw file
    """
    if not len(lxs) == 3:
        raise ValueError(f"lxs must have 3 elements, you have lxs={lxs}")

    return np.fromfile(f, np.float64, np.prod(lxs)).reshape(*lxs, order="F")


def read2D(f, lxs: tuple[int, ...] | list[int]) -> np.ndarray:
    """
    read 2D array from raw file
    """
    if not len(lxs) == 3:
        raise ValueError(f"lxs must have 3 elements, you have lxs={lxs}")

    return np.fromfile(f, np.float64, np.prod(lxs[1:])).reshape(*lxs[1:], order="F")


def glow_aurmap(file: Path) -> xarray.Dataset:
    """
    read the auroral output from GLOW
    """

    lxs = simsize(file.parent)
    xg = grid(file.parent)
    dat = xarray.Dataset(coords={"wavelength": WAVELEN, "x2": xg["x2"][2:-2], "x3": xg["x3"][2:-2]})

    if not len(lxs) == 3:
        raise ValueError(f"lxs must have 3 elements, you have lxs={lxs}")

    with file.open("r") as f:
        raw = np.fromfile(f, np.float64, np.prod(lxs[1:]) * len(WAVELEN)).reshape(
            np.prod(lxs[1:]) * len(WAVELEN), order="F"
        )

    dat["rayleighs"] = (("wavelength", "x2", "x3"), raw)


def time(f) -> datetime:
    t = np.fromfile(f, np.float64, 4)
    return datetime(int(t[0]), int(t[1]), int(t[2])) + timedelta(hours=t[3])
