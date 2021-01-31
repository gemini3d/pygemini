"""
functions for interfacing with Matlab or using Matlab data formats
"""

from __future__ import annotations
from pathlib import Path
import typing as T
import scipy.io


def grid(fn: Path, shape: bool = False) -> dict[str, T.Any]:
    """
    get simulation grid

    Parameters
    ----------
    fn: pathlib.Path
        filepath to simgrid file

    Returns
    -------
    grid: dict
        grid parameters
    """

    if shape:
        raise NotImplementedError("shape with Matlab")

    grid = scipy.io.loadmat(fn, squeeze_me=True)

    return grid


def simsize(path: Path) -> tuple[int, ...]:
    """
    get simulation size
    """
    path = Path(path).expanduser().resolve()

    f = scipy.io.loadmat(path, squeeze_me=True)
    if "lxs" in f:
        lxs = f["lxs"]
    elif "lx" in f:
        lxs = f["lx"]
    elif "lx1" in f:
        if f["lx1"].ndim > 0:
            lxs = (
                f["lx1"],
                f["lx2"],
                f["lx3"],
            )
        else:
            lxs = (f["lx1"], f["lx2"], f["lx3"])
    else:
        raise KeyError(f"could not find '/lxs', '/lx' or '/lx1' in {path.as_posix()}")

    return lxs


def state(fn: Path) -> dict[str, T.Any]:
    """
    load initial condition data
    """

    return scipy.io.loadmat(fn, squeeze_me=True)


def precip(fn: Path) -> dict[str, T.Any]:
    """
    load precipitation data
    """

    return scipy.io.loadmat(fn, squeeze_me=True)
