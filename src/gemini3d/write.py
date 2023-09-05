from __future__ import annotations
from pathlib import Path
import typing as T
import sys
import logging
import json

import xarray

from .utils import git_meta
from .hdf5 import write as h5write


def state(out_file: Path, dat: xarray.Dataset, **kwargs) -> None:
    """
    WRITE STATE VARIABLE DATA.
    NOTE: WE don't write ANY OF THE ELECTRODYNAMIC
    VARIABLES SINCE THEY ARE NOT NEEDED TO START THINGS
    UP IN THE FORTRAN CODE.

    INPUT ARRAYS SHOULD BE TRIMMED TO THE CORRECT SIZE
    I.E. THEY SHOULD NOT INCLUDE GHOST CELLS
    """

    # %% allow overriding "dat"
    if "time" in kwargs:
        dat.attrs["time"] = kwargs["time"]

    for k in {"ns", "vs1", "Ts"}:
        if k in kwargs:
            dat[k] = (("species", "x1", "x2", "x3"), kwargs[k])

    if "Phitop" in kwargs:
        dat["Phitop"] = (("x2", "x3"), kwargs["Phitop"])

    h5write.state(out_file, dat)


def grid(cfg: dict[str, T.Any], xg: dict[str, T.Any]) -> None:
    """writes grid to disk

    Parameters
    ----------

    cfg: dict
        simulation parameters
    xg: dict
        grid values
    """

    input_dir = cfg["indat_size"].parent
    if input_dir.is_file():
        raise OSError(f"{input_dir} is a file instead of directory")

    input_dir.mkdir(parents=True, exist_ok=True)

    h5write.grid(cfg["indat_size"], cfg["indat_grid"], xg)

    meta(input_dir / "setup_grid.json", git_meta(), cfg)


def Efield(E: xarray.Dataset, outdir: Path) -> None:
    """writes E-field to disk

    Parameters
    ----------

    E: dict
        E-field values
    outdir: pathlib.Path
        directory to write files into
    """

    print("write E-field data to", outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    h5write.Efield(outdir, E)


def precip(precip: xarray.Dataset, outdir: Path) -> None:
    """writes precipitation to disk

    Parameters
    ----------
    precip: dict
        preicipitation values
    outdir: pathlib.Path
        directory to write files into
    """

    print("write precipitation data to", outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    h5write.precip(outdir, precip)


def neutral2(data: dict[str, T.Any], outfile: Path):
    """
    writes 2D neutral data to disk

    Parameters
    ----------
    data: dict
        neutral data
    outdir: pathlib.Path
        directory to write files into
    """

    h5write.neutral(outfile, data)


def meta(fn: Path, git_meta: dict[str, str], cfg: dict[str, T.Any]) -> None:
    """
    writes JSON file with sim setup metadata
    """

    fn = fn.expanduser()
    if fn.is_dir():
        raise FileNotFoundError(
            f"{fn} is a directory, but I need a JSON file name to write."
        )

    jm = {"python": {"platform": sys.platform, "version": sys.version}, "git": git_meta}

    if "eq_dir" in cfg:
        # JSON does not allow unescaped backslash
        jm["equilibrium"] = {"eq_dir": cfg["eq_dir"].as_posix()}
        hf = cfg["eq_dir"] / "sha256sum.txt"
        if hf.is_file():
            jm["equilibrium"]["sha256"] = hf.read_text().strip()

    js = json.dumps(jm, sort_keys=True, indent=2)

    fn.write_text(js)


def maggrid(filename: Path, xmag: dict[str, T.Any]) -> None:
    filename = Path(filename).expanduser()

    # %% default value for gridsize
    if "gridsize" not in xmag:
        if xmag["r"].ndim == 1:
            logging.warning("Defaulting gridsize to flat list")
            gridsize = (xmag["r"].size, -1, -1)
        else:
            gridsize = xmag["r"].shape
    else:
        gridsize = xmag["gridsize"]

    # %% write the file
    h5write.maggrid(filename, xmag, gridsize)
