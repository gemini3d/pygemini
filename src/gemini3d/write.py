from __future__ import annotations
from pathlib import Path
import typing as T
import sys
import logging
import json

import xarray

from .utils import git_meta
from .hdf5 import write as h5write
from .nc4 import write as ncwrite


def state(out_file: Path, dat: xarray.Dataset, file_format: str = None, **kwargs):
    """
    WRITE STATE VARIABLE DATA.
    NOTE: WE don't write ANY OF THE ELECTRODYNAMIC
    VARIABLES SINCE THEY ARE NOT NEEDED TO START THINGS
    UP IN THE FORTRAN CODE.

    INPUT ARRAYS SHOULD BE TRIMMED TO THE CORRECT SIZE
    I.E. THEY SHOULD NOT INCLUDE GHOST CELLS
    """

    ext = file_format if file_format else out_file.suffix

    # %% allow overriding "dat"
    if "time" in kwargs:
        dat.attrs["time"] = kwargs["time"]

    for k in {"ns", "vs1", "Ts"}:
        if k in kwargs:
            dat[k] = (("species", "x1", "x2", "x3"), kwargs[k])

    if "Phitop" in kwargs:
        dat["Phitop"] = (("x2", "x3"), kwargs["Phitop"])

    # %% dispatch to format-specific writers
    if ext.endswith("h5"):
        h5write.state(out_file.with_suffix(".h5"), dat)
    elif ext.endswith("nc"):
        ncwrite.state(out_file.with_suffix(".nc"), dat)
    else:
        raise ValueError(f"unknown file format {ext}")


def data(out_file: Path, dat: xarray.Dataset, file_format: str, xg: dict[str, T.Any] = None):
    """
    used by scripts/convert_data.py
    """

    if file_format.endswith("h5"):
        h5write.data(out_file, dat)
    elif file_format.endswith("nc"):
        assert isinstance(xg, dict)
        ncwrite.data(out_file, dat, xg)
    else:
        raise ValueError(f"Unknown file format {file_format}")


def grid(cfg: dict[str, T.Any], xg: dict[str, T.Any], *, file_format: str = ""):
    """writes grid to disk

    Parameters
    ----------

    cfg: dict
        simulation parameters
    xg: dict
        grid values

    NOTE: we use .with_suffix() in case file_format was overridden by user
    that allows writing NetCDF4 and HDF5 by scripts using same input files
    """

    input_dir = cfg["indat_size"].parent
    if input_dir.is_file():
        raise OSError(f"{input_dir} is a file instead of directory")

    input_dir.mkdir(parents=True, exist_ok=True)

    if not file_format:
        file_format = cfg.get("file_format", cfg["indat_size"].suffix)

    if file_format.endswith("h5"):
        h5write.grid(cfg["indat_size"].with_suffix(".h5"), cfg["indat_grid"].with_suffix(".h5"), xg)
    elif file_format.endswith("nc"):
        ncwrite.grid(cfg["indat_size"].with_suffix(".nc"), cfg["indat_grid"].with_suffix(".nc"), xg)
    else:
        raise ValueError(f'unknown file format {cfg["file_format"]}')

    meta(input_dir / "setup_grid.json", git_meta(), cfg)


def Efield(E: xarray.Dataset, outdir: Path, file_format: str):
    """writes E-field to disk

    Parameters
    ----------

    E: dict
        E-field values
    outdir: pathlib.Path
        directory to write files into
    file_format: str
        requested file format to write
    """

    print("write E-field data to", outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if file_format.endswith("h5"):
        h5write.Efield(outdir, E)
    elif file_format.endswith("nc"):
        ncwrite.Efield(outdir, E)
    else:
        raise ValueError(f"unknown file format {file_format}")


def precip(precip: xarray.Dataset, outdir: Path, file_format: str):
    """writes precipitation to disk

    Parameters
    ----------
    precip: dict
        preicipitation values
    outdir: pathlib.Path
        directory to write files into
    file_format: str
        requested file format to write
    """

    print("write precipitation data to", outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if file_format.endswith("h5"):
        h5write.precip(outdir, precip)
    elif file_format.endswith("nc"):
        ncwrite.precip(outdir, precip)
    else:
        raise ValueError(f"unknown file format {file_format}")


def meta(fn: Path, git_meta: dict[str, str], cfg: dict[str, T.Any]):
    """
    writes JSON file with sim setup metadata
    """

    fn = fn.expanduser()
    if fn.is_dir():
        raise FileNotFoundError(f"{fn} is a directory, but I need a JSON file name to write.")

    jm = {"python": {"platform": sys.platform, "version": sys.version}, "git": git_meta}

    if "eq_dir" in cfg:
        # JSON does not allow unescaped backslash
        jm["equilibrium"] = {"eq_dir": cfg["eq_dir"].as_posix()}
        hf = cfg["eq_dir"] / "sha256sum.txt"
        if hf.is_file():
            jm["equilibrium"]["sha256"] = hf.read_text().strip()

    js = json.dumps(jm, sort_keys=True, indent=2)

    fn.write_text(js)


def maggrid(filename: Path, xmag: dict[str, T.Any]):

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
    if not filename.parent.is_dir():
        raise FileNotFoundError(f"{filename.parent} parent directory does not exist")

    if filename.suffix.endswith("h5"):
        h5write.maggrid(filename, xmag, gridsize)
    else:
        raise ValueError(f"{filename.suffix} not handled yet. Please open GitHub issue.")
