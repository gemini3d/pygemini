from __future__ import annotations
import numpy as np
import xarray
from pathlib import Path
import typing as T
import sys

from .utils import git_meta
from .hdf5 import write as h5write
from .nc4 import write as ncwrite
from . import namelist


def state(out_file: Path, dat: xarray.Dataset, file_format: str = None, **kwargs):
    """
    WRITE STATE VARIABLE DATA.
    NOTE: WE don't write ANY OF THE ELECTRODYNAMIC
    VARIABLES SINCE THEY ARE NOT NEEDED TO START THINGS
    UP IN THE FORTRAN CODE.

    INPUT ARRAYS SHOULD BE TRIMMED TO THE CORRECT SIZE
    I.E. THEY SHOULD NOT INCLUDE GHOST CELLS
    """

    ext = file_format if file_format else out_file.suffix[1:]

    # %% allow overriding "dat"
    if "time" in kwargs:
        dat.attrs["time"] = kwargs["time"]

    for k in {"ns", "vs1", "Ts"}:
        if k in kwargs:
            dat[k] = (("species", "x1", "x2", "x3"), kwargs[k])

    if "Phitop" in kwargs:
        dat["Phitop"] = (("x2", "x3"), kwargs["Phitop"])

    # %% dispatch to format-specific writers
    if ext == "h5":
        h5write.state(out_file.with_suffix(".h5"), dat)
    elif ext == "nc":
        ncwrite.state(out_file.with_suffix(".nc"), dat)
    else:
        raise ValueError(f"unknown file format {ext}")


def data(out_file: Path, dat: np.ndarray, file_format: str, xg: dict[str, T.Any] = None):

    if file_format == "h5":
        h5write.data(out_file, dat)
    elif file_format == "nc":
        ncwrite.data(out_file, dat, xg)
    else:
        raise ValueError(f"Unknown file format {file_format}")


def grid(cfg: dict[str, T.Any], xg: dict[str, T.Any], *, file_format: str = None):
    """writes grid to disk

    Parameters
    ----------

    cfg: dict
        simulation parameters
    xg: dict
        grid values

    NOTE: we use .with_suffix() in case file_format was overriden by user
    that allows writing NetCDF4 and HDF5 by scripts using same input files
    """

    input_dir = cfg["indat_size"].parent
    if input_dir.is_file():
        raise OSError(f"{input_dir} is a file instead of directory")

    input_dir.mkdir(parents=True, exist_ok=True)

    if not file_format:
        file_format = cfg["file_format"] if "file_format" in cfg else cfg["indat_size"].suffix[1:]

    if file_format == "h5":
        h5write.grid(cfg["indat_size"].with_suffix(".h5"), cfg["indat_grid"].with_suffix(".h5"), xg)
    elif file_format == "nc":
        ncwrite.grid(cfg["indat_size"].with_suffix(".nc"), cfg["indat_grid"].with_suffix(".nc"), xg)
    else:
        raise ValueError(f'unknown file format {cfg["file_format"]}')

    meta(cfg["out_dir"] / "setup_meta.nml", git_meta(), "setup_python")


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

    if file_format == "h5":
        h5write.Efield(outdir, E)
    elif file_format == "nc":
        ncwrite.Efield(outdir, E)
    else:
        raise ValueError(f"unknown file format {file_format}")


def precip(precip: dict[str, T.Any], outdir: Path, file_format: str):
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

    if file_format == "h5":
        h5write.precip(outdir, precip)
    elif file_format == "nc":
        ncwrite.precip(outdir, precip)
    else:
        raise ValueError(f"unknown file format {file_format}")


def meta(fn: Path, meta: dict[str, str], nml: str):
    """
    writes Namelist file with metadata
    """

    fn = fn.expanduser()
    if fn.is_dir():
        fn = fn / "setup_meta.nml"

    meta["python_version"] = sys.version

    namelist.write(fn, nml, meta)
