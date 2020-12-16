from datetime import datetime
import numpy as np
from pathlib import Path
import typing as T
import sys

from .utils import git_meta
from .hdf5 import write as h5write
from .nc4 import write as ncwrite


def state(
    time: datetime,
    ns: np.ndarray,
    vs: np.ndarray,
    Ts: np.ndarray,
    out_file: Path,
):
    """
     WRITE STATE VARIABLE DATA.
    NOTE THAT WE don't write ANY OF THE ELECTRODYNAMIC
    VARIABLES SINCE THEY ARE NOT NEEDED TO START THINGS
    UP IN THE FORTRAN CODE.

    INPUT ARRAYS SHOULD BE TRIMMED TO THE CORRECT SIZE
    I.E. THEY SHOULD NOT INCLUDE GHOST CELLS
    """

    if out_file.suffix == ".h5":
        h5write.state(time, ns, vs, Ts, out_file.with_suffix(".h5"))
    elif out_file.suffix == ".nc":
        ncwrite.state(time, ns, vs, Ts, out_file.with_suffix(".nc"))
    else:
        raise ValueError(f"unknown file format {out_file.suffix}")


def data(dat: np.ndarray, out_file: Path, file_format: str, xg: T.Dict[str, T.Any] = None):

    if file_format == "h5":
        h5write.data(dat, out_file)
    elif file_format == "nc":
        ncwrite.data(dat, xg, out_file)
    else:
        raise ValueError(f"Unknown file format {file_format}")


def grid(p: T.Dict[str, T.Any], xg: T.Dict[str, T.Any]):
    """writes grid to disk

    Parameters
    ----------

    p: dict
        simulation parameters
    xg: dict
        grid values

    NOTE: we use .with_suffix() in case file_format was overriden by user
    that allows writing NetCDF4 and HDF5 by scripts using same input files
    """

    input_dir = p["indat_size"].parent
    if input_dir.is_file():
        raise OSError(f"{input_dir} is a file instead of directory")

    input_dir.mkdir(parents=True, exist_ok=True)

    if "format" not in p:
        p["format"] = p["indat_size"].suffix[1:]

    if p["format"] in ("hdf5", "h5"):
        h5write.grid(p["indat_size"].with_suffix(".h5"), p["indat_grid"].with_suffix(".h5"), xg)
    elif p["format"] in ("netcdf", "nc"):
        ncwrite.grid(p["indat_size"].with_suffix(".nc"), p["indat_grid"].with_suffix(".nc"), xg)
    else:
        raise ValueError(f'unknown file format {p["format"]}')

    meta(p["out_dir"] / "setup_meta.nml", git_meta(), "setup_python")


def Efield(E: T.Dict[str, T.Any], outdir: Path, file_format: str):
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

    if file_format in ("hdf5", "h5"):
        h5write.Efield(outdir, E)
    elif file_format in ("netcdf", "nc"):
        ncwrite.Efield(outdir, E)
    else:
        raise ValueError(f"unknown file format {file_format}")


def precip(precip: T.Dict[str, T.Any], outdir: Path, file_format: str):
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

    if file_format in ("hdf5", "h5"):
        h5write.precip(outdir, precip)
    elif file_format in ("netcdf", "nc"):
        ncwrite.precip(outdir, precip)
    else:
        raise ValueError(f"unknown file format {file_format}")


def meta(fn: Path, meta: T.Dict[str, str], namelist: str):
    """
    writes Namelist file with metadata
    """

    fn = fn.expanduser()
    if fn.is_dir():
        fn = fn / "setup_meta.nml"

    with fn.open(mode="a") as f:
        f.write(f"&{namelist}\n")

        # %% variable string values get quoted per NML standard
        f.write(f'python_version = "{sys.version}"\n')
        f.write('git_version = "{}"\n'.format(meta["git_version"]))
        f.write('git_remote = "{}"\n'.format(meta["remote"]))
        f.write('git_branch = "{}"\n'.format(meta["branch"]))
        f.write('git_commit = "{}"\n'.format(meta["commit"]))
        f.write('git_porcelain = "{}"\n'.format(meta["porcelain"]))

        f.write("/\n")