"""
struct manpage:
https://docs.python.org/3/library/struct.html#struct-format-strings
"""
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import typing as T

from . import raw
from .config import read_config
from .base import get_simsize
from .find import get_frame_filename, get_grid_filename
from . import matlab

try:
    import h5py
    from . import hdf
except (ImportError, AttributeError):
    # must be ImportError not ModuleNotFoundError for botched HDF5 linkage
    hdf = None

try:
    from netCDF4 import Dataset
    from . import nc4
except ImportError:
    nc4 = None


def readgrid(path: Path, file_format: str = None) -> T.Dict[str, np.ndarray]:

    fn = get_grid_filename(path)
    if not fn:
        return {}

    if not file_format:
        file_format = fn.suffix[1:]

    if file_format == "dat":
        grid = raw.readgrid(fn.with_suffix(".dat"))
    elif file_format == "h5":
        if hdf is None:
            raise ModuleNotFoundError("pip install h5py")
        grid = hdf.readgrid(fn.with_suffix(".h5"))
    elif file_format == "nc":
        if nc4 is None:
            raise ModuleNotFoundError("pip install netcdf4")
        grid = nc4.readgrid(fn.with_suffix(".nc"))
    elif file_format == "mat":
        grid = matlab.read_grid(fn.with_suffix(".mat"))
    else:
        raise ValueError(f"Unknown file type {fn}")

    return grid


def readdata(
    fn: Path, file_format: str = None, *, cfg: T.Dict[str, T.Any] = None, E0dir: Path = None
) -> T.Dict[str, T.Any]:
    """
    knowing the filename for a simulation time step, read the data for that time step

    Parameters
    ----------
    fn: pathlib.Path
        filename for this timestep
    file_format: str
        specify file extension of data files
    cfg: dict
        to avoid reading config.nml
    E0dir: pathlib.Path
        E0 directory

    Returns
    -------
    dat: dict
        simulation outputs as numpy.ndarray
    """

    wavelength = [
        "3371",
        "4278",
        "5200",
        "5577",
        "6300",
        "7320",
        "10400",
        "3466",
        "7774",
        "8446",
        "3726",
        "LBH",
        "1356",
        "1493",
        "1304",
    ]

    fn = Path(fn).expanduser()
    fn_aurora = fn.parent / "aurmaps" / fn.name
    if E0dir:
        fn_Efield = E0dir / fn.name

    input_dir = fn.parent / "inputs"
    if not cfg:
        cfg = read_config(input_dir)

    if not file_format:
        file_format = fn.suffix[1:]

    if file_format == "dat":
        lxs = get_simsize(fn.parent / "inputs/simsize.dat")

        flag = cfg.get("flagoutput")
        if flag == 0:
            dat = raw.loadframe3d_curvne(fn, lxs)
        elif flag == 1:
            dat = raw.loadframe3d_curv(fn, lxs)
        elif flag == 2:
            dat = raw.loadframe3d_curvavg(fn, lxs)
        else:
            raise ValueError(f"Unsure how to read {fn} with flagoutput {flag}")

        if fn_aurora.is_file():
            dat.update(raw.loadglow_aurmap(fn_aurora, lxs, len(wavelength)))
            dat["wavelength"] = wavelength

    elif file_format == "h5":
        if hdf is None:
            raise ModuleNotFoundError("pip install h5py")

        # detect output type
        with h5py.File(fn, "r") as f:
            if "flagoutput" in f:
                flag = f["/flagoutput"][()]
            elif "flagoutput" in cfg:
                flag = cfg["flagoutput"]
            else:
                if "ne" in f and f["/ne"].ndim == 3:
                    flag = 0
                elif "nsall" in f and f["/nsall"].ndim == 4:
                    flag = 1
                elif "neall" in f and f["/neall"].ndim == 3:
                    flag = 2
                else:
                    raise ValueError(f"please specify flagoutput in config.nml or {fn}")

        if flag == 0:
            dat = hdf.loadframe3d_curvne(fn)
        elif flag == 1:
            dat = hdf.loadframe3d_curv(fn)
        elif flag == 2:
            dat = hdf.loadframe3d_curvavg(fn)
        else:
            raise ValueError(f"Unsure how to read {fn} with flagoutput {flag}")

        if fn_aurora.is_file():
            dat.update(hdf.loadglow_aurmap(fn_aurora))
            dat["wavelength"] = wavelength
    elif file_format == "nc":
        if nc4 is None:
            raise ModuleNotFoundError("pip install netcdf4")

        # detect output type
        with Dataset(fn, "r") as f:
            if "flagoutput" in f.variables:
                flag = f["flagoutput"][()]
            elif "flagoutput" in cfg:
                flag = cfg["flagoutput"]
            else:
                if "ne" in f.variables and f["ne"].ndim == 3:
                    flag = 0
                elif "nsall" in f.variables and f["nsall"].ndim == 4:
                    flag = 1
                elif "neall" in f.variables and f["neall"].ndim == 3:
                    flag = 2
                else:
                    raise ValueError(f"please specify flagoutput in config.nml or {fn}")

        if flag == 0:
            dat = nc4.loadframe3d_curvne(fn)
        elif flag == 1:
            dat = nc4.loadframe3d_curv(fn)
        elif flag == 2:
            dat = nc4.loadframe3d_curvavg(fn)
        else:
            raise ValueError(f"Unsure how to read {fn} with flagoutput {flag}")

        if fn_aurora.is_file():
            dat.update(nc4.loadglow_aurmap(fn_aurora))
            dat["wavelength"] = wavelength
    else:
        raise ValueError(f"Unknown file type {fn}")

    if E0dir and fn_Efield.is_file():
        dat.update(read_Efield(fn_Efield))

    return dat


def read_Efield(fn: Path, file_format: str = None) -> T.Dict[str, T.Any]:
    """ load Efield data "Efield_inputs"

    Parameters
    ----------
    fn: pathlib.Path
        filename for this timestep

    Returns
    -------
    dat: dict of np.ndarray
        electric field
    """

    fn = Path(fn).expanduser().resolve(strict=True)

    if not file_format:
        file_format = fn.suffix[1:]

    if file_format == "h5":
        if hdf is None:
            raise ModuleNotFoundError("pip install h5py")
        E = hdf.read_Efield(fn)
    elif file_format == "nc":
        if nc4 is None:
            raise ModuleNotFoundError("pip install netcdf4")
        E = nc4.read_Efield(fn)
    elif file_format == "dat":
        E = raw.read_Efield(fn)
    else:
        raise ValueError(f"Unknown file type {fn}")

    return E


def read_precip(fn: Path, file_format: str = None) -> T.Dict[str, T.Any]:
    """ load precipitation to disk

    Parameters
    ----------
    fn: pathlib.Path
        path to precipitation file
    file_format: str
        file format to read

    Returns
    -------
    dat: dict
        precipitation
    """

    fn = Path(fn).expanduser().resolve(strict=True)

    if not file_format:
        file_format = fn.suffix[1:]

    if file_format == "h5":
        if hdf is None:
            raise ImportError("pip install h5py")
        dat = hdf.read_precip(fn)
    elif file_format == "nc":
        if nc4 is None:
            raise ImportError("pip install netcdf4")
        dat = nc4.read_precip(fn)
    else:
        raise ValueError(f"unknown file format {file_format}")

    return dat


def read_state(file: Path,) -> T.Dict[str, T.Any]:
    """
    load inital condition data
    """

    if file.suffix == ".h5":
        if hdf is None:
            raise ImportError("pip install h5py")
        dat = hdf.read_state(file)
    elif file.suffix == ".nc":
        if nc4 is None:
            raise ImportError("pip install netcdf4")
        dat = nc4.read_state(file)
    else:
        raise ValueError(f"unknown file format {file.suffix}")

    return dat


def datetime_range(start: datetime, stop: datetime, step: timedelta) -> T.List[datetime]:

    """
    Generate range of datetime

    Parameters
    ----------
    start : datetime
        start time
    stop : datetime
        stop time
    step : timedelta
        time step

    Returns
    -------
    times : list of datetime
        times requested
    """
    return [start + i * step for i in range((stop - start) // step)]


def loadframe(simdir: Path, time: datetime, file_format: str = None) -> T.Dict[str, T.Any]:
    """
    This is what users should normally use.
    load a frame of simulation data, automatically selecting the correct
    functions based on simulation parameters

    Parameters
    ----------
    simdir: pathlib.Path
        top-level directory of simulation output
    time: datetime.datetime
        time to load from simulation output
    file_format: str, optional
        "hdf5", "nc" for hdf5 or netcdf4 respectively

    Returns
    -------
    dat: dict
        simulation output for this time step
    """

    return readdata(get_frame_filename(simdir, time, file_format), file_format)
