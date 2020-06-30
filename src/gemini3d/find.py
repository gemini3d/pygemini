"""
functions for finding files
"""

from datetime import datetime
import logging
from pathlib import Path

FILE_FORMATS = [".h5", ".nc", ".dat"]


def get_simsize_path(path: Path) -> Path:
    """ gets path to simsize file """

    path = Path(path).expanduser()

    if path.is_dir():
        for suffix in [".h5", ".nc", ".dat", ".mat"]:
            for stem in ["", "inputs/"]:
                fn = path / (f"{stem}simsize" + suffix)
                if fn.is_file():
                    return fn
    elif path.is_file():
        fn = path
        if fn.stem == "simsize":
            return fn

        for stem in ["", "inputs/"]:
            fn = path.parent / (f"{stem}simsize" + path.suffix)
            if fn.is_file():
                return fn

    logging.error(f"simsize not found in {path}")
    return None


def get_frame_filename(simdir: Path, time: datetime, file_format: str = None) -> Path:
    """
    the frame filenames can have different file formats
    """

    simdir = Path(simdir).expanduser().resolve(True)

    stem = (
        time.strftime("%Y%m%d")
        + f"_{time.hour*3600 + time.minute*60 + time.second:05d}."
        + f"{time.microsecond:06d}"[:5]
    )

    suffixes = [f".{file_format}"] if file_format else FILE_FORMATS

    for ext in suffixes:
        for tick in ("0", "1"):
            fn = simdir / (stem + tick + ext)
            if fn.is_file():
                return fn

    raise FileNotFoundError(f"could not find data file in {simdir} at {time}")


def get_grid_filename(path: Path) -> Path:
    """ given a path or filename, return the full path to simgrid file
    we don't override FILE_FORMATS to allow outputs from a prior sim in a different
    file format to be used in this sim.
    """

    path = Path(path).expanduser().resolve()

    if path.is_dir():
        for p in (path, path / "inputs"):
            for suff in FILE_FORMATS:
                file = p / ("simgrid" + suff)
                if file.is_file():
                    return file
    elif path.is_file():
        name = path.name
        path = path.parent
        for p in (path, path / "inputs"):
            file = p / name
            if file.is_file():
                return file

    logging.error(f"could not find grid file in {path}")
    return None
