"""
functions for finding files
"""

from datetime import datetime, timedelta
from pathlib import Path
import typing as T

import numpy as np

from .utils import filename2datetime

FILE_FORMATS = [".h5", ".nc"]


def config(path: Path) -> Path:
    """given a path or config filename, return the full path to config file"""

    return find_stem(path, stem="config", suffix="nml")


def simsize(path: Path, suffix: str = None) -> Path:
    """gets path to simsize file"""

    return find_stem(path, stem="simsize", suffix=suffix)


def frame(simdir: Path, time: datetime, *, file_format: str = None) -> Path:
    """
    the frame filenames can have different file formats
    """

    simdir = Path(simdir).expanduser()

    stem = (
        time.strftime("%Y%m%d")
        + f"_{time.hour*3600 + time.minute*60 + time.second:05d}."
        + f"{time.microsecond:06d}"
    )

    suffixes = [f".{file_format}"] if file_format else FILE_FORMATS

    for ext in suffixes:
        fn = simdir / (stem + ext)
        if fn.is_file():
            return fn

    # %% WORKAROUND for real32 file ticks. This will be removed when datetime-fortran is implemented
    MAX_OFFSET = timedelta(seconds=1)  # 10 ms precision, allow extra accumulated tolerance
    pat = time.strftime("%Y%m%d") + "_*"
    for ext in suffixes:
        file_times = []
        files = list(simdir.glob(pat + ext))
        for fn in files:
            file_times.append(filename2datetime(fn))

        if file_times:
            afile_times = np.array(file_times)
            i = abs(afile_times - time).argmin()  # type: ignore

            if abs(afile_times[i] - time) <= MAX_OFFSET:
                return files[i]

    raise FileNotFoundError(f"{stem}{suffixes} not found in {simdir}")


def grid(path: Path, *, suffix=None) -> Path:
    """given a path or filename, return the full path to simgrid file
    we don't override FILE_FORMATS to allow outputs from a prior sim in a different
    file format to be used in this sim.
    """

    return find_stem(path, stem="simgrid", suffix=suffix)


def find_stem(path: Path, stem: str, suffix: str = None) -> Path:
    """find file containing stem"""

    path = Path(path).expanduser()

    if path.is_file():
        if stem in path.stem:
            return path
        else:
            found = find_stem(path.parent, stem, path.suffix)
            if not found:
                raise FileNotFoundError(f"{stem} not found in {path.parent}")
            return found

    if suffix:
        if isinstance(suffix, str):
            if not suffix.startswith("."):
                suffix = "." + suffix
            suffixes = [suffix]
        else:
            suffixes = suffix
    else:
        suffixes = FILE_FORMATS

    if path.is_dir():
        for p in (path, path / "inputs"):
            for suff in suffixes:
                f = p / (stem + suff)
                if f.is_file():
                    return f

    raise FileNotFoundError(f"{stem} not found in {path}")


def inputs(direc: Path, input_dir: T.Optional[Path] = None) -> Path:
    """
    find input parameter directory

    direc: pathlib.Path
        top level simulation dir
    input_dir: pathlib.Path
        relative to top level, the parameter dir. E.g. inputs/precip, input/Efield
    """

    direc = Path(direc).expanduser()
    if input_dir:
        input_path = Path(input_dir).expanduser()
        if not input_path.is_absolute():
            input_path = direc / input_path
    else:
        input_path = direc

    if not input_path.is_dir():
        raise NotADirectoryError(direc)

    return input_path
