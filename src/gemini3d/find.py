"""
functions for finding files
"""

from datetime import datetime, timedelta
from pathlib import Path
import numpy as np

FILE_FORMATS = [".h5", ".nc", ".dat"]


def config(path: Path) -> Path:
    """ given a path or config filename, return the full path to config file """

    return find_stem(path, "config", "nml")


def simsize(path: Path, suffix: str = None) -> Path:
    """ gets path to simsize file """

    return find_stem(path, "simsize", suffix)


def frame(simdir: Path, time: datetime, file_format: str = None) -> Path:
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
            file_times.append(
                datetime.strptime(fn.name[:8], "%Y%m%d") + timedelta(seconds=float(fn.name[9:21]))
            )

        if file_times:
            afile_times = np.array(file_times)
            i = abs(afile_times - time).argmin()  # type: ignore

            if abs(afile_times[i] - time) <= MAX_OFFSET:
                return files[i]

    return None


def grid(path: Path) -> Path:
    """given a path or filename, return the full path to simgrid file
    we don't override FILE_FORMATS to allow outputs from a prior sim in a different
    file format to be used in this sim.
    """

    return find_stem(path, "simgrid")


def find_stem(path: Path, stem: str, suffix: str = None) -> Path:
    """find file containing stem """

    path = Path(path).expanduser()

    if path.is_file():
        if stem in path.stem:
            return path
        else:
            return find_stem(path.parent, stem, path.suffix)

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

    return None


def inputs(direc: Path, input_dir: Path = None) -> Path:
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
        input_path = None

    return input_path
