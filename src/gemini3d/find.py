"""
functions for finding files
"""

from datetime import datetime, timedelta
from pathlib import Path
import numpy as np

FILE_FORMATS = [".h5", ".nc", ".dat"]


def config(path: Path) -> Path:
    """ given a path or config filename, return the full path to config file """

    if not path:
        return None

    path = Path(path).expanduser().resolve()

    if path.is_file():
        return path

    if path.is_dir():
        for p in (path, path / "inputs"):
            for suff in (".nml", ".ini"):
                for f in p.glob("config*" + suff):
                    if f.is_file():
                        return f

    return None


def simsize(path: Path, suffix: str = None) -> Path:
    """ gets path to simsize file """

    path = Path(path).expanduser().resolve()

    if suffix:
        if isinstance(suffix, str):
            suffixes = [suffix]
        else:
            suffixes = suffix
    else:
        suffixes = [".h5", ".nc", ".dat", ".mat"]

    if path.is_dir():
        for suffix in suffixes:
            for stem in ["", "inputs/"]:
                fn = path / (f"{stem}simsize" + suffix)
                if fn.is_file():
                    return fn
    elif path.is_file():
        if path.stem == "simsize":
            return path

        for stem in ["", "inputs/"]:
            fn = path.parent / (f"{stem}simsize" + path.suffix)
            if fn.is_file():
                return fn

    return None


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
            i = abs(afile_times - time).argmin()

            if abs(afile_times[i] - time) <= MAX_OFFSET:
                return files[i]

    return None


def grid(path: Path) -> Path:
    """given a path or filename, return the full path to simgrid file
    we don't override FILE_FORMATS to allow outputs from a prior sim in a different
    file format to be used in this sim.
    """

    path = Path(path).expanduser().resolve()

    if path.is_dir():
        for p in (path, path / "inputs"):
            for suff in FILE_FORMATS:
                f = p / ("simgrid" + suff)
                if f.is_file():
                    return f
    elif path.is_file():
        name = path.name
        path = path.parent
        for p in (path, path / "inputs"):
            f = p / name
            if f.is_file():
                return f

    return None
