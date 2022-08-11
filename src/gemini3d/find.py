"""
functions for finding files
"""

from __future__ import annotations
from datetime import datetime, timedelta
from pathlib import Path
import shutil
import os
import subprocess

import numpy as np

from .utils import filename2datetime

EXE_PATHS = [
    ".",
    "bin",
    "build",
    "build/bin",
    "build/Release",
    "build/RelWithDebInfo",
    "build/Debug",
]


def config(path: Path) -> Path:
    """given a path or config filename, return the full path to config file"""

    return find_stem(path, stem="config", suffix="nml")


def simsize(path: Path) -> Path:
    """gets path to simsize file"""

    return find_stem(path, stem="simsize")


def gemini_exe(exe: str = None) -> Path | None:
    """
    find and check that Gemini executable can run on this system
    """

    name = "gemini3d.run"

    paths = [os.environ.get("GEMINI_ROOT"), os.environ.get("CMAKE_PREFIX_PATH")]

    if not exe:  # allow for default dict empty
        for p in paths:
            if p:
                for n in EXE_PATHS:
                    # print(p, n, name)
                    e = shutil.which(name, path=str(Path(p).expanduser() / n))
                    if e:
                        break
            if e:
                break

    if not e:
        return None

    # %% ensure Gemini3D executable is runnable
    gemexe = Path(e).expanduser()
    ret = subprocess.run(
        [str(gemexe)],
        capture_output=True,
        timeout=10,
        text=True,
        cwd=gemexe.parent,
    )
    if ret.returncode == 0:
        return gemexe

    if ret.returncode == 3221225781 and os.name == "nt":
        # Windows 0xc0000135, missing DLL
        raise RuntimeError(
            "On Windows, it's best to build Gemini3D with static libraries--including all numeric libraries "
            "such as LAPACK.\n"
            "Currently, we are missing a DLL on your system and gemini.bin with shared libs cannot run."
        )

    raise EnvironmentError(
        f"\n{gemexe} was not runnable on your platform--try rebuilding:\n"
        "gemini3d.setup()\n"
        f"{ret.stderr}"
    )


def msis_exe(root: str = None) -> Path | None:
    """
    find MSIS_SETUP executable
    """

    name = "msis_setup"

    paths = [os.environ.get("CMAKE_PREFIX_PATH"), os.environ.get("GEMINI_ROOT")]
    if root:
        paths.insert(0, root)

    for p in paths:
        if p:
            for n in EXE_PATHS:
                # print(p, n, name)
                msis_exe = shutil.which(name, path=str(Path(p).expanduser() / n))
                if msis_exe:
                    return Path(msis_exe)

    return None


def frame(simdir: Path, time: datetime) -> Path:
    """
    find frame closest to time
    """

    suffix = ".h5"

    simdir = Path(simdir).expanduser()

    stem = (
        time.strftime("%Y%m%d")
        + f"_{time.hour*3600 + time.minute*60 + time.second:05d}."
        + f"{time.microsecond:06d}"
    )

    fn = simdir / (stem + suffix)
    if fn.is_file():
        return fn

    # %% WORKAROUND for real32 file ticks. This will be removed when datetime-fortran is implemented
    MAX_OFFSET = timedelta(seconds=1)  # 10 ms precision, allow extra accumulated tolerance
    pat = time.strftime("%Y%m%d") + "_*"

    file_times = []
    files = list(simdir.glob(pat + suffix))
    for fn in files:
        file_times.append(filename2datetime(fn))

    if file_times:
        afile_times = np.array(file_times)
        i = abs(afile_times - time).argmin()  # type: ignore

        if abs(afile_times[i] - time) <= MAX_OFFSET:
            return files[i]

    raise FileNotFoundError(f"{stem}{suffix} not found in {simdir}")


def grid(path: Path) -> Path:
    """given a path or filename, return the full path to simgrid file"""

    return find_stem(path, stem="simgrid")


def find_stem(path: Path, stem: str, suffix: str = ".h5") -> Path:
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

    if isinstance(suffix, str):
        if not suffix.startswith("."):
            suffix = "." + suffix
        suffixes = [suffix]
    else:
        suffixes = suffix

    if path.is_dir():
        for p in (path, path / "inputs"):
            for suff in suffixes:
                f = p / (stem + suff)
                if f.is_file():
                    return f

    raise FileNotFoundError(f"{stem} not found in {path}")


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
        raise NotADirectoryError(direc)

    return input_path
