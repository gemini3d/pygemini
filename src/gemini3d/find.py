"""
functions for finding files
"""

from __future__ import annotations
from datetime import datetime, timedelta
from pathlib import Path, PurePosixPath
import shutil
import os
import subprocess
import logging

import numpy as np

from .utils import filename2datetime
from . import wsl


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

    p = find_stem(path, stem="config", suffix=".nml")
    if not p:
        raise FileNotFoundError(f"config.nml not found in {path}")
    return p


def simsize(path: Path) -> Path:
    """gets path to simsize file"""

    p = find_stem(path, stem="simsize", suffix=".h5")
    if not p:
        raise FileNotFoundError(f"simsize.h5 not found in {path}")
    return p


def executable(name: str, root: Path | None = None) -> Path:
    if not name:
        raise ValueError("executable name must be non-empty")

    ep = Path(name).expanduser()
    if ep.is_file():
        return ep

    exe_paths = EXE_PATHS
    if name == "msis_setup":
        exe_paths.insert(2, "build/msis")

    paths = (root, os.environ.get("GEMINI_ROOT"), os.environ.get("CMAKE_PREFIX_PATH"))

    for p in paths:
        if not p:
            continue
        p = Path(p).expanduser()

        for n in exe_paths:
            e = p / n / name
            logging.debug(f"checking {e} for existance and executable permission")
            if wsl.is_wsl_path(p):
                # shutil.which() doesn't work on WSL paths
                if e.is_file():
                    return wsl.win_path2wsl_path(e)  # type: ignore
            else:
                if exe := shutil.which(e, path=p / n):
                    return Path(exe)

    raise FileNotFoundError(f"{name} not found, search paths: {paths}")


def gemini_exe(name: str = "gemini3d.run", root: Path | None = None) -> Path:
    """
    find and check that Gemini executable can run on this system
    """

    if not name:
        name = "gemini3d.run"

    exe = executable(name, root)

    # %% ensure Gemini3D executable is runnable
    if os.name == "nt" and isinstance(exe, PurePosixPath):
        cmd0 = ["wsl", str(exe), "-h"]
    else:
        cmd0 = [str(exe), "-h"]

    ret = subprocess.run(
        cmd0,
        capture_output=True,
        timeout=10,
        text=True,
    )

    if ret.returncode == 0:
        return exe

    if ret.returncode == 3221225781 and os.name == "nt":
        # Windows 0xc0000135, missing DLL
        raise RuntimeError(
            "On Windows, it's best to build Gemini3D with static libraries--including all numeric libraries "
            "such as LAPACK.\n"
            "Currently, we are missing a DLL on your system and gemini.bin with shared libs cannot run."
        )

    raise EnvironmentError(
        f"\n{exe} was not runnable on your platform--try rebuilding Gemini3D\n"
        f"{ret.stderr}"
    )


def frame(simdir: Path, time: datetime) -> Path:
    """
    find frame closest to time
    """

    suffix = ".h5"

    simdir = Path(simdir).expanduser()

    stem = (
        time.strftime("%Y%m%d")
        + f"_{time.hour * 3600 + time.minute * 60 + time.second:05d}."
        + f"{time.microsecond:06d}"
    )

    fn = simdir / (stem + suffix)
    if fn.is_file():
        return fn

    # %% WORKAROUND for real32 file ticks. This will be removed when datetime-fortran is implemented
    MAX_OFFSET = timedelta(
        seconds=1
    )  # 10 ms precision, allow extra accumulated tolerance
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

    for s in ("amrgrid", "simgrid"):
        p = find_stem(path, stem=s)
        if p:
            return p
    raise FileNotFoundError(f"{s} not found in {path}")


def find_stem(path: Path, stem: str, suffix: str = ".h5") -> Path | None:
    """find file containing stem"""

    path = Path(path).expanduser()

    if path.is_file():
        if stem in path.stem:
            return path
        else:
            found = find_stem(path.parent, stem, path.suffix)
            if found:
                return found
    elif path.is_dir():
        for p in (path, path / "inputs"):
            f = p / (stem + suffix)
            if f.is_file():
                return f

    return None


def inputs(direc: Path, input_dir: Path | None = None) -> Path:
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
