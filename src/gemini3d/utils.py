from __future__ import annotations
import subprocess
import os
import shutil
from pathlib import Path
import importlib.resources as pkgr

from datetime import datetime, timedelta
import typing as T
import logging
import importlib
import importlib.util

# the importlib.util import is necessary on non-Windows systems at least for MyPy 0.971

import xarray
import numpy as np


__all__ = ["get_pkg_file", "str2func", "to_datetime", "git_meta", "datetime2stem"]


def get_pkg_file(package: str, filename: str) -> Path:
    """Get a file from a package.
    This function works for Python 3.7, 3.8 using a deprecated method,
    and uses the recommended method for Python >= 3.9

    Parameters
    ----------
    package : str
        Package name.
    filename : str
        File name.

    Returns
    -------
    Path
        Path to the file.

    NOTE: this probably assumes the install is Zip safe
    """

    with pkgr.as_file(pkgr.files(package).joinpath(filename)) as f:
        return f


def str2func(name: str, path: Path | None = None) -> T.Callable:
    """
    expects one of (in priority order):

    0. file in "path" (if present)
    1. os.getcwd()/name.py containing function name()
    2. gemiin3d.<foo> <foo>/name.py module file containing function name()
    3. gemini3d.<foo> <foo>/__init__.py containing function name()


    Examples:

    1. os.getcwd()/perturb.py with function perturb()
    2. gemini3d.efield.Efield_erf returns function Efield_erf()
    """

    mod_name = ".".join(name.split(".")[:-1])
    func_name = name.split(".")[-1]

    if mod_name:
        try:
            # file with function of same name
            mod = importlib.import_module(name)
            return getattr(mod, func_name)
        except (ModuleNotFoundError, AttributeError):
            # __init__.py with function
            mod = importlib.import_module(mod_name)
            return getattr(mod, func_name)
    else:
        if path is None:
            # file in current working directory
            mod = importlib.import_module(func_name)
        else:
            # https://docs.python.org/3.10/library/importlib.html#importing-a-source-file-directly
            mod_file = path / (name + ".py")
            if not mod_file.is_file():
                raise FileNotFoundError(mod_file)
            spec = importlib.util.spec_from_file_location(name, mod_file)
            if spec is None:
                raise ModuleNotFoundError(f"{name} not found in {mod_file}")
            mod = importlib.util.module_from_spec(spec)
            if mod is None:
                raise ImportError(f"could not import {name} from {mod_file}")
            spec.loader.exec_module(mod)  # type: ignore

    return getattr(mod, func_name)


def to_datetime(times: xarray.DataArray | np.datetime64 | datetime) -> datetime:
    """

    Parameters
    ----------
    atimes : xarray time

    Returns
    -------
    times : list[datetime.datetime]
    """

    if isinstance(times, datetime):
        time = times
    elif isinstance(times, xarray.DataArray):
        time = times.data.squeeze()[()]
    elif isinstance(times, np.datetime64):
        time = times.squeeze()[()]  # type: ignore
    else:
        raise TypeError("expected datetime-like value")

    if isinstance(time, np.datetime64):
        time = time.astype("datetime64[us]").astype(datetime)  # type: ignore

    return time


def git_meta(path: Path | None = None) -> dict[str, str]:
    """
    provide metadata about a Git repo in a dictionary

    Dev note: use subprocess.run to avoid crashing program when Git meta is missing or broken (shallow clone)

    empty init in case can't read Git info
    this avoids needless if statements in consumers
    """

    git = shutil.which("git")
    meta = {
        "version": "",
        "remote": "",
        "branch": "",
        "commit": "",
        "porcelain": "false",
    }
    if not git:
        return meta

    if not path:
        if __file__:
            path = Path(__file__).resolve().parent
        else:
            return meta

    ret = subprocess.run(
        [git, "-C", str(path), "--version"], stdout=subprocess.PIPE, text=True
    )
    if ret.returncode != 0:
        logging.error("Git was not available or is too old")
        return meta

    meta["version"] = ret.stdout.strip()

    ret = subprocess.run([git, "-C", str(path), "rev-parse"])
    if ret.returncode != 0:
        logging.error(f"{path} is not a Git repo.")
        return meta

    ret = subprocess.run(
        [git, "-C", str(path), "rev-parse", "--abbrev-ref", "HEAD"],
        stdout=subprocess.PIPE,
        text=True,
    )
    if ret.returncode != 0:
        logging.error(f"{path} could not determine Git branch")
        return meta
    meta["branch"] = ret.stdout.strip()

    ret = subprocess.run(
        [git, "-C", str(path), "remote", "get-url", "origin"],
        stdout=subprocess.PIPE,
        text=True,
    )
    if ret.returncode != 0:
        logging.error(f"{path} could not determine Git remote")
        return meta
    meta["remote"] = ret.stdout.strip()

    ret = subprocess.run(
        [git, "-C", str(path), "describe", "--tags"], stdout=subprocess.PIPE, text=True
    )
    if ret.returncode != 0:
        ret = subprocess.run(
            [git, "-C", str(path), "rev-parse", "--short", "HEAD"],
            stdout=subprocess.PIPE,
            text=True,
        )
    if ret.returncode != 0:
        logging.error(f"{path} could not determine Git commit")
        return meta
    meta["commit"] = ret.stdout.strip()

    ret = subprocess.run(
        [git, "-C", str(path), "status", "--porcelain"], stdout=subprocess.PIPE, text=True
    )
    if ret.returncode != 0:
        logging.error(f"{path} could not determine Git status")
    msg = ret.stdout.strip()
    meta["porcelain"] = "false" if msg else "true"

    return meta


def get_cpu_count() -> int:
    """get a physical CPU count

    Note: len(os.sched_getaffinity(0)) and multiprocessing.cpu_count
    can give hyperthreaded CPU count rather than physical CPU count.
    Hence we use PSUtil. HWLOC would also work.

    Returns
    -------
    count: int
        detect number of physical CPU
    """

    extradiv = 1
    # without psutil, hyperthreaded CPU may overestimate physical count by factor of 2 (or more)
    try:
        import psutil

        max_cpu: int | None = psutil.cpu_count(logical=False)
        if max_cpu is None:
            max_cpu = psutil.cpu_count()
            extradiv = 2
    except ImportError:
        max_cpu = os.cpu_count()
        extradiv = 2

    max_cpu = 1 if max_cpu is None else max_cpu // extradiv

    return max_cpu


def datetime2stem(dt: datetime) -> str:
    """
    convert datetime to ymd_hourdec string for filename stem
    """

    dt = to_datetime(dt)

    assert isinstance(dt, datetime), "expect scalar datetime"

    return (
        dt.strftime("%Y%m%d")
        + f"_{dt.hour * 3600 + dt.minute * 60 + dt.second + dt.microsecond / 1e6:12.6f}"
    )


def filename2datetime(path: Path | str) -> datetime:
    """
    Gemini3D datafiles use a file naming pattern that we translate to a datetime

    path need not exist.
    """

    name = path.name if isinstance(path, Path) else path

    return datetime.strptime(name[:8], "%Y%m%d") + timedelta(seconds=float(name[9:21]))
