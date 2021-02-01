from __future__ import annotations
import subprocess
import os
import re
import shutil
from pathlib import Path
import xarray
import numpy as np
from datetime import datetime
import typing as T
import logging

try:
    import psutil
except ImportError:
    psutil = None
    # pip install psutil will improve CPU utilization.

Pathlike = T.Union[str, Path]

__all__ = ["to_datetime", "git_meta", "get_cpu_count", "datetime2ymd_hourdec"]


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
        time = times.values.squeeze()[()]  # numpy.datetime64
    elif isinstance(times, np.datetime64):
        time = times.squeeze()[()]
    else:
        raise TypeError("expected datetime-like value")

    if isinstance(time, np.datetime64):
        time = time.astype("datetime64[us]").astype(datetime)

    return time


def git_meta(path: Path = None) -> dict[str, str]:
    """
    provide metadata about a Git repo in a dictionary

    Dev note: use subprocess.run to avoid crashing program when Git meta is missing or broken (shallow clone)

    empty init in case can't read Git info
    this avoids needless if statements in consumers
    """

    git = shutil.which("git")
    meta = {
        "git_version": None,
        "remote": None,
        "branch": None,
        "commit": None,
        "porcelain": "false",
    }
    if not git:
        return meta

    if not path:
        if __file__:
            path = Path(__file__).resolve().parent
        else:
            return meta

    ret = subprocess.run([git, "-C", str(path), "--version"], stdout=subprocess.PIPE, text=True)
    if ret.returncode != 0:
        logging.error("Git was not available or is too old")
        return meta

    meta["git_version"] = ret.stdout.strip()

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
        logging.error(f"{path} coul not determine Git branch")
        return meta
    meta["branch"] = ret.stdout.strip()

    ret = subprocess.run([git, "-C", str(path), "remote", "-v"], stdout=subprocess.PIPE, text=True)
    if ret.returncode != 0:
        logging.error(f"{path} could not determine Git remote")
        return meta

    pat = re.compile(r"^origin\s+(.*)\s+\(fetch\)")
    mat = pat.match(ret.stdout.strip())
    if mat:
        meta["remote"] = mat.group(1)

    ret = subprocess.run(
        [git, "-C", str(path), "rev-parse", "--short", "HEAD"], stdout=subprocess.PIPE, text=True
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

    Note: len(os.sched_getaffinity(0)) and multiprocessing.cpu_count don't help either
    PSUtil is the most reliable, so we strongly recommend it.

    Returns
    -------
    count: int
        detect number of physical CPU
    """

    extradiv = 1
    max_cpu = None
    # without psutil, hyperthreaded CPU may overestimate physical count by factor of 2 (or more)
    if psutil is not None:
        max_cpu = psutil.cpu_count(logical=False)
        if max_cpu is None:
            max_cpu = psutil.cpu_count()
            extradiv = 2
    if max_cpu is None:
        max_cpu = os.cpu_count()
        if max_cpu is not None:
            extradiv = 2
        else:
            max_cpu = 1

    return max_cpu // extradiv


def datetime2ymd_hourdec(dt: datetime) -> str:
    """
    convert datetime to ymd_hourdec string for filename stem
    """

    dt = to_datetime(dt)

    assert isinstance(dt, datetime), "expect scalar datetime"

    return (
        dt.strftime("%Y%m%d")
        + f"_{dt.hour*3600 + dt.minute*60 + dt.second + dt.microsecond/1e6:12.6f}"
    )
