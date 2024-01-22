"""
functions related to running Gemini3D.
Either locally (laptop or interactive HPC) or creating an HPC batch script based on template.
"""

from __future__ import annotations
import typing
import os
import logging
import subprocess
import shutil
from pathlib import Path, PurePosixPath
import numpy as np

from . import find
from . import model
from . import write
from . import read
from . import wsl
from .utils import git_meta

try:
    from math import prod
except ImportError:
    # python < 3.8
    from numpy import prod  # type: ignore


def runner(pr: dict[str, typing.Any]) -> None:
    out_dir = check_outdir(pr["out_dir"])

    config_file = find.config(pr["config_file"])
    # load configuration to know what directories to check
    p = read.config(config_file)

    # we don't want to overwrite an expensive simulation output
    try:
        find.frame(out_dir, p["time"][0])
        raise FileExistsError(
            f"new simulation shouldn't have data in output directory: {out_dir}"
        )
    except FileNotFoundError:
        pass

    # %% setup grid and/or initial ionosphere state if needed
    for k in {"indat_size", "indat_grid", "indat_file"}:
        f = out_dir / p[k].expanduser()
        if pr.get("force") or not f.is_file():
            model.setup(p["nml"], out_dir)

    # %% estimate simulation RAM use on root MPI node
    ram_use_bytes = memory_estimate(out_dir)

    # %% setup Efield if needed
    if "E0dir" in p:
        E0dir = out_dir / p["E0dir"]
        if not E0dir.is_dir():
            model.setup(p["nml"], out_dir)

    # %% setup precip if needed
    if "precdir" in p:
        precdir = out_dir / p["precdir"]
        if not precdir.is_dir():
            model.setup(p["nml"], out_dir)

    # build checks
    gemexe = find.gemini_exe(pr.get("gemexe", ""))
    logging.info(f"gemini executable: {gemexe}")

    if os.name == "nt" and isinstance(gemexe, PurePosixPath):
        cmd = ["wsl", str(gemexe), str(wsl.win_path2wsl_path(out_dir))]
    else:
        cmd = [str(gemexe), str(out_dir)]

    # %% attempt dry run, but don't fail in case intended for HPC
    logging.info("Gemini dry run command:")
    logging.info(" ".join(cmd))
    proc = subprocess.run(cmd + ["-dryrun"])

    if proc.returncode != 0:
        raise RuntimeError(f"Gemini dry run failed. {' '.join(cmd)}")

    if pr.get("dryrun"):
        return None

    write.meta(out_dir / "setup_run.json", git_meta(gemexe.parent), p)

    try:
        import psutil

        avail_memory = psutil.virtual_memory().available
        if avail_memory < 2 * ram_use_bytes:
            logging.warning(
                f"""
Computer RAM available: {avail_memory / 1e9:.1} GB but simulation needs {ram_use_bytes / 1e9:.1}
Gemini3D may run out of RAM on this computer, which may make the run exceedingly slow or fail.
"""
            )
    except ImportError:
        pass

    print("\nBEGIN Gemini run with command:")
    print(" ".join(cmd), "\n")
    ret = subprocess.run(cmd).returncode
    if ret != 0:
        raise RuntimeError("Gemini run failed")


def memory_estimate(path: Path) -> int:
    """
    Estimate how must RAM Gemini3D will need.
    The current Gemini MPI architecture assumes the root node will use the most RAM,
    so that is the fundamental constraint.
    This neglects size of executable and library footprint,
    which would be minuscule in simulations larger than 1 GB where we
    are concerned about RAM limits.

    Parameters
    ----------

    path: pathlib.Path
        path to simgrid.*

    Returns
    -------

    memory_used: int
        estimated RAM usage (bytes)

    """

    SIZE = 8  # number of bytes for each element: real64 => 8 bytes
    PAD = 2  # factor to assume needed over variable itself (computations, work variables, ...)

    gs = read.grid(path, shape=True)

    grid_size = 0

    for k, v in gs.items():
        if k == "lx" or not isinstance(v, (tuple, list, np.ndarray)) or not v:
            continue
        print(k, v, grid_size)
        grid_size += int(prod(v))

    LSP = 7
    x1 = gs["x1"][0]
    x2 = gs["x2"][0]
    x3 = gs["x3"][0]

    Ns = LSP * x1 * x2 * x3
    Ts = Ns

    memory_used = SIZE * (grid_size + (Ns + Ts) * PAD)

    return memory_used


def check_compiler():
    fc = os.environ.get("FC")
    fc = shutil.which(fc) if fc else shutil.which("gfortran")
    if not fc:
        raise EnvironmentError("Cannot find Fortran compiler e.g. Gfortran")


def check_mpiexec(mpiexec: str, gemexe: Path) -> str:
    """
    check if specified mpiexec exists on this system.
    If not, error as most runs are exceedingly slow with one CPU core.
    """

    if not mpiexec:
        mpiexec = "mpiexec"

    mpi_root = os.environ.get("MPI_ROOT", None)
    if mpi_root:
        mpi_root += "/bin"

    mpiexec = shutil.which(mpiexec, path=mpi_root)
    if not mpiexec:
        raise FileNotFoundError(f"Cannot find mpiexec {mpiexec}")

    ret = subprocess.run([mpiexec, "-help"], capture_output=True, text=True, timeout=5)
    if ret.returncode != 0:
        raise RuntimeError(f"MPIexec error code {ret.returncode}\n{ret.stderr}")
    # %% check that compiler and MPIexec compatible
    if os.name != "nt":
        return mpiexec

    mpi_msg = ret.stdout.strip()
    ret = subprocess.run(
        [str(gemexe), "-compiler"],
        capture_output=True,
        text=True,
        timeout=5,
    )
    if ret.returncode != 0:
        raise RuntimeError(f"{gemexe} not executable")

    if "GNU" in ret.stdout.strip() and "Intel(R) MPI Library" in mpi_msg:
        raise EnvironmentError("MPIexec from MinGW is not compatible with Intel MPI")

    return mpiexec


def check_outdir(out_dir: str | Path) -> Path:
    out_dir = Path(out_dir).expanduser().resolve()
    if out_dir.is_file():
        raise NotADirectoryError(
            f"please specify output DIRECTORY, you specified {out_dir}"
        )
    if not out_dir.is_dir():
        out_dir.mkdir(parents=True, exist_ok=True)

    return out_dir
