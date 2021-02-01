"""
functions related to running gemini.bin.
Either locally (laptop or interactive HPC) or creating an HPC batch script based on template.

Builds gemini.bin if not found.
"""

from __future__ import annotations
import typing as T
import os
import logging
import subprocess
import shutil
from pathlib import Path
import numpy as np
import psutil

from . import find
from . import mpi
from . import cmake
from .hpc import hpc_batch_detect, hpc_batch_create
from . import model
from . import write
from . import read
from .utils import git_meta


Pathlike = T.Union[str, Path]


def runner(pr: dict[str, T.Any]) -> None:

    out_dir = check_outdir(pr["out_dir"])

    config_file = find.config(pr["config_file"])
    # load configuration to know what directories to check
    p = read.config(config_file)
    if not p:
        raise FileNotFoundError(f"{config_file} does not appear to contain config.nml")

    # we don't want to overwrite an expensive simulation output
    if find.frame(out_dir, p["time"][0], p.get("out_format")):
        raise FileExistsError(
            f"a fresh simulation should not have data in output directory: {out_dir}"
        )

    # %% setup grid and/or initial ionosphere state if needed
    for k in ("indat_size", "indat_grid", "indat_file"):
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
    gemexe = get_gemini_exe(pr.get("gemexe"))
    logging.info(f"gemini executable: {gemexe}")

    mpiexec = check_mpiexec(pr.get("mpiexec"), gemexe)
    if mpiexec:
        logging.info(f"mpiexec: {mpiexec}")
        Nmpi = mpi.count(out_dir / p["indat_size"], pr.get("cpu_count"))
        mpi_cmd = [mpiexec, "-n", str(Nmpi)]
    else:
        mpi_cmd = []

    cmd = mpi_cmd + [str(gemexe), str(out_dir)]

    if pr.get("out_format"):
        cmd += ["-out_format", pr["out_format"]]

    # %% attempt dry run, but don't fail in case intended for HPC
    logging.info("Gemini dry run command:")
    logging.info(" ".join(cmd))
    proc = subprocess.run(cmd + ["-dryrun"])

    if proc.returncode != 0:
        raise RuntimeError("Gemini dry run failed.")

    if pr.get("dryrun"):
        return None

    write.meta(out_dir / "setup_meta.nml", git_meta(gemexe.parent), "setup_gemini")

    batcher = hpc_batch_detect()
    if batcher:
        job_file = hpc_batch_create(batcher, out_dir, cmd)
        print("Please examine batch file", job_file, "and when ready submit the job as usual.")
    else:
        avail_memory = psutil.virtual_memory().available
        if avail_memory < 2 * ram_use_bytes:
            logging.warning(
                f"""
Computer RAM available: {avail_memory/1e9:.1} GB but simulation needs {ram_use_bytes/1e9:.1}
Gemini3D may run out of RAM on this computer, which may make the run exceedingly slow or fail.
"""
            )
        print("\nBEGIN Gemini run with command:")
        print(" ".join(cmd), "\n")
        ret = subprocess.run(cmd).returncode
        if ret != 0:
            raise RuntimeError("Gemini run failed")


def memory_estimate(path: Path) -> int:
    """
    Estimate how must RAM gemini.bin will need.
    The current Gemini MPI architecture assumes the root node will use the most RAM,
    so that is the fundamental constraint.
    This neglects size of executable and library footprint,
    which would be miniscule in simulations larger than 1 GB where we
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
        if k == "lxs" or not isinstance(v, (tuple, list, np.ndarray)):
            continue
        grid_size += np.prod(v)  # type: ignore

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


def check_mpiexec(mpiexec: Pathlike, gemexe: Pathlike) -> str:
    """check if specified mpiexec exists on this system
    If not, fall back to not using MPI (slow, but still works)."""

    if not mpiexec:
        mpiexec = "mpiexec"
    mpi_root = os.environ.get("MPI_ROOT")
    if mpi_root:
        mpi_root += "/bin"

    mpiexec = shutil.which(mpiexec, path=str(mpi_root))
    if not mpiexec:
        return None

    ret = subprocess.run([mpiexec, "-help"], stdout=subprocess.PIPE, text=True, timeout=5)
    if ret.returncode != 0:
        return None
    # %% check that compiler and MPIexec compatible
    if os.name != "nt":
        return mpiexec

    mpi_msg = ret.stdout.strip()
    ret = subprocess.run([str(gemexe), "-compiler"], stdout=subprocess.PIPE, text=True, timeout=5)
    if ret.returncode != 0:
        raise EnvironmentError(f"{gemexe} not executable")

    if "GNU" in ret.stdout.strip() and "Intel(R) MPI Library" in mpi_msg:
        mpiexec = None
        logging.error("Not using MPIexec since MinGW is not compatible with Intel MPI")

    return mpiexec


def get_gemini_exe(gemexe: Path = None) -> Path:
    """
    find and check that Gemini exectuable can run on this system
    download and build Gemini3D if needed
    """

    if not gemexe:  # allow for default dict empty
        gemexe = Path("gemini.bin")
    gemexe = cmake.build_gemini3d(gemexe)

    # %% ensure gemini.bin is runnable
    ret = subprocess.run([str(gemexe)], stdout=subprocess.DEVNULL, timeout=15)
    if ret.returncode != 0:
        raise RuntimeError(
            f"\n{gemexe} was not runnable on your platform. Try recompiling on this computer type."
            "E.g. different HPC nodes may not have the CPU feature sets."
        )

    return gemexe


def check_outdir(out_dir: Pathlike) -> Path:

    out_dir = Path(out_dir).expanduser().resolve()
    if out_dir.is_file():
        raise NotADirectoryError(f"please specify output DIRECTORY, you specified {out_dir}")
    if not out_dir.is_dir():
        out_dir.mkdir(parents=True, exist_ok=True)

    return out_dir
