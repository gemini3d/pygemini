import typing as T
import os
import logging
import subprocess
import shutil
from pathlib import Path
import importlib.resources

from .build import cmake_build
from .mpi import get_mpi_count
from .config import read_config, get_config_filename
from .hpc import hpc_batch_detect, hpc_batch_create
from .model_setup import model_setup
from .fileio import log_meta_nml
from .utils import git_meta

Pathlike = T.Union[str, Path]


def runner(pr: T.Dict[str, T.Any]) -> None:

    out_dir = check_outdir(pr["out_dir"])

    config_file = get_config_filename(pr["config_file"])
    # load configuration to know what directories to check
    p = read_config(config_file)
    if not p:
        raise FileNotFoundError(f"{config_file} does not appear to contain config.nml")

    for k in ("indat_size", "indat_grid", "indat_file"):
        f = out_dir / p[k].expanduser()
        if pr.get("force") or not f.is_file():
            model_setup(p["nml"], out_dir)

    if "E0dir" in p:
        E0dir = out_dir / p["E0dir"]
        if not E0dir.is_dir():
            model_setup(p["nml"], out_dir)

    if "precdir" in p:
        precdir = out_dir / p["precdir"]
        if not precdir.is_dir():
            model_setup(p["nml"], out_dir)

    # build checks
    gemexe = get_gemini_exe(pr.get("gemexe"))
    logging.info(f"gemini executable: {gemexe}")

    mpiexec = check_mpiexec(pr.get("mpiexec"), gemexe)
    if mpiexec:
        logging.info(f"mpiexec: {mpiexec}")
        Nmpi = get_mpi_count(out_dir / p["indat_size"], pr.get("cpu_count"))
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

    log_meta_nml(out_dir / "setup_meta.nml", git_meta(gemexe.parent), "setup_gemini")

    batcher = hpc_batch_detect()
    if batcher:
        job_file = hpc_batch_create(batcher, out_dir, cmd)  # noqa: F841
        # hpc_submit_job(job_file)
        print("Please examine batch file", job_file, "and when ready submit the job as usual.")
    else:
        print("\nBEGIN Gemini run with command:")
        print(" ".join(cmd), "\n")
        ret = subprocess.run(cmd).returncode
        if ret != 0:
            raise RuntimeError("Gemini run failed")


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

    mpiexec = shutil.which(mpiexec, path=mpi_root)
    if not mpiexec:
        return None

    ret = subprocess.run([mpiexec, "-help"], stdout=subprocess.PIPE, text=True)
    if ret.returncode != 0:
        return None
    # %% check that compiler and MPIexec compatible
    if os.name != "nt":
        return mpiexec

    mpi_msg = ret.stdout.strip()
    ret = subprocess.run([str(gemexe), "-compiler"], stdout=subprocess.PIPE, text=True)
    if ret.returncode != 0:
        raise EnvironmentError(f"{gemexe} not executable")

    if "GNU" in ret.stdout.strip() and "Intel(R) MPI Library" in mpi_msg:
        mpiexec = None
        logging.error("Not using MPIexec since MinGW is not compatible with Intel MPI")

    return mpiexec


def get_gemini_exe(gemexe: Path = None) -> Path:
    """
    find and check that Gemini exectuable can run on this system

    If not given a specific full path to gemini.bin, looks for gemini.bin under:

        build
        build / Release
        build / Debug
    """

    if not gemexe:  # allow for default dict empty
        gemexe = Path("gemini.bin.exe") if os.name == "nt" else Path("gemini.bin")
    gemexe = Path(gemexe).expanduser()  # not .resolve()

    src_dir = None

    if not gemexe.is_file():
        if os.environ.get("GEMINI_ROOT"):
            src_dir = Path(os.environ["GEMINI_ROOT"]).expanduser()
        if not src_dir or not src_dir.is_dir():
            # step 1: clone Gemini3D and do a test build
            with importlib.resources.path(__package__, "setup.cmake") as setup:
                subprocess.check_call(["ctest", "-S", str(setup), "-VV"])
                src_dir = setup.parent / "gemini-fortran"
        assert src_dir.is_dir(), f"could not find Gemini3D source directory {src_dir}"
        build_dir = src_dir / "build"
        gemexe = build_dir / gemexe.name

        if not gemexe.is_file():
            cmake_build(None, src_dir, build_dir, run_test=False, install=False)
            if not gemexe.is_file():
                raise RuntimeError(f"Gemini.bin not found in {build_dir}")
# %% ensure gemini.bin is runnable
    ret = subprocess.run([str(gemexe)], stdout=subprocess.DEVNULL)
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
