import typing as T
import os
import logging
import subprocess
import shutil
import sys
from pathlib import Path

from .mpi import get_mpi_count
from .config import read_config, get_config_filename
from .hpc import hpc_batch_detect, hpc_batch_create
from .model_setup import model_setup

Pathlike = T.Union[str, Path]


def runner(pr: T.Dict[str, T.Any]) -> None:

    out_dir = check_outdir(pr["out_dir"])

    config_file = get_config_filename(pr["config_file"])
    # load configuration to know what directories to check
    p = read_config(config_file)

    for k in ("indat_size", "indat_grid", "indat_file"):
        f = out_dir / p[k].expanduser()
        if pr["force"] or not f.is_file():
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
    mpiexec = check_mpiexec(pr["mpiexec"])

    gemexe = check_gemini_exe(pr["gemexe"])
    logging.info(f"gemini executable: {gemexe}")

    if mpiexec:
        logging.info(f"mpiexec: {mpiexec}")
        Nmpi = get_mpi_count(out_dir / p["indat_size"], pr["cpu_count"])
        mpi_cmd = [mpiexec, "-n", str(Nmpi)]
    else:
        mpi_cmd = []

    cmd = mpi_cmd + [str(gemexe), str(out_dir)]

    if pr["out_format"]:
        cmd += ["-out_format", pr["out_format"]]

    # %% attempt dry run, but don't fail in case intended for HPC
    logging.info("Gemini dry run of first time step")
    logging.info(" ".join(cmd))
    proc = subprocess.run(cmd + ["-dryrun"])

    if proc.returncode == 0:
        logging.info("OK: Gemini dry run")
    else:
        print(proc.stdout, file=sys.stderr)
        raise RuntimeError("Gemini dry run failed.")

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


def check_mpiexec(mpiexec: Pathlike) -> str:
    """check if specified mpiexec exists on this system
    If not, fall back to not using MPI (slow, but still works)."""

    if not mpiexec:
        mpiexec = "mpiexec"
    mpi_root = os.environ.get("MPI_ROOT")
    if mpi_root:
        mpi_root += "/bin"

    mpiexec = shutil.which(mpiexec, path=mpi_root)

    return mpiexec


def check_gemini_exe(gemexe: Pathlike) -> str:
    """
    check that Gemini exectuable can run on this system

    If not given a specific full path to gemini.bin, looks for gemini.bin under:

        build
        build / Release
        build / Debug
    """

    if gemexe:
        gemexe = Path(gemexe).expanduser()
        if not gemexe.is_file():
            raise EnvironmentError(f"Cannot find gemini.bin in {gemexe}")
    elif os.environ.get("GEMINI_ROOT"):
        build_dir = Path(os.environ["GEMINI_ROOT"]).expanduser().resolve() / "build"
        if not build_dir.is_dir():
            raise EnvironmentError(f"GEMINI build directory missing: {build_dir}")

        for d in (build_dir, build_dir / "Release", build_dir / "Debug"):
            gemexe = shutil.which("gemini.bin", path=str(d))
            if gemexe:
                break
        if not gemexe:
            raise EnvironmentError(f"\nCannot find gemini.bin under {build_dir}")
    else:
        raise EnvironmentError("Please specify path to gemini.bin")

    gemexe = str(Path(gemexe).resolve())

    ret = subprocess.run(gemexe, stdout=subprocess.DEVNULL)
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
