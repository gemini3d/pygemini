#!/usr/bin/env python3
"""
run test
"""

import argparse
import sys
import subprocess
from pathlib import Path
import shutil

import gemini3d.web
import gemini3d.read
from . import mpi


def cli():
    p = argparse.ArgumentParser()
    p.add_argument("testname", help="name of test")
    p.add_argument("-mpiexec", help="mpiexec path")
    p.add_argument("exe", help="Gemini.bin executable binary")
    p.add_argument("outdir", help="output directory")
    p.add_argument("refdir", help="reference directory")
    p.add_argument("-np", help="force number of MPI images", type=int)
    p.add_argument(
        "-out_format",
        help="override config.nml output file format",
        choices=["h5", "nc", "raw"],
    )
    p.add_argument("-dryrun", help="only run first time step", action="store_true")
    P = p.parse_args()

    runner(
        P.testname,
        P.mpiexec,
        P.exe,
        P.outdir,
        P.refdir,
        mpi_count=P.np,
        out_format=P.out_format,
        dryrun=P.dryrun,
    )


def runner(
    test_name: str,
    mpiexec: str,
    exe: str,
    outdir: Path,
    refdir: Path,
    *,
    mpi_count: int = None,
    out_format: str = None,
    dryrun: bool = False,
):
    """configure and run a test
    This is usually called from CMake Ctest
    """

    outdir = Path(outdir).expanduser().resolve()
    refdir = Path(refdir).expanduser().resolve()

    ref = gemini3d.web.download_and_extract(test_name, refdir)

    # prepare simulation output directory
    input_dir = outdir / "inputs"
    input_dir.mkdir(parents=True, exist_ok=True)

    # a normal, non-test simulation already has all these files in the
    # output directory. here, we use a reference simulation input data.
    # the input data generation is tested elsewhere in PyGemini.
    # Here, we want to test that we can create
    # data match to reference outputs from reference inputs.

    if not (input_dir / "config.nml").is_file():
        shutil.copy2(ref / "inputs/config.nml", input_dir)

    cfg = gemini3d.read.config(ref)

    # delete previous test run data to avoid restarting milestone and failing test
    if (outdir / "output.nml").is_file():
        stem = cfg["time"][0].strftime("%Y%m%d")
        for f in outdir.glob(f"{stem}*.h5"):
            f.unlink()

    # copy remaining input files needed
    if not (input_dir / cfg["indat_size"]).is_file():
        shutil.copy2(ref / cfg["indat_size"], input_dir)
        shutil.copy2(ref / cfg["indat_grid"], input_dir)
        shutil.copy2(ref / cfg["indat_file"], input_dir)
    if "precdir" in cfg and not (outdir / cfg["precdir"]).is_dir():
        shutil.copytree(ref / cfg["precdir"], outdir / cfg["precdir"])
    if "E0dir" in cfg and not (outdir / cfg["E0dir"]).is_dir():
        shutil.copytree(ref / cfg["E0dir"], outdir / cfg["E0dir"])
    if "neutral_perturb" in cfg and not (outdir / cfg["sourcedir"]).is_dir():
        shutil.copytree(ref / cfg["sourcedir"], outdir / cfg["sourcedir"])

    if not mpi_count:
        mpi_count = mpi.count(ref / cfg["indat_size"], 0)

    # have to get exe as absolute path
    exe_abs = Path(exe).resolve()
    if mpiexec:
        cmd = [mpiexec, "-np", str(mpi_count), str(exe_abs), str(outdir)]
    else:
        cmd = [str(exe_abs), str(outdir)]
    if out_format:
        cmd += ["-out_format", out_format]
    if dryrun:
        cmd.append("-dryrun")

    print(" ".join(cmd))

    ret = subprocess.run(cmd)
    if ret.returncode == 0:
        print("OK:", test_name)
    else:
        print("FAIL:", test_name, file=sys.stderr)

    raise SystemExit(ret.returncode)


if __name__ == "__main__":
    cli()
