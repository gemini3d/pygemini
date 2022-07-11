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


def cli():
    p = argparse.ArgumentParser()
    p.add_argument("testname", help="name of test")
    p.add_argument("exe", help="Gemini3D executable binary")
    p.add_argument("outdir", help="output directory")
    p.add_argument("refdir", help="reference directory")
    p.add_argument("-dryrun", help="only run first time step", action="store_true")
    P = p.parse_args()

    runner(
        P.testname,
        P.exe,
        P.outdir,
        P.refdir,
        dryrun=P.dryrun,
    )


def runner(
    test_name: str,
    exe: str,
    outdir: Path,
    refdir: Path,
    *,
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

    # have to get exe as absolute path
    cmd = [str(Path(exe).resolve()), str(outdir)]
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
