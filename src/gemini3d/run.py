import logging
import argparse
from pathlib import Path
import time

from .job import runner


def cli():
    p = argparse.ArgumentParser()
    p.add_argument("config_file", help="path to config*.nml file")
    p.add_argument("out_dir", help="simulation output directory")
    p.add_argument("-dryrun", help="quick check if sim is ok", action="store_true")
    p.add_argument("-mpiexec", help="path to desired mpiexec executable")
    p.add_argument("-gemexe", help="path to desired gemini.bin")
    p.add_argument("-n", "--cpu", help="number of CPU cores", type=int, default=0)
    p.add_argument("-f", "--force", help="force regeneration of simulation", action="store_true")
    p.add_argument(
        "-out_format", help="override Fortran output file format", choices=["h5", "nc", "raw"]
    )
    p.add_argument("-v", "--verbose", action="store_true")
    P = p.parse_args()

    level = logging.INFO if P.verbose else None
    logging.basicConfig(format="%(message)s", level=level)

    params = {
        "config_file": Path(P.config_file).expanduser(),
        "out_dir": Path(P.out_dir).expanduser().resolve(),
        "mpiexec": P.mpiexec,
        "gemexe": P.gemexe,
        "force": P.force,
        "out_format": P.out_format,
        "cpu_count": P.cpu,
        "dryrun": P.dryrun,
    }

    tic = time.monotonic()
    runner(params)
    print(f"job.py ran in {time.monotonic() - tic:.3f} seconds.")


if __name__ == "__main__":
    cli()
