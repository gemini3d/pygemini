#!/usr/bin/env python3
"""
convert Gemini .dat to .h5
"""

from pathlib import Path
import argparse

import gemini3d


def cli():
    p = argparse.ArgumentParser()
    p.add_argument("indir", help="Gemini simgrid.dat")
    p.add_argument("-o", "--outdir", help="directory to write HDF5 files")
    P = p.parse_args()

    infile = Path(P.indir).expanduser()
    if P.outdir:
        outdir = Path(P.outdir).expanduser()
    elif infile.is_file():
        outdir = infile.parent
    else:
        raise FileNotFoundError(infile)

    outfile = outdir / (infile.stem + ".h5")
    print(infile, "=>", outfile)

    xg = gemini3d.readgrid(infile)
    cfg = {"indat_size": infile.with_name("simsize.h5"), "indat_grid": infile.with_suffix(".h5")}
    gemini3d.base.write_grid(cfg, xg)


if __name__ == "__main__":
    cli()
