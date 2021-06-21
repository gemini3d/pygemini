#!/usr/bin/env python3
"""
convert Gemini grid to HDF5 .h5
"""

from pathlib import Path
import argparse

import gemini3d.read as read
import gemini3d.write as write


def cli():
    p = argparse.ArgumentParser()
    p.add_argument("format", help="file format", choices=["h5", "nc"])
    p.add_argument(
        "indir", help="Gemini3d path to simgrid.dat or path containing inputs/simgrid.dat"
    )
    p.add_argument("-i", "--intype", help="type of input file [.dat]", default=".dat")
    p.add_argument("-o", "--outdir", help="directory to write HDF5 files")
    P = p.parse_args()

    indir = Path(P.indir).expanduser()
    if P.outdir:
        outdir = Path(P.outdir).expanduser()
    elif indir.is_file():
        outdir = indir.parent
    elif indir.is_dir():
        outdir = indir
    else:
        raise FileNotFoundError(indir)

    outfile = outdir / f"simgrid.{P.format}"
    print(indir, "=>", outfile)

    xg = read.grid(indir, file_format=P.intype)

    cfg = {
        "indat_size": xg["filename"].with_name(f"simsize.{P.format}"),
        "indat_grid": xg["filename"].with_suffix(f".{P.format}"),
    }

    write.grid(cfg, file_format=P.format, xg=xg)


if __name__ == "__main__":
    cli()
