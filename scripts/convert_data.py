#!/usr/bin/env python3
"""
convert Gemini data to HDF5 .h5
"""

from pathlib import Path
import argparse

import gemini3d.read as read
import gemini3d.write as write

LSP = 7
CLVL = 6


def cli():
    p = argparse.ArgumentParser()
    p.add_argument("format", help="file format", choices=["h5", "nc"])
    p.add_argument("indir", help="Gemini .dat file directory")
    p.add_argument("-o", "--outdir", help="directory to write HDF5 files")
    p.add_argument(
        "-f",
        "--flagoutput",
        help="manually specify flagoutput, for if config.nml is missing",
        type=int,
    )
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

    if indir.is_file():
        infiles = [indir]
        indir = indir.parent
    elif indir.is_dir():
        infiles = sorted(indir.glob("*.dat"))
    else:
        raise FileNotFoundError(indir)

    if not infiles:
        raise FileNotFoundError(f"no files to convert in {indir}")

    lxs = read.simsize(indir)

    cfg = {"file_format": P.format}
    if P.flagoutput is not None:
        cfg["flagoutput"] = P.flagoutput

    xg = None
    if P.format == "nc":
        xg = read.grid(indir)

    for infile in infiles:
        outfile = outdir / (f"{infile.stem}.{cfg['file_format']}")
        print(infile, "=>", outfile)

        dat = read.data(infile, cfg=cfg)
        if "lxs" not in dat:
            dat["lxs"] = lxs

        write.data(outfile, dat, P.format, xg)


if __name__ == "__main__":
    cli()
