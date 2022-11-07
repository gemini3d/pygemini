#!/usr/bin/env python3
"""
convert Gemini3D old raw binary data to HDF5 .h5

For clarity, the user must provide a config.nml for the original raw data.
"""

from pathlib import Path
import argparse

import gemini3d.raw.read as raw_read
import gemini3d.read as read
import gemini3d.write as write

LSP = 7
CLVL = 6


p = argparse.ArgumentParser()
p.add_argument("indir", help="Gemini .dat file directory")
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

if indir.is_file():
    infiles = [indir]
    indir = indir.parent
elif indir.is_dir():
    infiles = sorted(indir.glob("*.dat"))
else:
    raise FileNotFoundError(indir)

if not infiles:
    raise FileNotFoundError(f"no *.dat files to convert in {indir}")

cfg = read.config(indir)
if "flagoutput" not in cfg:
    raise LookupError(f"need to specify flagoutput in {indir}/config.nml")

try:
    xg = raw_read.grid(indir)
except FileNotFoundError:
    xg = None

for infile in infiles:
    if infile.name in {"simsize", "simgrid", "initial_conditions"}:
        continue

    outfile = outdir / (f"{infile.stem}.h5")
    print(infile, "=>", outfile)

    dat = raw_read.data(infile, cfg=cfg, xg=xg)

    write.state(outfile, dat, xg=xg)
