#!/usr/bin/env python3
"""
convert Gemini3D old raw binary data to HDF5 .h5

For clarity, the user must provide a config.nml for the original raw data.
"""

from pathlib import Path
import argparse
import typing

import gemini3d.raw.read as raw_read
import gemini3d.read as read
import gemini3d.write as write

LSP = 7
CLVL = 6


p = argparse.ArgumentParser()
p.add_argument("indir", help="Gemini .dat file directory")
p.add_argument("outdir", help="directory to write HDF5 files")
P = p.parse_args()

indir = Path(P.indir).expanduser()
outdir = Path(P.outdir).expanduser()

infiles: typing.Iterable[Path]
if indir.is_file():
    infiles = [indir]
    indir = indir.parent
elif indir.is_dir():
    infiles = indir.glob("*.dat")
else:
    raise FileNotFoundError(indir)

cfg = read.config(indir)
if "flagoutput" not in cfg:
    raise LookupError(f"need to specify flagoutput in {indir}/config.nml")

try:
    xg = raw_read.grid(indir)
except FileNotFoundError:
    xg = {}

i = 0
for infile in infiles:
    if infile.stem in {"simsize", "simgrid", "initial_conditions"}:
        continue

    outfile = outdir / (f"{infile.stem}.h5")
    print(infile, "=>", outfile)
    i += 1

    dat = raw_read.data(infile, cfg=cfg, xg=xg)

    write.state(outfile, dat)

if i == 0:
    raise FileNotFoundError(f"no .dat files found in {indir}")

print(f"DONE: converted {i} files in {indir} to {outdir}")
