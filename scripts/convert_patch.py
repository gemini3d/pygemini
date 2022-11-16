#!/usr/bin/env python3
"""
convert ForestGemini per-worker patch HDF5 files (e.g. 1000-10000+ files per time step)
into one HDF5 file per time step
"""

from pathlib import Path
import argparse
import logging

from gemini3d.patch.convert import convert

data_vars = {
    "J1all",
    "J2all",
    "J3all",
    "Phiall",
    "Tsall",
    "nsall",
    "vs1all",
    "v2avgall",
    "v3avgall",
}


p = argparse.ArgumentParser()
p.add_argument("indir", help="ForestGemini patch .h5 data directory")
p.add_argument("-o", "--outdir", help="directory to write combined HDF5 files")
p.add_argument("-plotgrid", help="plot grid per patch", action="store_true")
p.add_argument("-v", "--verbose", action="store_true")
P = p.parse_args()

if P.verbose:
    logging.basicConfig(level=logging.DEBUG)

indir = Path(P.indir).expanduser()
if not indir.is_dir():
    raise NotADirectoryError(indir)

if P.outdir:
    outdir = Path(P.outdir).expanduser()
else:
    outdir = indir

convert(indir, outdir, data_vars, P.plotgrid)
