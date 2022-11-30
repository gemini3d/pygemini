#!/usr/bin/env python3
"""
plot ForestGemini per-worker patch HDF5 files (e.g. 1000-10000+ files per time step)
"""

from pathlib import Path
import argparse
import logging

import gemini3d.patch.plot as plot

plot_vars = {
    "ne",
    "Te",
    "Ti",
    "J1",
    "J2",
    "J3",
    "v1",
    "v2",
    "v3",
}


p = argparse.ArgumentParser()
p.add_argument("indir", help="ForestGemini patch .h5 data directory")
p.add_argument("-var", help="variable to plot", nargs="+", default=plot_vars)
p.add_argument("-v", "--verbose", action="store_true")
args = p.parse_args()

if args.verbose:
    logging.basicConfig(level=logging.DEBUG)

indir = Path(args.indir).expanduser()
if not indir.is_dir():
    raise NotADirectoryError(indir)

plot.patch(indir, var=set(args.var))
