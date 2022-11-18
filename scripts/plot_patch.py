#!/usr/bin/env python3
"""
plot ForestGemini per-worker patch HDF5 files (e.g. 1000-10000+ files per time step)
"""

from pathlib import Path
import argparse
import logging

import gemini3d.patch.plot as plot


p = argparse.ArgumentParser()
p.add_argument("indir", help="ForestGemini patch .h5 data directory")
p.add_argument("var", help="variable to plot")
p.add_argument("-v", "--verbose", action="store_true")
P = p.parse_args()

if P.verbose:
    logging.basicConfig(level=logging.DEBUG)

indir = Path(P.indir).expanduser()
if not indir.is_dir():
    raise NotADirectoryError(indir)

plot.patch(indir, P.var)
