#!/usr/bin/env python3
"""
Plot all subdirectories one level below the specified directory.
This is useful if making a change in PyGemini or Gemini3D, to plot many simulations
to visually check for something unexpected.
"""

from pathlib import Path
import argparse

import gemini3d.plot as plot


p = argparse.ArgumentParser()
p.add_argument("top_dir", help="Top level directory to look one level below")
P = p.parse_args()

top_dir = Path(P.top_dir).expanduser()

if not top_dir.is_dir():
    raise NotADirectoryError(top_dir)

dirs = (d for d in top_dir.iterdir() if d.is_dir())
for d in dirs:
    print(d)
    plot.plot_all(d)
