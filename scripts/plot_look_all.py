#!/usr/bin/env python3
"""
Look at specific pairs of plots between a reference directory and a "new" directory.
All subdirectories one level below the specified directory.
This is useful if making a change in PyGemini or Gemini3D, to plot many simulations
to visually check for something unexpected.
"""

from pathlib import Path
import argparse
import typing as T
from matplotlib.pyplot import figure, draw, pause
from matplotlib.image import imread
from datetime import datetime


var = {"ne", "Te", "Ti", "J1", "J2", "J3", "v1", "v2", "v3"}
fs = 8

p = argparse.ArgumentParser(
    description="Look at specific pairs of plots between a reference directory and a new directory."
)
p.add_argument("ref_dir", help="Reference: top level directory to look one level below")
p.add_argument("new_dir", help="New: top level directory to look one level below")
P = p.parse_args()

ref_dir = Path(P.ref_dir).expanduser().resolve()
new_dir = Path(P.new_dir).expanduser().resolve()

if not ref_dir.is_dir():
    raise NotADirectoryError(ref_dir)
if not new_dir.is_dir():
    raise NotADirectoryError(new_dir)
if ref_dir.samefile(new_dir):
    raise ValueError("ref_dir and new_dir must be different")


new_sims = (d for d in new_dir.iterdir() if d.is_dir())
ref_sims = (d for d in ref_dir.iterdir() if d.is_dir())

fg = figure()
axs: T.Any = fg.subplots(2, 1, sharex=True, sharey=True)

for new_sim in new_sims:
    new_name = new_sim.name
    if not (ref_dir / new_name).is_dir():
        raise NotADirectoryError(ref_dir / new_name)

    for v in var:
        plot_name = sorted((new_sim / "plots").glob(f"{v}-*.png"))[-1].name
        ref_file = ref_dir / new_name / "plots" / plot_name
        new_file = new_sim / "plots" / plot_name

        if not ref_file.is_file():
            raise FileNotFoundError(ref_file)

        axs[0].imshow(imread(ref_file))
        axs[0].set_title(ref_file, fontsize=fs)
        axs[0].set_xlabel(datetime.utcfromtimestamp(ref_file.stat().st_ctime))

        axs[1].imshow(imread(new_file))
        axs[1].set_title(new_file, fontsize=fs)
        axs[1].set_xlabel(datetime.utcfromtimestamp(new_file.stat().st_ctime))
        draw()
        pause(0.1)
