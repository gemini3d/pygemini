#!/usr/bin/env python3
"""
NOTE: this is made for very basic plots. The axes quantities are notional and not scaled.
"""

from pathlib import Path
import argparse

from matplotlib.pyplot import figure, show
from matplotlib.figure import figaspect

import gemini3d.plot.slices as slices
import gemini3d.find as find
import gemini3d.read as read


clim = (None, None)

p = argparse.ArgumentParser()
p.add_argument("fn", help=".dat filename to load directly")
p.add_argument(
    "-s", "--saveplot", help="save plots to data file directory", action="store_true"
)
p.add_argument(
    "-f",
    "--flagoutput",
    help="manually specify flagoutput, for if config.nml is missing",
    type=int,
)
P = p.parse_args()

cfg = {}
if P.flagoutput is not None:
    cfg["flagoutput"] = P.flag

dat_file = Path(P.fn).expanduser()
dat = read.frame(dat_file, cfg=cfg)

try:
    x1 = dat["x1"]
    x2 = dat["x2"]
    x3 = dat["x3"]
except KeyError:
    grid = read.grid(find.grid(dat_file.parent))

    x1 = grid["x1"]
    x2 = grid["x2"]
    x3 = grid["x3"]

Ng = 4  # number of ghost cells
if x1.size == dat["ne"][1].shape[0] + Ng:
    x1 = x1[2:-2]
if x2.size == dat["ne"][1].shape[1] + Ng:
    x2 = x2[2:-2]
if x3.size == dat["ne"][1].shape[2] + Ng:
    x3 = x3[2:-2]

fg = figure(figsize=figaspect(1 / 3), tight_layout=True)  # , constrained_layout=True)
for p in {"ne", "v1", "Ti", "Te", "J1", "J2", "J3", "v2", "v3"}:
    if p in dat:
        fg.clf()
        # %% left panel
        ax = fg.add_subplot(1, 3, 1)
        ix3 = x3.size // 2 - 1  # arbitrary slice
        slices.plot12(
            x2, x1, dat[p][1][:, :, ix3], clim=clim, name=p, ax=ax, ref_alt=100.0
        )
        # %% middle panel
        ax = fg.add_subplot(1, 3, 2)
        ix1 = x1.size // 2 - 1  # arbitrary slice
        slices.plot23(x3, x2, dat[p][1][ix1, :, :], clim=clim, name=p, ax=ax)
        # %% right panel
        ax = fg.add_subplot(1, 3, 3)
        ix2 = x2.size // 2 - 1  # arbitrary slice
        slices.plot13(
            x3,
            x1,
            dat[p][1][:, ix2, :],
            clim=clim,
            name=p,
            ax=ax,
        )

        if P.saveplot:
            pfn = dat_file.parent / f"{p}-{dat_file.stem}.png"
            print("writing", pfn)
            fg.savefig(pfn, bbox_inches="tight")

show()
