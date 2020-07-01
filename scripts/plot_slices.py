#!/usr/bin/env python
"""
NOTE: this is made for very basic plots. The axes quantities are notional and not scaled.
"""

from pathlib import Path
from matplotlib.pyplot import figure, show
import argparse

import gemini3d
import gemini3d.vis as vis

try:
    import seaborn as sns
    sns.set_context("talk")
except ImportError:
    pass


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("flag", help="output flag", type=int)
    p.add_argument("fn", help=".dat filename to load directly")
    P = p.parse_args()

    cfg = {"flagoutput": P.flag}

    dat_file = Path(P.fn).expanduser()
    grid_file = dat_file.parent / ("inputs/simgrid" + dat_file.suffix)

    dat = gemini3d.readdata(dat_file, cfg=cfg)
    grid = gemini3d.readgrid(grid_file)

    x1 = grid["x1"]
    x2 = grid["x2"]
    x3 = grid["x3"]

    fg = figure(figsize=(15, 5), tight_layout=True)  # , constrained_layout=True)

    for p in ("ne", "v1", "Ti", "Te", "J1", "J2", "J3", "v2", "v3"):
        if p in dat:
            # %% left panel
            ax = fg.add_subplot(1, 3, 1)
            ix3 = x3.size // 2 - 1  # arbitrary slice
            vis.plot12(
                x2, x1, dat[p][1][:, :, ix3], name=p, cmap=None, vmin=None, vmax=None, fg=fg, ax=ax
            )
            # %% middle panel
            ax = fg.add_subplot(1, 3, 2)
            ix1 = x1.size // 2 - 1  # arbitrary slice
            vis.plot23(
                x3, x2, dat[p][1][ix1, :, :], name=p, cmap=None, vmin=None, vmax=None, fg=fg, ax=ax
            )
            # %% right panel
            ax = fg.add_subplot(1, 3, 3)
            ix2 = x2.size // 2 - 1  # arbitrary slice
            hi = vis.plot13(
                x3,
                x1,
                dat[p][1][:, ix2, :],
                name=p,
                cmap=None,
                vmin=None,
                vmax=None,
                fg=fg,
                ax=ax,
            )

            show()
