#!/usr/bin/env python3
"""
plot electric field input to simulation "Efield_inputs" for a single file
"""

from pathlib import Path
import argparse
from matplotlib.figure import Figure
import gemini3d.read
import numpy as np


def plotVmaxx1it(V: np.ndarray) -> Figure:

    V = V.squeeze()
    fg = Figure()
    ax = fg.gca()
    ax.set_title("Vmaxx1it: Potential")
    if V.ndim == 1:
        ax.plot(dat["mlat"], V)
        ax.set_ylabel("Potential [V]")
        ax.set_xlabel("mag. latitude [deg.]")
    elif V.ndim == 2:
        hi = ax.pcolormesh(dat["mlon"], dat["mlat"], V, cmap="bwr")
        ax.set_xlabel("mag. longitude [deg.]")
        ax.set_ylabel("mag. latitude [deg.]")
        fg.colorbar(hi, ax=ax).set_label("potential [V]")

    return fg


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("fn", help=".dat or .h5 filename to load directly")
    P = p.parse_args()

    fn = Path(P.fn).expanduser()

    dat = gemini3d.read.Efield(fn)

    fg = plotVmaxx1it(dat["Vmaxx1it"][1])

    plt_fn = fn.parent / "plots/Vmaxx1it.png"
    fg.savefig(plt_fn)
