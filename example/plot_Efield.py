#!/usr/bin/env python3
"""
plot electric field input to simulation "Efield_inputs" for a single file
"""

from pathlib import Path
import argparse

import xarray
import matplotlib as mpl

import gemini3d.read as read


def plotVmaxx1it(ax: mpl.axes.Axes, V: xarray.DataArray) -> None:
    ax.set_title("Vmaxx1it: Potential")
    if V.ndim == 1:
        ax.plot(dat["mlat"], V)
        ax.set_ylabel("Potential [V]")
        ax.set_xlabel("mag. latitude [deg.]")
    elif V.ndim == 2:
        hi = ax.pcolormesh(dat["mlon"], dat["mlat"], V, cmap="bwr", shading="nearest")
        ax.set_xlabel("mag. longitude [deg.]")
        ax.set_ylabel("mag. latitude [deg.]")
        ax.figure.colorbar(hi, ax=ax).set_label("potential [V]")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("fn", help=".dat or .h5 filename to load directly")
    P = p.parse_args()

    fn = Path(P.fn).expanduser()

    dat = read.Efield(fn)

    fg = mpl.figure.Figure()
    ax = fg.gca()

    plotVmaxx1it(ax, dat["Vmaxx1it"][1].squeeze())

    plt_fn = fn.parent / "plots/Vmaxx1it.png"
    fg.savefig(plt_fn)
