from __future__ import annotations
from pathlib import Path
from datetime import datetime
import typing as T

import xarray
from matplotlib.figure import Figure

from .. import read
from .. import find


def glow(
    direc: T.Optional[Path],
    time: datetime,
    saveplot_fmt: str,
    xg: dict[str, T.Any] = None,
    fg: Figure = None,
):
    """plots Gemini-Glow auroral emissions"""

    if direc is None:
        # no glow in this sim
        return

    cfg = read.config(direc.parent)

    if xg is None:
        xg = read.grid(direc.parent)

    # %% get filename
    fn = find.frame(direc, time)

    # %% make plots
    if fg is None:
        fg = Figure(constrained_layout=True)

    B = read.glow(fn)
    t_str = time.strftime("%Y-%m-%dT%H:%M:%S") + " UT"

    if xg["lx"][1] > 1 and xg["lx"][2] > 1:
        # 3D sim
        emission_line(B, t_str, fg)
    elif xg["lx"][1] > 1:
        # 2D east-west
        emissions(B, t_str, fg, "Eastward")
    elif xg["lx"][2] > 1:
        # 2D north-south
        emissions(B, t_str, fg, "Northward")
    else:
        raise ValueError("impossible GLOW configuration")

    if cfg["flagoutput"] != 3:
        save_glowframe(fg, fn, saveplot_fmt)


def emissions(B: xarray.Dataset, time_str: str, fg: Figure, txt: str):

    fg.clf()

    x = B.x2 if B.x2.size > 1 else B.x3

    R = B["rayleighs"].squeeze().transpose()

    ax = fg.gca()
    hi = ax.pcolormesh(range(B.wavelength.size), x / 1e3, R, shading="nearest")

    # set(ax, 'xtick', 1:length(wavelengths), 'xticklabel', wavelengths)

    ax.set_ylabel(f"{txt} Distance (km)")

    ax.set_xlabel(r"emission wavelength ($\AA$)")
    ax.set_title(time_str)
    fg.colorbar(hi, ax=ax).set_label("Intensity (R)")


def emission_line(B: xarray.Dataset, time_str: str, fg: Figure):

    fg.clf()

    # arbitrary pick of which emission lines to plot lat/lon slices
    inds = [1, 3, 4, 8]

    axs = fg.subplots(len(inds), 1, sharex=True, sharey=True)

    for i, j in enumerate(inds):
        ax = axs[i]
        R = B["rayleighs"][j].transpose()
        hi = ax.pcolormesh(B.x2 / 1e3, B.x3 / 1e3, R, shading="nearest")
        hc = fg.colorbar(hi, ax=ax)
        # set(cb,'yticklabel',sprintf('10^{%g}|', get(cb,'ytick')))
        ax.set_title(rf"{B.wavelength[j].item()} $\AA$")

    hc.set_label("Intensity (R)")
    ax.set_xlabel("Eastward Distance (km)")
    ax.set_ylabel("Northward Distance (km)")
    fg.suptitle(f"intensity: {time_str}")


def save_glowframe(fg: Figure, filename: Path, saveplot_fmt: str):
    """CREATES IMAGE FILES FROM PLOTS"""

    outdir = filename.parents[1] / "plots"

    outfile = outdir / f"aurora-{filename.stem}.{saveplot_fmt}"
    print("write:", outfile)
    fg.savefig(outfile)
