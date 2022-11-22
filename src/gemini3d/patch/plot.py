from pathlib import Path

from matplotlib.pyplot import figure, Axes, draw, pause
import numpy as np
import h5py

from . import filenames2times, patch_grid
from .. import utils


def grid_step(x2new, x3new, i: int, N: int, ax: Axes) -> None:

    M = 0.9  # arbitrary max grayscale value [0, 1]

    ax.plot(*np.meshgrid(x2new, x3new), linestyle="", marker=".", color=str(i / N * M))
    ax.set_ylabel("x2")
    ax.set_xlabel("x3")
    draw()
    pause(0.05)


def patch(indir: Path, var: str):

    # a priori parameter based on Gemini3D data structure in the HDF5 files and memory
    LSP = 7
    p4 = (0, 3, 2, 1)
    # p3 = (2, 1, 0)

    # arbitrary user parameters
    clim_ne = (1e8, 1e11)

    ix1 = 10  # TODO arbitrary, needs to be physically referenced like in plot12, plot23, etc.

    times = filenames2times(indir)

    for t in times:
        first = True

        fg = figure()
        ax = fg.gca()
        ax.set_title(f"{var}: {t}")

        pat = utils.datetime2stem(t) + "_*.h5"
        for file in indir.glob(pat):
            wid = file.stem.split("_")[-1]
            with h5py.File(file, "r") as fh:
                if var == "ne":
                    v = fh["/nsall"][:].transpose(p4)[LSP - 1, :, :, :]
                    clim = clim_ne

            x2, x3 = patch_grid(file)

            h = ax.pcolormesh(x2, x3, v[ix1, :, :].transpose(), vmin=clim[0], vmax=clim[1])
            ax.text(
                x2[x2.size // 2],
                x3[x3.size // 2],
                wid,
                verticalalignment="center",
                horizontalalignment="center",
            )
            if first:
                fg.colorbar(h, ax=ax)
                first = False

            draw()
            pause(0.05)
