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

    LSP = 7
    p4 = (0, 3, 2, 1)
    # p3 = (2, 1, 0)

    ix1 = 10  # TODO arbitrary, needs to be physically references like in plot12, plot23, etc.

    times = filenames2times(indir)

    for t in times:
        fg = figure()
        ax = fg.gca()
        ax.set_title(str(t))
        pat = utils.datetime2stem(t) + "_*.h5"
        for file in indir.glob(pat):
            with h5py.File(file, "r") as fh:
                if var == "ne":
                    v = fh["/nsall"][:].transpose(p4)[LSP - 1, :, :, :]

            x2, x3 = patch_grid(file)

            ax.pcolormesh(x2, x3, v[ix1, :, :].transpose())
            draw()
            pause(0.05)
