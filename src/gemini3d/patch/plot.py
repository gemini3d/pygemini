from pathlib import Path

from matplotlib.pyplot import Figure, Axes, draw, pause
import numpy as np
import h5py

from . import filenames2times, time2filename, patch_grid


def grid_step(x2new, x3new, i: int, N: int, ax: Axes) -> None:

    M = 0.9  # arbitrary max grayscale value [0, 1]

    ax.plot(*np.meshgrid(x2new, x3new), linestyle="", marker=".", color=str(i / N * M))
    ax.set_ylabel("x2")
    ax.set_xlabel("x3")
    draw()
    pause(0.05)


def patch(indir: Path, var: str):

    times = filenames2times(indir)

    fg = Figure()
    ax = fg.gca()

    for t in times:
        file = time2filename(indir, t)
        with h5py.File(file, "r") as fh:
            v = fh[var][:]

        x2, x3 = patch_grid(file)

        ax.pcolormesh(*np.meshgrid(x2, x3), v)
