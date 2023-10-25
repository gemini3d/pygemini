from __future__ import annotations
import typing
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


def read_var(file: Path, var: set[str]) -> dict[str, typing.Any]:
    # a priori parameter based on Gemini3D data structure in the HDF5 files and memory
    LSP = 7
    p4 = (0, 3, 2, 1)
    p3 = (2, 1, 0)

    dat = {}

    with h5py.File(file, "r") as fh:
        if {"ne", "ns", "v1", "Ti"} & var:
            dat["ns"] = fh["/nsall"][:].transpose(p4)
        if {"v1", "vs1"} & var:
            dat["vs1"] = fh["/vs1all"][:].transpose(p4)
        if {"Te", "Ti", "Ts"} & var:
            dat["Ts"] = fh["/Tsall"][:].transpose(p4)
        for k in {"J1", "J2", "J3"} & var:
            dat[k] = fh[f"/{k}all"][:].transpose(p3)
        for k in {"v2", "v3"} & var:
            dat[k] = fh[f"/{k}avgall"][:].transpose(p3)
        if "Phi" in var:
            dat["Phi"] = fh["/Phiall"][:].transpose()

    if {"ne", "v1", "Ti"} & var:
        dat["ne"] = dat["ns"][LSP - 1, :, :, :]
    if "v1" in var:
        dat["v1"] = (dat["ns"][:6, :, :, :] * dat["vs1"][:6, :, :, :]).sum(axis=0) / dat[
            "ne"
        ]
    if "Ti" in var:
        dat["Ti"] = (dat["ns"][:6, :, :, :] * dat["Ts"][:6, :, :, :]).sum(axis=0) / dat[
            "ne"
        ]
    if "Te" in var:
        dat["Te"] = dat["Ts"][LSP - 1, :, :, :]

    return dat


def patch(indir: Path, var: set[str]) -> None:
    # arbitrary user parameters
    clim = {
        "ne": (1e8, 1e11),
        "Te": (100, 750),
    }
    plot_dir = indir / "plots"

    ix1 = 10  # TODO arbitrary, needs to be physically referenced like in plot12, plot23, etc.

    times = filenames2times(indir)

    fg = figure()

    for t in times:
        for k in var:
            first = True

            fg.clf()
            ax = fg.gca()
            ax.set_title(f"{k}: {t}")

            pat = utils.datetime2stem(t) + "_*.h5"
            for file in indir.glob(pat):
                dat = read_var(file, {k})

                x2, x3 = patch_grid(file)

                h = ax.pcolormesh(
                    x2,
                    x3,
                    dat[k][ix1, :, :].transpose(),
                    vmin=clim[k][0],
                    vmax=clim[k][1],
                    shading="nearest",
                )
                ax.text(
                    x2[x2.size // 2],
                    x3[x3.size // 2],
                    s=file.stem.split("_")[-1],
                    verticalalignment="center",
                    horizontalalignment="center",
                )
                if first:
                    fg.colorbar(h, ax=ax)
                    first = False

                draw()
                pause(0.05)

            pause(0.5)
            plot_fn = plot_dir / f"{k}_{t:%Y-%m-%d_%H%M%S}.png"
            print("save plot", plot_fn)
            fg.savefig(plot_fn)
