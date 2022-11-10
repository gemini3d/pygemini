#!/usr/bin/env python3
"""
convert ForestGemini per-worker patch HDF5 files (e.g. 1000-10000+ files per time step)
into one HDF5 file per time step
"""

from __future__ import annotations
from pathlib import Path
import argparse
import typing
from datetime import datetime

import numpy as np
import h5py
from matplotlib.pyplot import figure, draw, pause

import gemini3d.utils as utils

data_vars = {"J1all", "J2all", "J3all", "Phiall", "Tsall", "nsall", "vsall", "v2avgall", "v3avgall"}


def time2stem(time: datetime) -> str:
    UTsec = time.hour * 3600 + time.minute * 60 + time.second
    return f"{time:%Y%m%d}_{UTsec}"


def get_xlims(path: Path, time: datetime) -> tuple[typing.Any, typing.Any, typing.Any]:
    """
    build up each axis by scanning files

    Remember HDF5 datasets are always in C-order, even when written from Fortran program.
    """

    x1: typing.Any = np.ndarray(0)
    x2: typing.Any = np.ndarray(0)
    x3: typing.Any = np.ndarray(0)

    pat = time2stem(time) + ".*.h5"

    fg = figure()
    ax = fg.gca()

    files = sorted(path.glob(pat))

    N = len(files)  # number of shades to plot
    M = 0.9  # arbitrary max grayscale value [0, 1]

    for i, f in enumerate(files):
        with h5py.File(f, "r") as fh:
            if fh["x1lims"][0] not in x1 and fh["x1lims"][1] not in x1:
                x1new = np.linspace(fh["x1lims"][0], fh["x1lims"][1], fh["nsall"].shape[-1])
                x1 = np.append(x1, x1new)

            x2new = np.linspace(fh["x2lims"][0], fh["x2lims"][1], fh["nsall"].shape[-2])
            x2 = np.append(x2, x2new)

            x3new = np.linspace(fh["x3lims"][0], fh["x3lims"][1], fh["nsall"].shape[-3])
            x3 = np.append(x3, x3new)

            ax.plot(*np.meshgrid(x2new, x3new), linestyle="", marker=".", color=str(i / N * M))
            ax.set_ylabel("x2")
            ax.set_xlabel("x3")
            draw()
            pause(0.05)

    x1.sort()
    x2.sort()
    x3.sort()

    return x1, x2, x3


def combine_files(indir: Path, outdir: Path, time: datetime, var: set[str], x1, x2, x3):

    stem = time2stem(time)
    pat = stem + ".*.h5"
    outfn = outdir / (stem + ".h5")

    with h5py.File(outfn, "w") as oh:
        oh.create_dataset(name="x1", dtype=np.float32, data=x1)
        oh.create_dataset(name="x2", dtype=np.float32, data=x2)
        oh.create_dataset(name="x3", dtype=np.float32, data=x3)

        lx1 = x1.size
        lx2 = x2.size
        lx3 = x3.size

        print("write", outfn, "  lx1, lx2, lx3 =", lx1, lx2, lx3)

        for f in indir.glob(pat):
            with h5py.File(f, "r") as ih:
                ix1 = (
                    (ih["x1lims"][0] == x1).nonzero()[0].item(),
                    (ih["x1lims"][1] == x1).nonzero()[0].item(),
                )
                ix2 = (
                    (ih["x2lims"][0] == x2).nonzero()[0].item(),
                    (ih["x2lims"][1] == x2).nonzero()[0].item(),
                )
                ix3 = (
                    (ih["x3lims"][0] == x3).nonzero()[0].item(),
                    (ih["x3lims"][1] == x3).nonzero()[0].item(),
                )
                for v in var:
                    if v in ih:
                        if v not in oh:
                            if ih[v].ndim == 4:
                                shape: tuple[int, ...] = (ih[v].shape[0], lx3, lx2, lx1)
                            elif ih[v].ndim == 3:
                                shape = (lx3, lx2, lx1)
                            oh.create_dataset(name=v, shape=shape, dtype=ih[v].dtype)

                        if ih[v].ndim == 4:
                            oh[v][
                                :, ix3[0] : ix3[1] + 1, ix2[0] : ix2[1] + 1, ix1[0] : ix1[1] + 1
                            ] = ih[v]
                        elif ih[v].ndim == 3:
                            oh[v][
                                ix3[0] : ix3[1] + 1, ix2[0] : ix2[1] + 1, ix1[0] : ix1[1] + 1
                            ] = ih[v]


p = argparse.ArgumentParser()
p.add_argument("indir", help="ForestGemini patch .h5 data directory")
p.add_argument("-o", "--outdir", help="directory to write combined HDF5 files")
P = p.parse_args()

indir = Path(P.indir).expanduser()
if not indir.is_dir():
    raise NotADirectoryError(indir)

if P.outdir:
    outdir = Path(P.outdir).expanduser()
else:
    outdir = indir

# get times from filenames
times = []
names = set([f.stem[:21] for f in indir.glob("*_*.*_*.h5")])
for n in names:
    times.append(utils.filename2datetime(n))

# Need to get extents by scanning all files.
# FIXME: Does ForestClaw have a way to write this without this inefficient scan?

x1, x2, x3 = get_xlims(indir, times[0])

for t in times:
    combine_files(indir, outdir, t, data_vars, x1, x2, x3)
