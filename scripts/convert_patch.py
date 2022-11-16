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
import logging

import numpy as np
import h5py
from matplotlib.pyplot import figure, draw, pause

import gemini3d.utils as utils

data_vars = {
    "J1all",
    "J2all",
    "J3all",
    "Phiall",
    "Tsall",
    "nsall",
    "vs1all",
    "v2avgall",
    "v3avgall",
}
COMP_LEVEL = 6  # arbitrary GZIP compression level


def time2stem(time: datetime) -> str:
    UTsec = time.hour * 3600 + time.minute * 60 + time.second
    return f"{time:%Y%m%d}_{UTsec}.{time.microsecond:06d}"


def get_xlims(path: Path, time: datetime, plotgrid: bool = False) -> tuple[typing.Any, typing.Any]:
    """
    build up each axis by scanning files

    Remember HDF5 datasets are always in C-order, even when written from Fortran program.
    """

    x2: typing.Any = np.ndarray(0)
    x3: typing.Any = np.ndarray(0)

    pat = time2stem(time) + "_*.h5"

    if plotgrid:
        fg = figure()
        ax = fg.gca()

    files = sorted(path.glob(pat))

    N = len(files)  # number of shades to plot
    M = 0.9  # arbitrary max grayscale value [0, 1]

    for i, f in enumerate(files):
        with h5py.File(f, "r") as fh:
            # TODO: AMR code needs to write actual x1 points, as it's not linearly spaced
            # x1new = np.linspace(
            #     fh["x1lims"][0], fh["x1lims"][1], num=fh["nsall"].shape[-1], endpoint=False
            # )
            # x1 = np.append(x1, x1new)

            x2new = np.linspace(
                fh["x2lims"][0], fh["x2lims"][1], num=fh["nsall"].shape[-2], endpoint=False
            )
            x2 = np.append(x2, x2new)

            x3new = np.linspace(
                fh["x3lims"][0], fh["x3lims"][1], num=fh["nsall"].shape[-3], endpoint=False
            )
            x3 = np.append(x3, x3new)

            if plotgrid:
                ax.plot(*np.meshgrid(x2new, x3new), linestyle="", marker=".", color=str(i / N * M))
                ax.set_ylabel("x2")
                ax.set_xlabel("x3")
                draw()
                pause(0.05)

    x2 = np.unique(x2)
    x3 = np.unique(x3)

    return x2, x3


def combine_files(indir: Path, outdir: Path, time: datetime, var: set[str], x1, x2, x3):

    stem = time2stem(time)
    pat = stem + "_*.h5"
    outfn = outdir / (stem + ".h5")

    lx = (x1.size, x2.size, x3.size)

    print("write", outfn, "lx: ", lx)
    with h5py.File(outfn, "w") as oh:
        for f in indir.glob(pat):
            with h5py.File(f, "r") as ih:
                ix2 = get_indices(ih["x2lims"], x2)
                ix3 = get_indices(ih["x3lims"], x3)
                for v in var:
                    convert_var(oh, ih, v, lx, ix2, ix3)


def get_indices(lims: tuple[float, float], x) -> tuple[int, int]:

    i0 = (lims[0] >= x).nonzero()[0][-1].item()
    i1 = (lims[1] > x).nonzero()[0][-1].item()

    return i0, i1


def convert_var(
    oh: h5py.File,
    ih: h5py.File,
    v: str,
    lx: tuple[int, int, int],
    ix2: tuple[int, int],
    ix3: tuple[int, int],
):

    if v not in ih:
        logging.debug(f"variable {v} not in {ih.filename}")
        return

    if v not in oh:
        if ih[v].ndim == 4:
            shape: tuple[int, ...] = (ih[v].shape[0], *lx[::-1])
        elif ih[v].ndim == 3:
            shape = lx[::-1]
        elif ih[v].ndim == 2:
            shape = (lx[2], lx[1])
        else:
            raise ValueError(f"{v}: ndim {ih[v].ndim} not supported: {oh.filename}")

        oh.create_dataset(
            name=v,
            shape=shape,
            dtype=ih[v].dtype,
            compression="gzip",
            compression_opts=COMP_LEVEL,
            shuffle=True,
            fletcher32=True,
            fillvalue=np.nan,
        )

    logging.debug(
        f"{Path(ih.filename).stem}=>{Path(oh.filename).stem}:{v}:  {ih[v].shape}  ix2: {ix2} ix3: {ix3} {oh[v].shape}"
    )

    if ih[v].ndim == 4:
        oh[v][:, ix3[0] : ix3[1] + 1, ix2[0] : ix2[1] + 1, :] = ih[v]
    elif ih[v].ndim == 3:
        oh[v][ix3[0] : ix3[1] + 1, ix2[0] : ix2[1] + 1, :] = ih[v]
    elif ih[v].ndim == 2:
        oh[v][ix3[0] : ix3[1] + 1, ix2[0] : ix2[1] + 1] = ih[v]


p = argparse.ArgumentParser()
p.add_argument("indir", help="ForestGemini patch .h5 data directory")
p.add_argument("-o", "--outdir", help="directory to write combined HDF5 files")
p.add_argument("-plotgrid", help="plot grid per patch", action="store_true")
p.add_argument("-v", "--verbose", action="store_true")
P = p.parse_args()

if P.verbose:
    logging.basicConfig(level=logging.DEBUG)

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

x2, x3 = get_xlims(indir, times[0], P.plotgrid)

simgrid = indir / "inputs/simgrid.h5"
with h5py.File(simgrid, "r") as fh:
    x1 = fh["x1"][2:-2]

outgrid = outdir / "amrgrid.h5"
print("write", outgrid)
with h5py.File(outgrid, "w") as oh:
    oh["alt"] = h5py.ExternalLink(simgrid, "/alt")
    oh["theta"] = h5py.ExternalLink(simgrid, "/theta")

    oh["x1"] = x1.astype(np.float32)
    oh["x2"] = x2.astype(np.float32)
    oh["x3"] = x3.astype(np.float32)


for t in times:
    combine_files(indir, outdir, t, data_vars, x1, x2, x3)
