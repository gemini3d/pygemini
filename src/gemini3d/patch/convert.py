"""
for converting AMR patch files to single file per time step
"""

from __future__ import annotations
from pathlib import Path
from datetime import datetime
import logging

import numpy as np
import h5py

from . import filenames2times, time2filename, patch_grid
from .. import utils
from .plot import grid_step

from matplotlib.pyplot import figure


COMP_LEVEL = 6  # arbitrary GZIP compression level


def convert(indir: Path, outdir: Path, data_vars: set[str], plotgrid: bool = False):
    times = filenames2times(indir)

    # Need to get extents by scanning all files.
    # FIXME: Does ForestClaw have a way to write this without this inefficient scan?

    x2, x3 = get_xlims(indir, times[0], plotgrid)

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


def get_xlims(path: Path, time: datetime, plotgrid: bool = False) -> tuple:
    """
    build up each axis by scanning files

    Remember HDF5 datasets are always in C-order, even when written from Fortran program.
    """

    x2 = np.array(0)
    x3 = np.array(0)

    pat = utils.datetime2stem(time) + "_*.h5"

    if plotgrid:
        fg = figure()
        ax = fg.gca()

    files = sorted(path.glob(pat))

    N = len(files)  # number of shades to plot

    for i, f in enumerate(files):
        x2new, x3new = patch_grid(f)

        x2 = np.append(x2, x2new)
        x3 = np.append(x3, x3new)

        if plotgrid:
            grid_step(x2new, x3new, i, N, ax)

    x2 = np.unique(x2)
    x3 = np.unique(x3)

    return x2, x3


def get_indices(lims: tuple[float, float], x) -> tuple[int, int]:
    """
    get the indices of the first and last elements within the limits
    """

    i0 = (lims[0] >= x).nonzero()[0][-1].item()
    i1 = (lims[1] > x).nonzero()[0][-1].item()

    return i0, i1


def combine_files(
    indir: Path,
    outdir: Path,
    time: datetime,
    var: set[str],
    x1,
    x2,
    x3,
):
    outfn = time2filename(outdir, time)

    lx = (x1.size, x2.size, x3.size)

    pat = utils.datetime2stem(time) + "_*.h5"

    print("write", outfn, "lx: ", lx)
    with h5py.File(outfn, "w") as oh:
        oh.create_dataset(
            "/time/ymd", dtype=np.int32, data=(time.year, time.month, time.day)
        )
        oh.create_dataset(
            "/time/hms", dtype=np.int32, data=(time.hour, time.minute, time.second)
        )
        oh.create_dataset("/time/microsecond", dtype=np.int32, data=time.microsecond)
        oh.create_dataset(
            "/time/UThour",
            dtype=np.float64,
            data=time.hour
            + time.minute / 60
            + time.second / 3600
            + time.microsecond / 3600e6,
        )
        for f in indir.glob(pat):
            with h5py.File(f, "r") as ih:
                ix2 = get_indices(ih["x2lims"], x2)
                ix3 = get_indices(ih["x3lims"], x3)
                for v in var:
                    convert_var(oh, ih, v, lx, ix2, ix3)


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
