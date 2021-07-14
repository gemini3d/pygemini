from __future__ import annotations
from pathlib import Path
from datetime import datetime
import logging

import numpy as np
import xarray

from .. import read

try:
    from matplotlib.figure import Figure
except ImportError:
    Figure = None


def plotdiff(
    A: xarray.DataArray,
    B: xarray.DataArray,
    time: datetime,
    new_dir: Path,
    ref_dir: Path,
    *,
    name: str = None,
    imax: int = None,
):
    """

    Parameters
    ----------

    imax: int, optional
        index of maximum difference
    """

    if Figure is None:
        logging.error("Matplotlib not available")
        return

    if not name:
        name = str(A.name)

    assert A.shape == B.shape, f"{name}: size of new and ref arrays don't match"
    assert 1 < A.ndim <= 4, f"failed to plot {A.ndim}-D array {name}: for 4D, 3D, or 2D arrays"

    lx = read.simsize(new_dir)
    is3d = lx[1] != 1 and lx[2] != 1

    A = A.squeeze()
    B = B.squeeze()

    if A.ndim == 4:
        assert A.shape[0] == 7, "4-D arrays must have species as first axis"
        # loop over the species, which are in the first dimension
        for i in range(A.shape[0]):
            plotdiff(A[i], B[i], time, new_dir, ref_dir)

    if A.ndim == 3:
        if A.shape[0] == 7:
            # loop over the species, which are in the first dimension
            for i in range(A.shape[0]):
                plotdiff(A[i], B[i], time, new_dir, ref_dir, name=f"{name}-{i}")
        elif is3d:
            # pick x2 and x3 slice at maximum difference
            im = abs(A - B).argmax(dim=A.dims)
            ix2 = im["x2"].data
            ix3 = im["x3"].data
            plotdiff(
                A[:, :, ix3], B[:, :, ix3], time, new_dir, ref_dir, name=name + "-x2", imax=ix3
            )
            plotdiff(
                A[:, ix2, :], B[:, ix2, :], time, new_dir, ref_dir, name=name + "-x3", imax=ix2
            )
        else:
            raise ValueError("unexpected case, 2D data but in if-tree only for 3D")

        return

    fg = Figure(constrained_layout=True, figsize=(12, 5))
    axs = fg.subplots(1, 3)

    if A.ndim == 2:
        maxdiff = diff2d(A, B, name, fg, axs)
    elif A.ndim == 1:
        maxdiff = diff1d(A, B, name, fg, axs)
    else:
        raise ValueError("expected 2D or 1D")

    axs[0].set_title(str(new_dir))
    axs[1].set_title(str(ref_dir))
    axs[2].set_title(f"diff: {name}")

    tstr = time.isoformat()
    ttxt = f"{name}  {tstr}  maxDiff: {maxdiff:.1e}"
    if imax is not None:
        ttxt += f" maxIndex: {imax}"

    fg.suptitle(ttxt)

    plot_dir = new_dir / "plot_diff"
    plot_dir.mkdir(exist_ok=True)

    fn = plot_dir / f"{name}-{tstr.replace(':','')}.png"
    print("write: ", fn)
    fg.savefig(fn)


def diff1d(A: np.ndarray, B: np.ndarray, name: str, fg, axs) -> float:

    axs[0].plot(A)

    axs[1].plot(B)

    d = A - B

    axs[2].plot(d)

    return abs(d).max().data


def diff2d(A: np.ndarray, B: np.ndarray, name: str, fg, axs) -> float:

    cmap = "bwr" if name.startswith(("J", "v")) else None

    bmin = min(A.min(), B.min())
    bmax = max(A.max(), B.max())

    hi = axs[0].pcolormesh(A, cmap=cmap, vmin=bmin, vmax=bmax)
    fg.colorbar(hi, ax=axs[0])

    hi = axs[1].pcolormesh(B, cmap=cmap, vmin=bmin, vmax=bmax)
    fg.colorbar(hi, ax=axs[1])

    dAB = A - B
    b = max(abs(dAB.min()), abs(dAB.max()))

    hi = axs[2].pcolormesh(A - B, cmap="bwr", vmin=-b, vmax=b)
    fg.colorbar(hi, ax=axs[2])

    return abs(dAB).max().data
