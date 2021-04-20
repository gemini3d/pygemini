import numpy as np
from pathlib import Path
from datetime import datetime
import argparse
import logging
from dateutil.parser import parse

from .. import read
from ..utils import to_datetime

try:
    from matplotlib.figure import Figure
except ImportError:
    Figure = None


def plotdiff(
    A: np.ndarray,
    B: np.ndarray,
    name: str,
    time: datetime,
    new_dir: Path,
    refdir: Path,
):

    if Figure is None:
        logging.error("Matplotlib not available")
        return

    assert A.shape == B.shape, "size of new and ref arrays don't match"
    assert A.ndim <= 3, "for 3D or 2D arrays only"

    lx = read.simsize(new_dir)
    is3d = lx[1] != 1 and lx[2] != 1

    A = A.squeeze()
    B = B.squeeze()

    if A.ndim == 3:
        if A.shape[0] == 7:
            # loop over the species, which are in the first dimension
            for i in range(A.shape[0]):
                plotdiff(A[i], B[i], f"{name}-{i}", time, new_dir, refdir)
        elif is3d:
            # pick x2 and x3 slice halfway
            i = round(A.shape[2] / 2)
            plotdiff(A[:, :, i], B[:, :, i], name + "-x2", time, new_dir, refdir)
            i = round(A.shape[1] / 2)
            plotdiff(A[:, i, :], B[:, i, :], name + "-x3", time, new_dir, refdir)
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
    axs[1].set_title(str(refdir))
    axs[2].set_title(f"diff: {name}")

    tstr = time.isoformat()
    ttxt = f"{name}  {tstr}  maxDiff: {maxdiff}"

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

    return abs(d.max())


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

    return abs(dAB.max())


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="plot differences between a variable in ref/new data files"
    )
    p.add_argument("new_path", help="new data directory or file")
    p.add_argument("ref_path", help="Reference data directory or file")
    p.add_argument("name", help="variable name")
    p.add_argument("-t", "--time", help="requested time (if directory given)")
    P = p.parse_args()

    ref_path = Path(P.ref_path).expanduser().resolve(strict=True)
    new_path = Path(P.new_path).expanduser().resolve(strict=True)

    if P.time:
        time = parse(P.time)
        new = read.frame(new_path, time, var=P.name)
        ref = read.frame(ref_path, time, var=P.name)
    else:
        if not ref_path.is_file():
            raise FileNotFoundError(f"{ref_path} must be a file when not specifying time")
        if not new_path.is_file():
            raise FileNotFoundError(f"{new_path} must be a file when not specifying time")

        new = read.data(new_path, var=P.name)
        ref = read.data(ref_path, var=P.name)

        new_path = new_path.parent
        ref_path = ref_path.parent

    new = new[P.name]
    ref = ref[P.name]

    plotdiff(new, ref, P.name, to_datetime(new.time), new_path, ref_path)
