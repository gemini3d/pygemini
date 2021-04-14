from matplotlib.figure import Figure
import numpy as np
from pathlib import Path
from datetime import datetime
import logging
import argparse
from dateutil.parser import parse

from . import read
from ..utils import to_datetime


def plotdiff(
    A: np.ndarray,
    B: np.ndarray,
    name: str,
    time: datetime,
    new_dir: Path,
    refdir: Path,
):

    A = A.squeeze()
    B = B.squeeze()

    if A.ndim == 3:
        # loop over the species, which are in the first dimension
        for i in range(A.shape[0]):
            plotdiff(A[i], B[i], f"{name}-{i}", time, new_dir, refdir)

    if A.ndim not in (1, 2):
        logging.error(f"skipping diff plot: {name}")
        return None

    fg = Figure(constrained_layout=True, figsize=(12, 5))
    axs = fg.subplots(1, 3)

    if A.ndim == 2:
        diff2d(A, B, name, fg, axs)
    elif A.ndim == 1:
        diff1d(A, B, name, fg, axs)

    axs[0].set_title(str(new_dir))
    axs[1].set_title(str(refdir))
    axs[2].set_title(f"diff: {name}")

    tstr = time.isoformat()
    ttxt = f"{name}  {tstr}"

    fg.suptitle(ttxt)

    fn = new_dir / f"{name}-diff-{tstr.replace(':','')}.png"
    print("writing", fn)
    fg.savefig(fn)


def diff1d(A: np.ndarray, B: np.ndarray, name: str, fg, axs):

    axs[0].plot(A)

    axs[1].plot(B)

    axs[2].plot(A - B)


def diff2d(A: np.ndarray, B: np.ndarray, name: str, fg, axs):

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
