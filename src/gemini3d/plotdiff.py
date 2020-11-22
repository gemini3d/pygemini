from matplotlib.pyplot import figure, close
import numpy as np
from pathlib import Path
from datetime import datetime


def plotdiff(
    A: np.ndarray,
    B: np.ndarray,
    name: str,
    time: datetime,
    outdir: Path,
    refdir: Path,
    save: bool = True,
):

    A = A.squeeze()
    B = B.squeeze()

    if A.ndim == 3:
        # loop over the species, which are in the first dimension
        for i in range(A.shape[0]):
            plotdiff(A[i], B[i], f"{name}-{i}", time, outdir, refdir, save)

    fg = figure(constrained_layout=True, figsize=(12, 5))
    axs = fg.subplots(1, 3)

    if A.ndim == 2:
        diff2d(A, B, name, fg, axs)
    elif A.ndim == 1:
        diff1d(A, B, name, fg, axs)
    else:
        close(fg)
        print(f"skipping diff plot: {name}")
        return None

    axs[0].set_title(str(outdir))
    axs[1].set_title(str(refdir))
    axs[2].set_title(f"diff: {name}")

    tstr = time.isoformat()
    ttxt = f"{name}  {tstr}"

    fg.suptitle(ttxt)

    if save:
        fn = outdir / f"{name}-diff-{tstr.replace(':','')}.png"
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
