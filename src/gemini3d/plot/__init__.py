from pathlib import Path
from matplotlib.pyplot import close
import typing as T
from datetime import datetime

from .. import read
from .vis import grid2plotfun


PARAMS = ["ne", "v1", "Ti", "Te", "J1", "v2", "v3", "J2", "J3", "Phitop"]


def get_files(direc: Path) -> T.List[Path]:
    for ext in ("h5", "nc", "dat"):
        flist = sorted(direc.glob(f"*.{ext}"))
        if len(flist):
            break
    if not flist:
        raise FileNotFoundError(f"No files to plot in {direc}")

    return flist


def plot_3d(direc: Path, var: T.Sequence[str], saveplot_fmt: str = None):
    from . import vis3d

    direc = Path(direc).expanduser().resolve(strict=True)

    cfg = read.config(direc)
    for t in cfg["time"]:
        vis3d.frame(direc, time=t, var=var, saveplot_fmt=saveplot_fmt)


def plot_all(direc: Path, var: T.Sequence[str] = None, saveplot_fmt: str = None):

    direc = Path(direc).expanduser().resolve(strict=True)

    cfg = read.config(direc)
    # %% loop over files / time
    for t in cfg["time"]:
        frame(direc, time=t, var=var, saveplot_fmt=saveplot_fmt)


def frame(
    direc: Path,
    time: datetime,
    saveplot_fmt: str = None,
    var: T.Sequence[str] = None,
    xg: T.Dict[str, T.Any] = None,
):
    """
    if save_dir, plots will not be visible while generating to speed plot writing
    """
    if not var:
        var = PARAMS

    if not xg:
        xg = read.grid(direc)
    plotfun = grid2plotfun(xg)

    dat = read.frame(direc, time)

    for k in var:
        if k not in dat:  # not present at this time step, often just the first time step
            continue

        fg = plotfun(time, xg, dat[k][1].squeeze(), k, wavelength=dat.get("wavelength"))

        if saveplot_fmt:
            plot_fn = direc / "plots" / f"{k}-{time.isoformat().replace(':','')}.png"
            plot_fn.parent.mkdir(exist_ok=True)
            print(f"{dat['time']} => {plot_fn}")
            fg.savefig(plot_fn)
            close(fg)
