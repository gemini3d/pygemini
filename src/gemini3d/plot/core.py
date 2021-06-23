from __future__ import annotations
import typing as T
from pathlib import Path
from datetime import datetime

from matplotlib.figure import Figure


def basic(xg: dict[str, T.Any]) -> Figure:
    fig = Figure()
    axs = fig.subplots(1, 3)
    # %% x1
    lx1 = xg["x1"].size
    ax = axs[0]
    ax.plot(range(lx1), xg["x1"] / 1000, marker=".")
    ax.set_ylabel("x1 [km]")
    ax.set_xlabel("index (dimensionless)")
    ax.set_title(f"x1 (upward) lx1 = {lx1}")

    # %% x2
    lx2 = xg["x2"].size
    ax = axs[1]
    ax.plot(xg["x2"] / 1e3, range(lx2), marker=".")
    ax.set_xlabel("x2 [km]")
    ax.set_ylabel("index (dimensionless)")
    ax.set_title(f"x2 (eastward) lx2 = {lx2}")

    # %% x3
    lx3 = xg["x3"].size
    ax = axs[2]
    ax.plot(range(lx3), xg["x3"] / 1e3, marker=".")
    ax.set_ylabel("x3 [km]")
    ax.set_xlabel("index (dimensionless)")
    ax.set_title(f"x3 (northward) lx3 = {lx3}")

    return fig


def stitle(fig: Figure, xg: dict[str, T.Any], ttxt: str = ""):
    """suptitle"""
    if "time" in xg:
        ttxt += f" {xg['time']}"

    if "filename" in xg:
        ttxt += f" {xg['filename']}"

    fig.suptitle(ttxt)


def save_fig(
    fg: Figure, direc: Path, name: str, *, fmt: T.Optional[str] = "png", time: datetime = None
):
    if not fmt:
        fmt = "png"

    if time is None:
        tstr = ""
    else:
        tstr = f"-{time.isoformat().replace(':','')}"

    plot_fn = direc / f"plots/{name}{tstr}.{fmt}"
    plot_fn.parent.mkdir(exist_ok=True)
    print(f"{time} => {plot_fn}")
    fg.savefig(plot_fn)
