"""
The gemini3d.patch module is provisional and subject to change.
It is used for reading and plotting data with adaptive mesh refinement (AMR).
"""

from __future__ import annotations
import typing
from pathlib import Path
from datetime import datetime

import h5py
import numpy as np

from .. import utils


def time2filename(path: Path, time: datetime) -> Path:
    return path / (utils.datetime2stem(time) + ".h5")


def filenames2times(indir: Path) -> list[datetime]:
    """
    get times from filenames
    """

    indir = Path(indir).expanduser()
    if not indir.is_dir():
        raise NotADirectoryError(indir)

    times = []
    names = set([f.stem[:21] for f in indir.glob("*_*.*_*.h5")])
    for n in names:
        times.append(utils.filename2datetime(n))

    return times


def patch_grid(file: Path) -> tuple[typing.Any, typing.Any]:
    with h5py.File(file, "r") as fh:
        # TODO: AMR code needs to write actual x1 points, as it's not linearly spaced
        # x1new = np.linspace(
        #     fh["x1lims"][0], fh["x1lims"][1], num=fh["nsall"].shape[-1], endpoint=False
        # )
        # x1 = np.append(x1, x1new)

        x2 = np.linspace(
            fh["x2lims"][0], fh["x2lims"][1], num=fh["nsall"].shape[-2], endpoint=False
        )
        x3 = np.linspace(
            fh["x3lims"][0], fh["x3lims"][1], num=fh["nsall"].shape[-3], endpoint=False
        )

    return x2, x3
