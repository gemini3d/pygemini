from __future__ import annotations
from pathlib import Path
from datetime import datetime
import logging

import numpy as np

from .utils import err_pct, load_tol
from .plot import plotdiff
from .. import read, find


def compare_Efield(
    times: list[datetime],
    newdir: Path,
    refdir: Path,
    *,
    tol: dict[str, float] = None,
    plot: bool = False,
    file_format: str = None,
) -> int:

    if tol is None:
        tol = load_tol()

    efield_errs = 0
    # often we reuse Efield inputs without copying over files
    for t in times:
        ref = read.Efield(find.frame(refdir, t))
        new = read.Efield(find.frame(newdir, t), file_format=file_format)
        for k in ("Exit", "Eyit", "Vminx1it", "Vmaxx1it"):
            b = ref[k]
            a = new[k]

            assert a.shape == b.shape, f"{k}: ref shape {b.shape} != data shape {a.shape}"

            if not np.allclose(a, b, rtol=tol["rtol"], atol=tol["atol"]):
                efield_errs += 1
                logging.error(f"{k} {t}  {err_pct(a, b):.1f} %")
                if plot:
                    plotdiff(a, b, t, newdir.parent, refdir.parent)

    if efield_errs == 0:
        logging.info(f"OK: Efield {newdir}")

    return efield_errs
