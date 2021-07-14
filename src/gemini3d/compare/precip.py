from __future__ import annotations
from datetime import datetime
from pathlib import Path
import logging

import numpy as np

from .plot import plotdiff
from .utils import err_pct, load_tol
from .. import read, find


def compare_precip(
    times: list[datetime],
    newdir: Path,
    refdir: Path,
    *,
    tol: dict[str, float] = None,
    plot: bool = False,
    file_format: str = None,
) -> int:

    prec_errs = 0

    if tol is None:
        tol = load_tol()

    # often we reuse precipitation inputs without copying over files
    for t in times:
        ref = read.precip(find.frame(refdir, t))
        new = read.precip(find.frame(newdir, t), file_format=file_format)

        for k in ref.keys():
            b = ref[k]
            a = new[k]

            assert a.shape == b.shape, f"{k}: ref shape {b.shape} != data shape {a.shape}"

            if not np.allclose(a, b, rtol=tol["rtol"], atol=tol["atol"]):
                prec_errs += 1
                logging.error(f"{k} {t}  {err_pct(a, b):.1f} %")
                if plot:
                    plotdiff(a, b, t, newdir.parent, refdir.parent)
            if prec_errs == 0:
                logging.info(f"OK: {k}  {newdir}")

    return prec_errs
