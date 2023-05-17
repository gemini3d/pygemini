from __future__ import annotations
import typing
from pathlib import Path
import logging

import numpy as np

from .plot import plotdiff
from .utils import err_pct, load_tol
from .. import read, find
from ..particles import get_times as precip_times


def compare_precip(
    cfg: dict[str, typing.Any],
    new_dir: Path,
    ref_dir: Path,
    *,
    tol: dict[str, float] | None = None,
    plot: bool = False,
) -> int:
    new_dir = Path(new_dir).expanduser().resolve(strict=True)
    ref_dir = Path(ref_dir).expanduser().resolve(strict=True)

    prec_errs = 0

    if tol is None:
        tol = load_tol()

    # often we reuse precipitation inputs without copying over files
    for t in precip_times(cfg):
        ref = read.precip(find.frame(ref_dir, t))
        new = read.precip(find.frame(new_dir, t))

        for k in ref.keys():
            b = ref[k]
            a = new[k]

            assert a.shape == b.shape, f"{k}: ref shape {b.shape} != data shape {a.shape}"

            if not np.allclose(a, b, rtol=tol["rtol"], atol=tol["atol"]):
                prec_errs += 1
                logging.error(f"{k} {t}  {err_pct(a, b):.1f} %")
                if plot:
                    plotdiff(a, b, t, new_dir.parent, ref_dir.parent)
            if prec_errs == 0:
                logging.info(f"OK: {k}  {new_dir}")

    return prec_errs
