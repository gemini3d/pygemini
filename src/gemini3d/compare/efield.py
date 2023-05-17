from __future__ import annotations
from pathlib import Path
import typing
import logging

import numpy as np

from .utils import err_pct, load_tol
from .plot import plotdiff
from ..efield import get_times as efield_times
from .. import read, find


def compare_Efield(
    ref_cfg: dict[str, typing.Any],
    new_dir: Path,
    ref_dir: Path,
    *,
    tol: dict[str, float] | None = None,
    plot: bool = False,
) -> int:
    """
    Compare input electric field data to reference data
    """

    new_dir = Path(new_dir).expanduser().resolve(strict=True)
    ref_dir = Path(ref_dir).expanduser().resolve(strict=True)

    if tol is None:
        tol = load_tol()

    efield_errs = 0

    for t in efield_times(ref_cfg):
        ref = read.Efield(find.frame(ref_dir, t))
        new = read.Efield(find.frame(new_dir, t))
        for k in {"Exit", "Eyit", "Vminx1it", "Vmaxx1it"}:
            b = ref[k]
            a = new[k]

            assert a.shape == b.shape, f"{k}: ref shape {b.shape} != data shape {a.shape}"

            if not np.allclose(a, b, rtol=tol["rtol"], atol=tol["atol"]):
                efield_errs += 1
                logging.error(f"{k} {t}  {err_pct(a, b):.1f} %")
                if plot:
                    plotdiff(a, b, t, new_dir.parent, ref_dir.parent)

    if efield_errs == 0:
        logging.info(f"OK: Efield {new_dir}")

    return efield_errs
