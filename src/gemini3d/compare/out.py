from __future__ import annotations
from pathlib import Path
import logging

import xarray
import numpy as np

from .plot import plotdiff
from .utils import err_pct, load_tol
from .. import read


def compare_output(
    new_dir: Path,
    ref_dir: Path,
    *,
    tol: dict[str, float] | None = None,
    plot: bool = True,
) -> int:
    """compare simulation outputs"""

    new_dir = Path(new_dir).expanduser().resolve(strict=True)
    ref_dir = Path(ref_dir).expanduser().resolve(strict=True)

    ref = xarray.Dataset()
    errs = 0

    params = read.config(new_dir)
    if len(params["time"]) <= 1:
        raise ValueError(
            f"{new_dir} simulation did not run long enough, must run for more than one time step"
        )

    if tol is None:
        tol = load_tol()

    for i, t in enumerate(params["time"]):
        st = f"UTsec {t}"
        A = read.frame(new_dir, t)
        if not A:
            raise FileNotFoundError(f"{new_dir} does not appear to contain data at {t}")
        B = read.frame(ref_dir, t)

        names = ["ne", "v1", "v2", "v3", "Ti", "Te", "J1", "J2", "J3"]
        itols = ["N", "V", "V", "V", "T", "T", "J", "J", "J"]

        for k, j in zip(names, itols):
            a = A[k]
            b = B[k]

            assert (
                a.shape == b.shape
            ), f"{k} time {i} {t}: shape: ref {b.shape} != data {a.shape}"

            if not np.allclose(a, b, rtol=tol[f"rtol{j}"], atol=tol[f"atol{j}"]):
                errs += 1
                logging.error(f"{k} {st}   {err_pct(a, b):.1f}")
                if plot:
                    plotdiff(a, b, t, new_dir, ref_dir)

        ref.update(A)

    return errs
