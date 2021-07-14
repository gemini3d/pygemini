from __future__ import annotations
from pathlib import Path
import logging
import typing as T

import numpy as np

from .plot import plotdiff
from .utils import err_pct, load_tol
from .. import read


def compare_output(
    new_dir: Path,
    refdir: Path,
    *,
    tol: dict[str, float] = None,
    file_format: str = "",
    plot: bool = True,
) -> int:
    """compare simulation outputs"""

    ref: dict[str, T.Any] = {}
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
        A = read.frame(new_dir, t, file_format=file_format)
        if not A:
            raise FileNotFoundError(f"{new_dir} does not appear to contain data at {t}")
        B = read.frame(refdir, t)
        # don't specify file_format for reference,
        # so that one reference file format can check multiple "out" format

        names = ["ne", "v1", "v2", "v3", "Ti", "Te", "J1", "J2", "J3"]
        itols = ["N", "V", "V", "V", "T", "T", "J", "J", "J"]

        for k, j in zip(names, itols):
            a = A[k]
            b = B[k]

            assert a.shape == b.shape, f"{k} time {i} {t}: shape: ref {b.shape} != data {a.shape}"

            if not np.allclose(a, b, rtol=tol[f"rtol{j}"], atol=tol[f"atol{j}"]):
                errs += 1
                logging.error(f"{k} {st}   {err_pct(a, b):.1f}")
                if plot:
                    plotdiff(a, b, t, new_dir, refdir)
        # %% assert time steps have unique output (earth always rotating...)
        if i > 1:
            names = ["ne", "v1", "v2", "v3"]
            itols = ["N", "V", "V", "V"]
            for k, j in zip(names, itols):
                if np.allclose(
                    ref[k], a, rtol=0.0001 * tol[f"rtol{j}"], atol=0.0001 * tol[f"atol{j}"]
                ):
                    errs += 1
                    logging.error(f"{k} {st} too similar to prior step")

        if i == 3:
            for k in ("Ti", "Te"):
                if np.allclose(ref[k], A[k], rtol=0.01 * tol["rtolT"], atol=0.1 * tol["atolT"]):
                    errs += 1
                    logging.error(f"{k} {st} too similar to prior step")

        if i == 2:
            for k in ("J1", "J2", "J3"):
                if np.allclose(ref[k], a, rtol=0.01 * tol["rtolJ"], atol=0.1 * tol["atolJ"]):
                    errs += 1
                    logging.error(f"{k} {st} too similar to prior step")

        ref.update(A)

    return errs
