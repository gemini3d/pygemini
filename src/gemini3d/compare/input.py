from __future__ import annotations
from pathlib import Path
import logging

import numpy as np

from .. import read

from .plot import plotdiff
from .utils import err_pct, load_tol
from .precip import compare_precip
from .efield import compare_Efield


def compare_input(
    new_dir: Path,
    ref_dir: Path,
    *,
    tol: dict[str, float] | None = None,
    plot: bool = True,
) -> int:
    """
    compares simulation input data to reference data, including:

    * background plasma
    * precipitation
    * electric field
    """

    names = {"ns", "Ts", "vs1"}

    new_dir = Path(new_dir).expanduser().resolve(strict=True)
    ref_dir = Path(ref_dir).expanduser().resolve(strict=True)

    ref_cfg = read.config(ref_dir)
    ref_indir = ref_dir / ref_cfg["indat_file"].parts[-2]
    ref = read.frame(ref_indir / ref_cfg["indat_file"].name, var=names)

    new_cfg = read.config(new_dir)
    if len(new_cfg["time"]) <= 1:
        raise ValueError(
            f"{new_dir} simulation did not run long enough, must run for more than one time step"
        )
    new_indir = new_dir / new_cfg["indat_file"].parts[-2]
    new = read.frame(new_indir / new_cfg["indat_file"].name, var=names)

    if tol is None:
        tol = load_tol()

    errs = 0
    # %% initial conditions

    for k in names:
        b = ref[k]
        a = new[k]

        n = k[0].upper()

        assert a.shape == b.shape, f"{k}: ref shape {b.shape} != data shape {a.shape}"

        if not np.allclose(a, b, rtol=0.1 * tol[f"rtol{n}"], atol=0.1 * tol[f"atol{n}"]):
            errs += 1
            logging.error(f"{k}  {err_pct(a, b):.1f} %")

            if plot:
                if k == "ns":
                    # just plot electron density
                    a = a[-1]
                    b = b[-1]
                plotdiff(a, b, ref_cfg["time"][0], new_dir, ref_dir)

    if "precdir" in new_cfg:
        prec_errs = compare_precip(
            ref_cfg,
            new_indir / new_cfg["precdir"].name,
            ref_indir / ref_cfg["precdir"].name,
            tol=tol,
            plot=plot,
        )
        errs += prec_errs

    if "E0dir" in new_cfg:
        efield_errs = compare_Efield(
            ref_cfg,
            new_indir / new_cfg["E0dir"].name,
            ref_indir / ref_cfg["E0dir"].name,
            tol=tol,
            plot=plot,
        )
        errs += efield_errs

    return errs
