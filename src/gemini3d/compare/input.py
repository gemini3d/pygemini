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
    refdir: Path,
    *,
    tol: dict[str, float] = None,
    file_format: str = None,
    plot: bool = True,
) -> int:

    names = {"ns", "Ts", "vs1"}

    ref_params = read.config(refdir)
    ref_indir = refdir / ref_params["indat_file"].parts[-2]
    ref = read.data(ref_indir / ref_params["indat_file"].name, var=names)

    new_params = read.config(new_dir)
    if len(new_params["time"]) <= 1:
        raise ValueError(
            f"{new_dir} simulation did not run long enough, must run for more than one time step"
        )
    new_indir = new_dir / new_params["indat_file"].parts[-2]
    new = read.data(new_indir / new_params["indat_file"].name, var=names)

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
                plotdiff(a, b, ref_params["time"][0], new_dir, refdir)

    if "precdir" in new_params:
        prec_errs = compare_precip(
            ref_params["time"],
            new_indir / new_params["precdir"].name,
            ref_indir / ref_params["precdir"].name,
            tol=tol,
            plot=plot,
            file_format=file_format,
        )
        errs += prec_errs

    if "E0dir" in new_params:
        efield_errs = compare_Efield(
            ref_params["time"],
            new_indir / new_params["E0dir"].name,
            ref_indir / ref_params["E0dir"].name,
            tol=tol,
            plot=plot,
            file_format=file_format,
        )
        errs += efield_errs

    return errs
