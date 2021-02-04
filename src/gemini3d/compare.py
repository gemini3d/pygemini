"""
compare simulation outputs to verify model performance
"""

from __future__ import annotations
from pathlib import Path
import numpy as np
import logging
import sys
import argparse
from datetime import datetime
import typing as T

from . import read
from . import find

try:
    from .plot.diff import plotdiff
except ImportError:
    plotdiff = None

TOL = {
    "rtol": 1e-5,
    "rtolN": 1e-5,
    "rtolT": 1e-5,
    "rtolJ": 1e-5,
    "rtolV": 1e-5,
    "atol": 1e-8,
    "atolN": 1e9,
    "atolT": 100,
    "atolJ": 1e-7,
    "atolV": 50,
}


def cli():

    p = argparse.ArgumentParser(description="Compare simulation file outputs and inputs")
    p.add_argument("outdir", help="directory to compare")
    p.add_argument("refdir", help="reference directory")
    p.add_argument("-only", help="only check in or out", choices=["in", "out"])
    p.add_argument(
        "-file_format",
        help="specify file format to read from output dir",
        choices=["h5", "nc", "raw"],
    )
    P = p.parse_args()

    errs = compare_all(P.outdir, refdir=P.refdir, tol=TOL, file_format=P.file_format, only=P.only)

    if errs:
        for e, v in errs.items():
            print(f"{e} has {v} errors", file=sys.stderr)

        raise SystemExit(f"FAIL: compare {P.outdir}")

    print(f"OK: Gemini comparison {P.outdir} {P.refdir}")


def compare_all(
    outdir: Path,
    refdir: Path,
    tol: dict[str, float] = TOL,
    *,
    plot: bool = True,
    file_format: str = None,
    only: str = None,
) -> dict[str, int]:
    """
    compare two directories across time steps
    """
    outdir = Path(outdir).expanduser()
    refdir = Path(refdir).expanduser()

    if outdir.samefile(refdir):
        raise OSError(f"reference and output are the same directory: {outdir}")

    # %% fail hard if grid doesn't match, because otherwise data is non-sensical
    if compare_grid(outdir, refdir, tol, file_format=file_format) != 0:
        raise ValueError("grid values do not match {outdir}  {refdir}")

    errs = {}
    if not only or only == "out":
        e = compare_output(outdir, refdir, tol, file_format=file_format, plot=plot)
        if e:
            errs["out"] = e

    if not only or only == "in":
        e = compare_input(outdir, refdir, tol, file_format=file_format, plot=plot)
        if e:
            errs["in"] = e

    return errs


def compare_grid(
    outdir: Path, refdir: Path, tol: dict[str, float] = TOL, *, file_format: str = None
) -> int:

    ref = read.grid(refdir)
    new = read.grid(outdir, file_format=file_format)

    if not ref:
        raise FileNotFoundError(f"No simulation grid in {refdir}")
    if not new:
        raise FileNotFoundError(f"No simulation grid in {outdir}")

    errs = 0

    for k in ref.keys():
        if not isinstance(ref[k], np.ndarray):
            continue

        assert (
            ref[k].shape == new[k].shape
        ), f"{k}: ref shape {ref[k].shape} != data shape {new[k].shape}"
        if not np.allclose(ref[k], new[k], rtol=tol["rtol"], atol=tol["atol"]):
            errs += 1
            logging.error(f"{k}  {err_pct(ref[k], new[k]):.1f} %")

    return errs


def compare_input(
    outdir: Path,
    refdir: Path,
    tol: dict[str, float] = TOL,
    *,
    file_format: str = None,
    plot: bool = True,
) -> int:

    names = ("ns", "Ts", "vs1")

    ref_params = read.config(refdir)
    if not ref_params:
        raise FileNotFoundError(f"{refdir} does not appear to contain config.nml")
    ref_indir = refdir / ref_params["indat_file"].parts[-2]
    ref = read.data(ref_indir / ref_params["indat_file"].name, var=names)

    new_params = read.config(outdir)
    if not new_params:
        raise FileNotFoundError(f"{outdir} does not appear to contain config.nml")
    if len(new_params["time"]) <= 1:
        raise ValueError(
            f"{outdir} simulation did not run long enough, must run for more than one time step"
        )
    new_indir = outdir / new_params["indat_file"].parts[-2]
    new = read.data(new_indir / new_params["indat_file"].name, var=names)

    errs = 0
    # %% initial conditions
    itols = ("N", "T", "V")

    for k, j in zip(names, itols):
        b = ref[k]
        a = new[k]

        assert a.shape == b.shape, f"{k}: ref shape {b.shape} != data shape {a.shape}"

        if not np.allclose(a, b, rtol=0.1 * tol[f"rtol{j}"], atol=0.1 * tol[f"atol{j}"]):
            errs += 1
            logging.error(f"{k}  {err_pct(a, b):.1f} %")

            if plot and plotdiff is not None:
                plotdiff(a, b, k, ref_params["time"][0], outdir, refdir)

    if "precdir" in new_params:
        prec_errs = compare_precip(
            ref_params["time"],
            new_indir / new_params["precdir"].name,
            ref_indir / ref_params["precdir"].name,
            tol,
            plot=plot,
            file_format=file_format,
        )
        errs += prec_errs

    if "E0dir" in new_params:
        efield_errs = compare_Efield(
            ref_params["time"],
            new_indir / new_params["E0dir"].name,
            ref_indir / ref_params["E0dir"].name,
            tol,
            plot=plot,
            file_format=file_format,
        )
        errs += efield_errs

    return errs


def err_pct(a: np.ndarray, b: np.ndarray) -> float:
    """ compute maximum error percent """

    return (abs(a - b).max() / abs(b).max()).item() * 100


def compare_precip(
    times: list[datetime],
    newdir: Path,
    refdir: Path,
    tol: dict[str, float] = TOL,
    *,
    plot: bool = False,
    file_format: str = None,
) -> int:

    prec_errs = 0

    # often we reuse precipitation inputs without copying over files
    for t in times:
        ref = read.precip(find.frame(refdir, t))
        new = read.precip(find.frame(newdir, t), file_format=file_format)

        for k in ref.keys():
            b = np.atleast_1d(ref[k])
            a = np.atleast_1d(new[k])

            assert a.shape == b.shape, f"{k}: ref shape {b.shape} != data shape {a.shape}"

            if not np.allclose(a, b, rtol=tol["rtol"], atol=tol["atol"]):
                prec_errs += 1
                logging.error(f"{k} {t}  {err_pct(a, b):.1f} %")
                if plot and plotdiff is not None:
                    plotdiff(a, b, k, t, newdir.parent, refdir.parent)
            if prec_errs == 0:
                logging.info(f"OK: {k}  {newdir}")

    return prec_errs


def compare_Efield(
    times: list[datetime],
    newdir: Path,
    refdir: Path,
    tol: dict[str, float] = TOL,
    *,
    plot: bool = False,
    file_format: str = None,
) -> int:

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
                if plot and plotdiff is not None:
                    plotdiff(a, b, k, t, newdir.parent, refdir.parent)

    if efield_errs == 0:
        logging.info(f"OK: Efield {newdir}")

    return efield_errs


def compare_output(
    outdir: Path,
    refdir: Path,
    tol: dict[str, float],
    *,
    file_format: str = None,
    plot: bool = True,
) -> int:
    """compare simulation outputs"""

    ref: dict[str, T.Any] = {}
    errs = 0
    a: np.ndarray = None

    params = read.config(outdir)
    if not params:
        raise FileNotFoundError(f"{outdir} does not appear to contain config.nml")
    if len(params["time"]) <= 1:
        raise ValueError(
            f"{outdir} simulation did not run long enough, must run for more than one time step"
        )

    for i, t in enumerate(params["time"]):
        st = f"UTsec {t}"
        A = read.frame(outdir, t, file_format=file_format)
        if not A:
            raise FileNotFoundError(f"{outdir} does not appear to contain data at {t}")
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
                if plot and plotdiff is not None:
                    plotdiff(a, b, k, t, outdir, refdir)
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


if __name__ == "__main__":
    cli()
