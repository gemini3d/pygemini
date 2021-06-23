"""
compare simulation outputs to verify model performance
"""

from __future__ import annotations
from pathlib import Path

from .input import compare_input
from .out import compare_output
from .grid import compare_grid

# keep these two convenience imports for users
from .efield import compare_Efield
from .precip import compare_precip


def compare_all(
    new_dir: Path,
    refdir: Path,
    *,
    tol: dict[str, float] = None,
    plot: bool = True,
    file_format: str = "",
    only: str = None,
) -> dict[str, int]:
    """
    compare two directories across time steps
    """
    new_dir = Path(new_dir).expanduser()
    refdir = Path(refdir).expanduser()

    if new_dir.samefile(refdir):
        raise OSError(f"reference and output are the same directory: {new_dir}")

    # %% fail immediately if grid doesn't match as data would be non-sensical
    if compare_grid(new_dir, refdir, tol=tol, file_format=file_format) != 0:
        raise ValueError(f"grid values do not match {new_dir}  {refdir}")

    errs = {}
    if not only or only == "out":
        e = compare_output(new_dir, refdir, tol=tol, file_format=file_format, plot=plot)
        if e:
            errs["out"] = e

    if not only or only == "in":
        e = compare_input(new_dir, refdir, tol=tol, file_format=file_format, plot=plot)
        if e:
            errs["in"] = e

    return errs
