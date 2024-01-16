"""
gemini3d.compare: compare simulation outputs to verify model performance

Command line usage:

    python -m gemini3d.compare /path/to/ref_data /path/to/new_data
"""

from __future__ import annotations
from pathlib import Path

from .inputs import compare_input
from .out import compare_output
from .grid import compare_grid

# convenience imports
from .efield import compare_Efield
from .precip import compare_precip

__all__ = ["compare_all", "compare_Efield", "compare_precip"]


def compare_all(
    new_dir: Path,
    ref_dir: Path,
    *,
    tol: dict[str, float] | None = None,
    plot: bool = True,
    only: str | None = None,
) -> dict[str, int]:
    """
    compare two directories across time steps
    """
    new_dir = Path(new_dir).expanduser().resolve(strict=True)
    ref_dir = Path(ref_dir).expanduser().resolve(strict=True)

    if new_dir.samefile(ref_dir):
        raise OSError(f"reference and output are the same directory: {new_dir}")

    # %% fail immediately if grid doesn't match as data would be non-sensical
    if compare_grid(new_dir, ref_dir, tol=tol) != 0:
        raise ValueError(f"grid values do not match {new_dir}  {ref_dir}")

    errs = {}
    if not only or only == "out":
        e = compare_output(new_dir, ref_dir, tol=tol, plot=plot)
        if e:
            errs["out"] = e

    if not only or only == "in":
        e = compare_input(new_dir, ref_dir, tol=tol, plot=plot)
        if e:
            errs["in"] = e

    return errs
