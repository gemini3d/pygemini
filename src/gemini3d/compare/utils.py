from __future__ import annotations
import json

import xarray

from ..utils import get_pkg_file


def err_pct(a: xarray.DataArray, b: xarray.DataArray) -> float:
    """compute maximum error percent"""

    return (abs(a - b).max() / abs(b).max()).item() * 100


def load_tol() -> dict[str, float]:
    tol_json = get_pkg_file("gemini3d.compare", "tolerance.json").read_text()
    return json.loads(tol_json)
