from __future__ import annotations
import importlib.resources
import json

import xarray


def err_pct(a: xarray.DataArray, b: xarray.DataArray) -> float:
    """compute maximum error percent"""

    return (abs(a - b).max() / abs(b).max()).item() * 100


def load_tol() -> dict[str, float]:
    tol_json = importlib.resources.read_text("gemini3d.compare", "tolerance.json")
    return json.loads(tol_json)
