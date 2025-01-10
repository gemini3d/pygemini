from __future__ import annotations
import json
import importlib.resources as ir


def err_pct(a, b) -> float:
    """compute maximum error percent

    Parameters
    ----------

    a: numpy.ndarray
        new data
    b: numpy.ndarray
        reference data

    Returns
    -------

    float
        maximum error percent
    """

    return (abs(a - b).max() / abs(b).max()).item() * 100


def load_tol() -> dict[str, float]:
    file = ir.files(f"{__package__}") / "tolerance.json"
    tol_json = file.read_text()
    return json.loads(tol_json)
