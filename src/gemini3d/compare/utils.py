import json
import importlib.resources as ir
import xarray


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

    Compare raw numeric values only.
    xarray.DataArray arithmetic aligns coordinates,
    which can produce empty arrays when float coordinates
    differ slightly.
    """

    if isinstance(a, xarray.DataArray):
        a = a.data
    if isinstance(b, xarray.DataArray):
        b = b.data

    return (abs(a - b).max() / abs(b).max()) * 100


def load_tol() -> dict[str, float]:
    file = ir.files(f"{__package__}") / "tolerance.json"
    tol_json = file.read_text()
    return json.loads(tol_json)
