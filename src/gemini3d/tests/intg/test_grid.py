"""
these test that PyGemini generates inputs that match expectations
"""

from datetime import datetime
import importlib.resources
import pytest
import os

import gemini3d
import gemini3d.web
import gemini3d.write as write
import gemini3d.grid
from gemini3d.compare import compare_grid
import gemini3d.read
import gemini3d.model
import gemini3d.find


@pytest.mark.parametrize("name", ["mini2dew_fang"])
def test_file_time(name):
    # get files if needed
    with importlib.resources.path("gemini3d.tests.data", "__init__.py") as fn:
        test_dir = gemini3d.web.download_and_extract(name, fn.parent)

    t0 = datetime(2013, 2, 20, 5, 0, 0)

    file = gemini3d.find.frame(test_dir, time=t0)
    time = gemini3d.read.time(file)
    assert time == t0


@pytest.mark.parametrize(
    "name,file_format",
    [
        ("mini2dew_fang", "h5"),
        ("mini3d_fang", "h5"),
        ("mini2dew_fang", "nc"),
        ("mini3d_fang", "nc"),
    ],
)
def test_grid(name, file_format, tmp_path, monkeypatch):

    if file_format.endswith("nc"):
        pytest.importorskip("netCDF4")

    if not os.environ.get("GEMINI_CIROOT"):
        monkeypatch.setenv("GEMINI_CIROOT", str(tmp_path / "gemini_data"))

    # get files if needed
    with importlib.resources.path("gemini3d.tests.data", "__init__.py") as fn:
        test_dir = gemini3d.web.download_and_extract(name, fn.parent)

    # setup new test data
    cfg = gemini3d.read.config(test_dir)
    xg = gemini3d.grid.cartesian.cart3d(cfg)

    # path patch
    cfg["out_dir"] = tmp_path
    cfg["indat_size"] = cfg["out_dir"] / cfg["indat_size"]
    cfg["indat_grid"] = cfg["out_dir"] / cfg["indat_grid"]

    write.grid(cfg, xg, file_format=file_format)

    errs = compare_grid(cfg["out_dir"], test_dir, file_format=file_format)

    assert errs == 0, f"grid mismatch {cfg['out_dir']}  {test_dir}"
