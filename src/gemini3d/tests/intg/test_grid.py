"""
these test that PyGemini generates inputs that match expectations
"""

from pytest import approx

import numpy as np
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
import gemini3d.grid.convert as cvt


@pytest.mark.parametrize("name", ["mini2dew_fang"])
def test_file_time(name):
    # get files if needed
    with importlib.resources.path("gemini3d.tests.data", "__init__.py") as fn:
        test_dir = gemini3d.web.download_and_extract(name, fn.parent)

    t0 = datetime(2013, 2, 20, 5, 0, 0)

    file = gemini3d.find.frame(test_dir, time=t0)
    time = gemini3d.read.time(file)
    assert time == t0


@pytest.mark.parametrize("glon, glat", [(30, -60), (45, -45), (90, 30), (180, 45), (270, 60)])
def test_convert_scalar(glon, glat):
    phi, theta = cvt.geog2geomag(glon, glat)

    glon2, glat2 = cvt.geomag2geog(phi, theta)

    assert glon2 == approx(glon, rel=1e-6, abs=1e-8)
    assert glat2 == approx(glat, rel=1e-6, abs=1e-8)


def test_convert_numpy():

    glon = np.array([30, 45, 90, 180, 270])
    glat = np.array([-60, -45, 30, 45, 60])

    # from IGRF, not even close to function output
    # phir = np.array([0, 24.15, 17.54, 115.54, 180 ])
    # thetar = np.array([80.59, 36.34, 8.98, 41.36, 80.59])

    phi, theta = cvt.geog2geomag(glon, glat)

    # assert phi == approx(np.radians(phir))
    # assert theta == approx(np.radians(thetar))

    glon2, glat2 = cvt.geomag2geog(phi, theta)

    assert glon2 == approx(glon, rel=1e-6, abs=1e-8)
    assert glat2 == approx(glat, rel=1e-6, abs=1e-8)


@pytest.mark.parametrize(
    "name",
    [
        "mini2dew_fang",
        "mini3d_fang",
    ],
)
def test_grid(name, tmp_path, monkeypatch):

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

    write.grid(cfg, xg)

    errs = compare_grid(cfg["out_dir"], test_dir)

    assert errs == 0, f"grid mismatch {cfg['out_dir']}  {test_dir}"
