"""
these test that PyGemini generates inputs that match expectations
"""

from pytest import approx

import numpy as np
from datetime import datetime
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
def test_file_time(name, helpers):
    # get files if needed
    test_dir = gemini3d.web.download_and_extract(name, helpers.get_test_datadir())

    t0 = datetime(2013, 2, 20, 5, 0, 0)

    file = gemini3d.find.frame(test_dir, time=t0)
    time = gemini3d.read.time(file)
    assert time == t0


def test_rotvec_gg2gm():
    x = np.linspace(-np.pi, np.pi, 2 * 3 * 4 * 3)
    x = x.reshape((2, 3, 4, 3))

    yt = np.array(
        [
            [
                [
                    [2.39538397, -3.96442539, -2.55447358],
                    [2.18316209, -3.62696877, -2.3252696],
                    [1.9709402, -3.28951214, -2.09606562],
                    [1.75871831, -2.95205552, -1.86686164],
                ],
                [
                    [1.54649642, -2.61459889, -1.63765766],
                    [1.33427453, -2.27714227, -1.40845368],
                    [1.12205264, -1.93968564, -1.1792497],
                    [0.90983075, -1.60222902, -0.95004572],
                ],
                [
                    [0.69760887, -1.26477239, -0.72084174],
                    [0.48538698, -0.92731577, -0.49163776],
                    [0.27316509, -0.58985914, -0.26243378],
                    [0.0609432, -0.25240252, -0.0332298],
                ],
            ],
            [
                [
                    [-0.15127869, 0.08505411, 0.19597418],
                    [-0.36350058, 0.42251073, 0.42517816],
                    [-0.57572246, 0.75996736, 0.65438214],
                    [-0.78794435, 1.09742399, 0.88358612],
                ],
                [
                    [-1.00016624, 1.43488061, 1.1127901],
                    [-1.21238813, 1.77233724, 1.34199408],
                    [-1.42461002, 2.10979386, 1.57119807],
                    [-1.63683191, 2.44725049, 1.80040205],
                ],
                [
                    [-1.8490538, 2.78470711, 2.02960603],
                    [-2.06127568, 3.12216374, 2.25881001],
                    [-2.27349757, 3.45962036, 2.48801399],
                    [-2.48571946, 3.79707699, 2.71721797],
                ],
            ],
        ]
    )

    assert cvt.rotvec_gg2gm(x) == approx(yt)


def test_rotvec_gg2gm_points():
    x = np.array([[1, 2, 3], [-3, -2, 1]])
    yt = np.array(
        [[-2.10913391, 1.59665488, 2.64617598], [0.70672483, -3.48769204, 1.15609009]]
    )

    assert cvt.rotvec_gg2gm_points(x) == approx(yt)


@pytest.mark.parametrize(
    "glon, glat", [(30, -60), (45, -45), (90, 30), (180, 45), (270, 60)]
)
def test_convert_scalar(glon, glat):
    phi, theta = cvt.geog2geomag(glon, glat)

    glon2, glat2 = cvt.geomag2geog(phi, theta)

    assert glon2 == approx(glon, rel=1e-6, abs=1e-8)
    assert glat2 == approx(glat, rel=1e-6, abs=1e-8)


def test_convert_numpy():
    glon = np.array((30, 45, 90, 180, 270))
    glat = np.array((-60, -45, 30, 45, 60))

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
def test_grid(name, tmp_path, monkeypatch, helpers):
    if not os.environ.get("GEMINI_CIROOT"):
        monkeypatch.setenv("GEMINI_CIROOT", str(tmp_path / "gemini_data"))

    # get files if needed
    test_dir = gemini3d.web.download_and_extract(name, helpers.get_test_datadir())

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
