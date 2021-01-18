"""
test msis
"""

import pytest
import numpy as np
from pytest import approx
from datetime import datetime

import gemini3d.msis as gp


@pytest.mark.parametrize("version", [0, 20])
def test_msis(version, tmp_path):
    cfg = {
        "time": [datetime(2015, 1, 2, 12)],
        "f107": 100.0,
        "f107a": 100.0,
        "Ap": 4,
        "msis_version": version,
        "indat_size": tmp_path / "inputs/simsize.h5",
    }

    (tmp_path / "inputs").mkdir(exist_ok=True)

    lx = [4, 2, 3]

    glon, alt, glat = np.meshgrid(
        np.linspace(-147, -145, lx[1]),
        np.linspace(100e3, 200e3, lx[0]),
        np.linspace(65, 66, lx[2]),
    )
    xg = {
        "glat": glat,
        "glon": glon,
        "lx": lx,
        "alt": alt,
    }

    try:
        atmos = gp.msis_setup(cfg, xg)
    except RuntimeError as e:
        if "-Dmsis20=" in str(e):
            pytest.skip("MSIS 2.0 wasn't available")
        raise

    assert atmos.shape == (7, *lx)

    truth = {
        0: [
            5.8859558e17,
            9.2224699e18,
            2.2804600e18,
            1.8559000e02,
            2.0134611e11,
            2.9630136e11,
            2.3853709e13,
        ],
        20: [
            4.3219809e17,
            7.6246557e18,
            1.9486360e18,
            1.8972000e02,
            4.4032811e10,
            2.1874142e11,
            1.7495320e13,
        ],
    }

    assert atmos[:, 0, 1, 2] == approx(
        truth[version],
        rel=1e-5,
    )
