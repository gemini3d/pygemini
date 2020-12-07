"""
test msis
"""

import numpy as np
from pytest import approx
from datetime import datetime

import gemini3d.plasma as gp


def test_msis():
    cfg = {
        "time": [datetime(2015, 1, 2, 12)],
        "f107": 100.0,
        "f107a": 100.0,
        "Ap": 4,
    }

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

    atmos = gp.msis_setup(cfg, xg)
    assert atmos.shape == (7, *lx)
    assert atmos[:, 0, 1, 2] == approx(
        [
            5.8859558e17,
            9.2224699e18,
            2.2804600e18,
            1.8559000e02,
            2.0134611e11,
            2.9630136e11,
            2.3853709e13,
        ],
        rel=1e-5,
    )
