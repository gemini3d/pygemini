"""
test msis
"""

import pytest
import numpy as np
from pytest import approx
from datetime import datetime

import gemini3d.msis as gp
import gemini3d.find as find


def test_find_msis():
    msis_exe = find.executable("msis_setup")
    print(msis_exe)


@pytest.mark.parametrize("version", [0, 21])
def test_msis(version, tmp_path):
    if version == 21:
        features = gp.get_msis_features(find.executable("msis_setup"))
        if not features["msis2"]:
            pytest.skip("MSIS2 not available")

    cfg = {
        "time": [datetime(2015, 1, 2, 12)],
        "f107": 100.0,
        "f107a": 100.0,
        "Ap": 4,
        "msis_version": version,
        "indat_size": tmp_path / "inputs/simsize.h5",
    }

    (tmp_path / "inputs").mkdir(exist_ok=True)

    lx = (4, 2, 3)

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

    assert atmos["Tn"].shape == lx

    truth = {
        0: {
            "nO": 5.8859558e17,
            "nN2": 9.2224699e18,
            "nO2": 2.2804600e18,
            "Tn": 185.59,
            "nN": 2.0134611e11,
            "nNO": 2.9630136e11,
            "nH": 2.3853709e13,
        },
        21: {
            "nO": 4.3219809e17,
            "nN2": 7.6246557e18,
            "nO2": 1.9486360e18,
            "Tn": 189.72,
            "nN": 4.4032811e10,
            "nNO": 2.1874142e11,
            "nH": 1.7495320e13,
        },
    }

    # pick an arbitrary 3D location
    a1 = atmos.isel({"alt_km": 0, "glat": 1, "glon": 2})

    for k in a1.data_vars:
        assert a1[k].data == approx(
            truth[version][k],
            rel=1e-5,
        )
    # general check of all test locations
    assert (a1["nO"].data >= 3.5e15).all() and (a1["nO"].data <= 6e17).all()
    assert (a1["nN2"].data >= 1.5e15).all() and (a1["nN2"].data <= 9.5e18).all()
    assert (a1["nO2"].data >= 1.5e14).all() and (a1["nO2"].data <= 2.5e18).all()
    assert (a1["Tn"].data >= 185).all() and (a1["Tn"].data <= 690).all()
    assert (a1["nN"].data >= 4e10).all() and (a1["nN"].data <= 2.5e13).all()
    assert (a1["nH"].data >= 5e11).all() and (a1["nH"].data <= 2.5e13).all()
