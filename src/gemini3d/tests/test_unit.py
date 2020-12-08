"""
unit tests of non-Fortran modules
"""

from pytest import approx
import numpy as np
import math

import gemini3d.mpi as gm
import gemini3d.grid as grid
import gemini3d.coord as coord

pi = math.pi


def test_grid1d():
    x = grid.grid1d(100.0, 5)
    assert x == approx(np.arange(-100, 125, 25.0), rel=1e-6, abs=1e-8)

    x = grid.grid1d(100.0, 5, [200, 0.5, 9.5, 10])
    assert x == approx(
        [-50.25, -40.25, -30.25, -20.25, -10.25, -0.25, 0.25, 10.25, 20.25, 30.25, 40.25, 50.25],
        rel=1e-6,
        abs=1e-8,
    )


def test_max_mpi():
    assert gm.max_mpi([48, 1, 40], 5) == 5
    assert gm.max_mpi([48, 40, 1], 5) == 5
    assert gm.max_mpi([48, 1, 40], 6) == 5
    assert gm.max_mpi([48, 40, 1], 6) == 5
    assert gm.max_mpi([48, 1, 40], 8) == 8
    assert gm.max_mpi([48, 40, 1], 8) == 8
    assert gm.max_mpi([48, 1, 40], 28) == 20
    assert gm.max_mpi([48, 40, 1], 28) == 20
    assert gm.max_mpi([48, 40, 36], 28) == 18


def test_coord():
    lat, lon = coord.geomag2geog(pi / 2, pi / 2)
    assert [lat, lon] == approx([0, 19], abs=1e-6, rel=0.001)

    theta, phi = coord.geog2geomag(0, 0)
    assert [theta, phi] == approx([1.50863496978059, 1.24485046147953], abs=1e-6, rel=0.001)

    alt, lon, lat = coord.UEN2geog(0, 0, 0, pi / 2, pi / 2)
    assert [alt, lat, lon] == approx([0, 0, 19], abs=1e-6, rel=0.001)

    z, x, y = coord.geog2UEN(0, 0, 0, pi / 2, pi / 2)
    assert [z, x, y] == approx([0, -2076275.16205889, 395967.844181141], abs=1e-6, rel=0.001)
