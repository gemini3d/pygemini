"""
unit tests of non-Fortran modules
"""

import gemini3d.mpi as gm


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
