"""
because the reference data is gigabyte+, we are trying using known-working Matlab and
comparing vs. Matlab generated grid.
This also allows freely changing parameters to test behavior.

We also verify on our self-hosted CI vs. reference data files.

to setup Matlab engine:

  python pygemini/scripts/setup_matlab_engine.py
"""

from pathlib import Path
import pytest
import numpy as np
import os

import gemini3d.grid.tilted_dipole as grid


def test_tilted_dipole():
    try:
        mateng = pytest.importorskip("matlab.engine")
    except (ImportError, RuntimeError):  # can also get RuntimeError, let's just catch all
        pytest.skip("Matlab engine not available")

    # find MatGemini
    root = os.environ.get("MATGEMINI")
    if not root:
        # guess
        root = Path(__file__).parents[5] / "mat_gemini"

    root = Path(root).expanduser()
    if not root.is_dir():
        raise FileNotFoundError(
            "Please set MATGEMINI environment variable with top-level mat_gemini directory."
            f"\nMatGemini not found at {root}"
        )
    mg_setup = root / "setup.m"
    if not mg_setup.is_file():
        raise FileNotFoundError(str(mg_setup))

    eng = mateng.start_matlab("-nojvm")
    eng.addpath(str(root))

    # Inputs:  cfg dictionary with same data as provided for matlab generation
    # NOTE: Matlab Engine needs explicit floats where float is intended--put a decimal point.
    parm = {
        "lq": 4,
        "lp": 6,
        "lphi": 1,
        "dtheta": 7.5,
        "dphi": 12.0,
        "altmin": 80e3,
        "gridflag": 1,
        "glon": 143.4,
        "glat": 42.45,
    }

    # grid generated in python
    xg = grid.tilted_dipole3d(parm)

    # grid generated with MATLAB
    xg_matlab = eng.gemini3d.grid.tilted_dipole(parm)
    eng.quit()

    # # https://github.com/gemini3d/gemini-examples/tree/main/ci/daily/tohoku20112D_medres_axineu_CI
    # direc="~/sims/tohoku20112D_medres_axineu/"     # source directory
    # xgorig=gemini3d.read.grid(direc)

    # Fortran reads lx from simsize.h5 instead

    fail = []
    for k in xg.keys():
        amat = np.asarray(xg_matlab[k]).squeeze()
        if np.squeeze(xg[k]) != pytest.approx(amat, rel=1e-6):
            print(
                f"MISMATCH: {k}: python / matlab shapes:  {np.shape(xg[k])} / {np.shape(xg_matlab[k])}"
            )
            fail.append(k)
            print(np.squeeze(xg[k]))
            print(amat)

    assert not fail, f"{len(fail)} / {len(xg.keys())} keys failed: {' '.join(fail)}"
