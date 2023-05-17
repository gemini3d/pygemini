"""
these test that PyGemini generates inputs that match expectations
"""

import pytest
import os

import gemini3d
import gemini3d.web
import gemini3d.grid
from gemini3d.compare import compare_Efield, compare_precip
import gemini3d.read
import gemini3d.model
import gemini3d.find
from gemini3d.efield import Efield_BCs
from gemini3d.particles import particles_BCs


@pytest.mark.parametrize("name", ["mini2dew_fang", "mini2dns_fang", "mini3d_fang"])
def test_Efield(name, tmp_path, monkeypatch, helpers):
    if not os.environ.get("GEMINI_CIROOT"):
        monkeypatch.setenv("GEMINI_CIROOT", str(tmp_path / "gemini_data"))

    # get files if needed
    test_dir = gemini3d.web.download_and_extract(name, helpers.get_test_datadir())

    cfg = gemini3d.read.config(test_dir)
    xg = gemini3d.read.grid(test_dir)

    # patch paths
    cfg["out_dir"] = tmp_path
    E0dir = cfg["E0dir"]
    cfg["E0dir"] = cfg["out_dir"] / cfg["E0dir"]
    Efield_BCs(cfg, xg)
    errs = compare_Efield(
        cfg,
        new_dir=cfg["E0dir"],
        ref_dir=test_dir / E0dir,
        plot=False,
    )

    assert errs == 0, f"Efield mismatch {cfg['out_dir']}  {test_dir}"


@pytest.mark.parametrize("name", ["mini2dew_fang", "mini2dns_fang", "mini3d_fang"])
def test_precip(name, tmp_path, monkeypatch, helpers):
    if not os.environ.get("GEMINI_CIROOT"):
        monkeypatch.setenv("GEMINI_CIROOT", str(tmp_path / "gemini_data"))

    # get files if needed
    test_dir = gemini3d.web.download_and_extract(name, helpers.get_test_datadir())

    cfg = gemini3d.read.config(test_dir)
    xg = gemini3d.read.grid(test_dir)

    # patch paths
    cfg["out_dir"] = tmp_path
    precdir = cfg["precdir"]
    cfg["precdir"] = cfg["out_dir"] / cfg["precdir"]

    particles_BCs(cfg, xg)

    errs = compare_precip(
        cfg,
        new_dir=cfg["precdir"],
        ref_dir=test_dir / precdir,
        plot=False,
    )

    assert errs == 0, f"precipitation mismatch {cfg['out_dir']}  {test_dir}"
