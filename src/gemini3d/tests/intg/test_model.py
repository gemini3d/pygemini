"""
these test that PyGemini generates inputs that match expectations
"""

import logging
import pytest
import os

import gemini3d
import gemini3d.web
import gemini3d.grid
from gemini3d.compare import compare_all
import gemini3d.read
import gemini3d.model
import gemini3d.find
import gemini3d.msis


@pytest.mark.parametrize(
    "name",
    [
        "mini2dew_eq",
        "mini2dew_fang",
        "mini2dew_glow",
        "mini2dns_eq",
        "mini2dns_fang",
        "mini2dns_glow",
        "mini3d_eq",
        "mini3d_fang",
        "mini3d_glow",
    ],
)
def test_runner(name, tmp_path, monkeypatch, helpers):

    if not os.environ.get("GEMINI_CIROOT"):
        monkeypatch.setenv("GEMINI_CIROOT", str(tmp_path / "gemini_data"))

    out_dir = tmp_path
    # get files if needed
    test_dir = gemini3d.web.download_and_extract(name, helpers.get_test_datadir())

    # setup new test data
    params = gemini3d.read.config(test_dir)

    params["out_dir"] = out_dir

    # patch eq_dir to use reference data
    if "eq_dir" in params:
        eq_dir = test_dir.parent / params["eq_dir"].name
        if eq_dir.is_dir():
            print(f"Using {eq_dir} for equilibrium data")
        params["eq_dir"] = eq_dir

    # %% generate initial condition files
    gemini3d.model.setup(params, out_dir)

    # %% check generated files
    errs = compare_all(params["out_dir"], ref_dir=test_dir, only="in", plot=False)

    if errs:
        for err, v in errs.items():
            logging.error(f"compare:{err}: {v} errors")
        raise ValueError(f"compare_input: new generated inputs do not match reference for: {name}")
