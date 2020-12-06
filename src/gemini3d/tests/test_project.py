"""
these test that PyGemini generates inputs that match expectations
"""

import logging
import pytest
from pathlib import Path

import gemini3d.web
from gemini3d.compare import compare_all
from gemini3d.readdata import read_config
from gemini3d.model_setup import model_setup


R = Path(__file__).parents[1] / "tests/data"


@pytest.mark.parametrize(
    "name,file_format",
    [
        ("2dew_eq", "h5"),
        ("2dew_fang", "h5"),
        ("2dew_glow", "h5"),
        ("2dns_eq", "h5"),
        ("2dns_fang", "h5"),
        ("2dns_glow", "h5"),
        ("3d_eq", "h5"),
        ("3d_fang", "h5"),
        ("3d_glow", "h5"),
    ],
)
def test_runner(name, file_format, tmp_path):

    out_dir = tmp_path
    # get files if needed
    test_dir = gemini3d.web.download_and_extract(name, R)

    # setup new test data
    params = read_config(test_dir)

    params["format"] = file_format
    params["out_dir"] = out_dir

    for k in ("indat_file", "indat_size", "indat_grid"):
        params[k] = params[k].with_suffix("." + file_format)

    # patch eq_dir to use reference data
    if "eqdir" in params:
        eq_dir = test_dir.parent / params["eqdir"].name
        if eq_dir.is_dir():
            print(f"Using {eq_dir} for equilibrium data")
        params["eqdir"] = eq_dir

    # %% generate initial condition files
    model_setup(params, out_dir)

    # %% check generated files
    errs = compare_all(
        params["out_dir"], refdir=test_dir, only="in", plot=False, file_format=file_format
    )

    if errs:
        for e, v in errs.items():
            logging.error(f"compare:{e}: {v} errors")
        raise ValueError(f"compare_input: new generated inputs do not match reference for: {name}")
