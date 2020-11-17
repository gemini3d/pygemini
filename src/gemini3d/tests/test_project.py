"""
these test that PyGemini generates inputs that match expectations
"""

import logging
import pytest
from pathlib import Path

import gemini3d.web
from gemini3d.fileio import write_grid
from gemini3d.grid import makegrid_cart3d
from gemini3d.compare import compare_all, compare_grid, compare_Efield, compare_precip
import gemini3d.read as read
from gemini3d.model_setup import model_setup
from gemini3d.efield import Efield_BCs
from gemini3d.particles import particles_BCs


R = Path(__file__).parents[1] / "tests/data"


@pytest.mark.parametrize("name,file_format", [("2dew_fang", "h5"), ("3d_fang", "h5")])
def test_grid(name, file_format, tmp_path):

    # get files if needed
    test_dir = gemini3d.web.download_and_extract(name, R)
    # setup new test data
    cfg = read.config(test_dir)
    xg = makegrid_cart3d(cfg)

    # path patch
    cfg["out_dir"] = tmp_path
    cfg["indat_size"] = cfg["out_dir"] / cfg["indat_size"]
    cfg["indat_grid"] = cfg["out_dir"] / cfg["indat_grid"]

    write_grid(cfg, xg)

    assert (
        compare_grid(cfg["indat_grid"], test_dir) == 0
    ), f"grid mismatch {cfg['out_dir']}  {test_dir}"


@pytest.mark.parametrize(
    "name,file_format", [("2dew_fang", "h5"), ("2dns_fang", "h5"), ("3d_fang", "h5")]
)
def test_Efield(name, file_format, tmp_path):

    # get files if needed
    test_dir = gemini3d.web.download_and_extract(name, R)

    cfg = read.config(test_dir)
    xg = read.grid(test_dir)

    # patch paths
    cfg["out_dir"] = tmp_path
    E0dir = cfg["E0dir"]
    cfg["E0dir"] = cfg["out_dir"] / cfg["E0dir"]
    Efield_BCs(cfg, xg)
    compare_Efield(
        cfg["time"], cfg["E0dir"], refdir=test_dir / E0dir, plot=False, file_format=file_format
    )


@pytest.mark.parametrize(
    "name,file_format", [("2dew_fang", "h5"), ("2dns_fang", "h5"), ("3d_fang", "h5")]
)
def test_precip(name, file_format, tmp_path):

    # get files if needed
    test_dir = gemini3d.web.download_and_extract(name, R)

    cfg = read.config(test_dir)
    xg = read.grid(test_dir)

    # patch paths
    cfg["out_dir"] = tmp_path
    precdir = cfg["precdir"]
    cfg["precdir"] = cfg["out_dir"] / cfg["precdir"]
    particles_BCs(cfg, xg)

    compare_precip(
        cfg["time"], cfg["precdir"], refdir=test_dir / precdir, plot=False, file_format=file_format
    )


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
    params = read.config(test_dir)

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
