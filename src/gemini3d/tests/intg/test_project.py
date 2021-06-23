"""
these test that PyGemini generates inputs that match expectations
"""

from datetime import datetime
import logging
import importlib.resources
import pytest

import gemini3d
import gemini3d.web
import gemini3d.write as write
import gemini3d.grid
from gemini3d.compare import compare_all, compare_grid, compare_Efield, compare_precip
import gemini3d.read
import gemini3d.model
import gemini3d.find
from gemini3d.efield import Efield_BCs
from gemini3d.particles import particles_BCs


@pytest.mark.parametrize("name", ["mini2dew_fang"])
def test_file_time(name):
    # get files if needed
    with importlib.resources.path("gemini3d.tests.data", "__init__.py") as fn:
        test_dir = gemini3d.web.download_and_extract(name, fn.parent)

    t0 = datetime(2013, 2, 20, 5, 0, 0)

    file = gemini3d.find.frame(test_dir, time=t0)
    time = gemini3d.read.time(file)
    assert time == t0


@pytest.mark.parametrize(
    "name,file_format",
    [
        ("mini2dew_fang", "h5"),
        ("mini3d_fang", "h5"),
        ("mini2dew_fang", "nc"),
        ("mini3d_fang", "nc"),
    ],
)
def test_grid(name, file_format, tmp_path):

    if file_format.endswith("nc"):
        pytest.importorskip("netCDF4")

    # get files if needed
    with importlib.resources.path("gemini3d.tests.data", "__init__.py") as fn:
        test_dir = gemini3d.web.download_and_extract(name, fn.parent)

    # setup new test data
    cfg = gemini3d.read.config(test_dir)
    xg = gemini3d.grid.cartesian.cart3d(cfg)

    # path patch
    cfg["out_dir"] = tmp_path
    cfg["indat_size"] = cfg["out_dir"] / cfg["indat_size"]
    cfg["indat_grid"] = cfg["out_dir"] / cfg["indat_grid"]

    write.grid(cfg, xg, file_format=file_format)

    errs = compare_grid(cfg["out_dir"], test_dir, file_format=file_format)

    assert errs == 0, f"grid mismatch {cfg['out_dir']}  {test_dir}"


@pytest.mark.parametrize(
    "name,file_format", [("mini2dew_fang", "h5"), ("mini2dns_fang", "h5"), ("mini3d_fang", "h5")]
)
def test_Efield(name, file_format, tmp_path):

    # get files if needed
    with importlib.resources.path("gemini3d.tests.data", "__init__.py") as fn:
        test_dir = gemini3d.web.download_and_extract(name, fn.parent)

    cfg = gemini3d.read.config(test_dir)
    xg = gemini3d.read.grid(test_dir)

    # patch paths
    cfg["out_dir"] = tmp_path
    E0dir = cfg["E0dir"]
    cfg["E0dir"] = cfg["out_dir"] / cfg["E0dir"]
    Efield_BCs(cfg, xg)
    errs = compare_Efield(
        cfg["time"], cfg["E0dir"], refdir=test_dir / E0dir, plot=False, file_format=file_format
    )

    assert errs == 0, f"Efield mismatch {cfg['out_dir']}  {test_dir}"


@pytest.mark.parametrize(
    "name,file_format", [("mini2dew_fang", "h5"), ("mini2dns_fang", "h5"), ("mini3d_fang", "h5")]
)
def test_precip(name, file_format, tmp_path):

    # get files if needed
    with importlib.resources.path("gemini3d.tests.data", "__init__.py") as fn:
        test_dir = gemini3d.web.download_and_extract(name, fn.parent)

    cfg = gemini3d.read.config(test_dir)
    xg = gemini3d.read.grid(test_dir)

    # patch paths
    cfg["out_dir"] = tmp_path
    precdir = cfg["precdir"]
    cfg["precdir"] = cfg["out_dir"] / cfg["precdir"]

    particles_BCs(cfg, xg)

    errs = compare_precip(
        cfg["time"], cfg["precdir"], refdir=test_dir / precdir, plot=False, file_format=file_format
    )

    assert errs == 0, f"precipitation mismatch {cfg['out_dir']}  {test_dir}"


@pytest.mark.parametrize(
    "name,file_format",
    [
        ("mini2dew_eq", "h5"),
        ("mini2dew_fang", "h5"),
        ("mini2dew_glow", "h5"),
        ("mini2dns_eq", "h5"),
        ("mini2dns_fang", "h5"),
        ("mini2dns_glow", "h5"),
        ("mini3d_eq", "h5"),
        ("mini3d_fang", "h5"),
        ("mini3d_glow", "h5"),
    ],
)
def test_runner(name, file_format, tmp_path):

    gemini3d.setup()

    out_dir = tmp_path
    # get files if needed
    with importlib.resources.path("gemini3d.tests.data", "__init__.py") as fn:
        test_dir = gemini3d.web.download_and_extract(name, fn.parent)

    # setup new test data
    params = gemini3d.read.config(test_dir)

    params["file_format"] = file_format
    params["out_dir"] = out_dir

    for k in {"indat_file", "indat_size", "indat_grid"}:
        params[k] = params[k].with_suffix("." + file_format)

    # patch eq_dir to use reference data
    if "eq_dir" in params:
        eq_dir = test_dir.parent / params["eq_dir"].name
        if eq_dir.is_dir():
            print(f"Using {eq_dir} for equilibrium data")
        params["eq_dir"] = eq_dir

    # %% generate initial condition files
    gemini3d.model.setup(params, out_dir)

    # %% check generated files
    errs = compare_all(
        params["out_dir"], refdir=test_dir, only="in", plot=False, file_format=file_format
    )

    if errs:
        for err, v in errs.items():
            logging.error(f"compare:{err}: {v} errors")
        raise ValueError(f"compare_input: new generated inputs do not match reference for: {name}")
