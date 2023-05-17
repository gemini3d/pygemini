"""
these test that PyGemini generates inputs that match expectations
"""

import logging
import os
from pathlib import Path
import h5py

import pytest

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
def test_model_setup(name, tmp_path, monkeypatch, helpers):
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
        raise ValueError(
            f"compare_input: new generated inputs do not match reference for: {name}"
        )


@pytest.mark.parametrize("name", ["tohoku2d_eq", "tohoku3d_eq"])
@pytest.mark.skipif(
    not os.environ.get("GEMCI_ROOT") or not Path(os.environ["GEMCI_ROOT"]).is_dir(),
    reason="GEMCI_ROOT not set",
)
def test_equilibrium_setup(name, tmp_path, monkeypatch):
    cfg_dir = Path(os.environ["GEMCI_ROOT"]) / "cfg/equilibrium" / name

    if not os.environ.get("GEMINI_CIROOT"):
        monkeypatch.setenv("GEMINI_CIROOT", str(tmp_path / "gemini_data"))

    params = gemini3d.read.config(cfg_dir)

    out_dir = tmp_path
    print("generating equilibrium files in", out_dir)
    gemini3d.model.setup(params, out_dir)

    assert (out_dir / "inputs").is_dir()
    for n in (
        "config.nml",
        "initial_conditions.h5",
        "msis_setup_in.h5",
        "msis_setup_out.h5",
        "setup_grid.json",
        "simgrid.h5",
        "simsize.h5",
    ):
        assert (out_dir / "inputs" / n).is_file()

    with h5py.File(out_dir / "inputs/simsize.h5", "r") as f:
        lx = f["/lx"][:]
    assert lx[0] == params["lq"]
    assert lx[1] == params["lp"]
    assert lx[2] == params["lphi"]
