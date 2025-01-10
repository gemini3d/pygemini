""" test dryrun, that PyGemini can correctly invoke Gemini3D """

import importlib.resources as ir

import pytest
from pytest import approx

import gemini3d
import gemini3d.run
import gemini3d.job as job
import gemini3d.find as find
import gemini3d.web
import gemini3d.msis


@pytest.mark.parametrize("name,bref", [("mini2dew_eq", 1238112), ("mini3d_eq", 2323072)])
def test_memory(name, bref):
    ref = gemini3d.web.download_and_extract(name, ir.files(__package__) / "data")

    est = job.memory_estimate(ref)

    assert isinstance(est, int)

    assert est == approx(bref, abs=0, rel=0)


def test_mpiexec():
    exe = find.gemini_exe()

    mpiexec = job.check_mpiexec("mpiexec", exe)
    assert isinstance(mpiexec, str)


@pytest.mark.parametrize("name", ["mini2dew_eq"])
def test_dryrun(name, tmp_path):
    ref = gemini3d.web.download_and_extract(name, ir.files(__package__) / "data")

    params = {
        "config_file": ref,
        "out_dir": tmp_path,
        "dryrun": True,
    }

    job.runner(params)
