""" test dryrun, that PyGemini can correctly invoke Gemini3D """

import pytest
from pytest import approx
from pathlib import PurePosixPath
import os

import gemini3d
import gemini3d.run
import gemini3d.job as job
import gemini3d.find as find
import gemini3d.web
import gemini3d.msis


@pytest.mark.parametrize("name,bref", [("mini2dew_eq", 1238112), ("mini3d_eq", 2323072)])
def test_memory(name, bref, helpers):
    ref = gemini3d.web.download_and_extract(name, helpers.get_test_datadir())

    est = job.memory_estimate(ref)

    assert isinstance(est, int)

    assert est == approx(bref, abs=0, rel=0)


def test_mpiexec():
    exe = find.gemini_exe()

    if os.name == "nt" and isinstance(exe, PurePosixPath):
        pytest.skip("WSL check_mpiexec() not implemented")

    mpiexec = job.check_mpiexec("mpiexec", exe)
    assert isinstance(mpiexec, str)


@pytest.mark.parametrize("name", ["mini2dew_eq"])
def test_dryrun(name, tmp_path, helpers):
    ref = gemini3d.web.download_and_extract(name, helpers.get_test_datadir())

    params = {
        "config_file": ref,
        "out_dir": tmp_path,
        "dryrun": True,
    }

    job.runner(params)
