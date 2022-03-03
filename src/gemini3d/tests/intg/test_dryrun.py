""" test dryrun, that PyGemini can correctly invoke Gemini3D """

import shutil
import pytest
from pytest import approx
from pathlib import Path
import importlib.resources

import gemini3d
import gemini3d.run
import gemini3d.job as job
import gemini3d.web


@pytest.mark.parametrize("name,bref", [("mini2dew_eq", 1238112), ("mini3d_eq", 2323072)])
def test_memory(name, bref):

    with importlib.resources.path("gemini3d.tests.data", "__init__.py") as fn:
        ref = gemini3d.web.download_and_extract(name, fn.parent)

    est = job.memory_estimate(ref)

    assert isinstance(est, int)

    assert est == approx(bref, abs=0, rel=0)


@pytest.mark.skipif(shutil.which("mpiexec") is None, reason="no Mpiexec available")
def test_mpiexec():

    exe = job.get_gemini_exe()
    assert isinstance(exe, Path)

    mpiexec = job.check_mpiexec("mpiexec", exe)
    assert isinstance(mpiexec, str)


@pytest.mark.skipif(shutil.which("mpiexec") is None, reason="no Mpiexec available")
@pytest.mark.parametrize("name", ["mini2dew_eq"])
def test_dryrun(name, tmp_path):

    with importlib.resources.path("gemini3d.tests.data", "__init__.py") as fn:
        ref = gemini3d.web.download_and_extract(name, fn.parent)

    params = {
        "config_file": ref,
        "out_dir": tmp_path,
        "dryrun": True,
    }

    job.runner(params)
