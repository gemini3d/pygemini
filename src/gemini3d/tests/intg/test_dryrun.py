""" test dryrun, that PyGemini can correctly invoke Gemini3D """

import shutil
import pytest
import sys
from pathlib import Path
import importlib.resources

import gemini3d
import gemini3d.run
import gemini3d.job as job
import gemini3d.web


@pytest.mark.skipif(sys.version_info < (3, 8), reason="test requires Python >= 3.8")
@pytest.mark.parametrize("name,bref", [("mini2dew_eq", 1238112), ("mini3d_eq", 2323072)])
def test_memory(name, bref):

    with importlib.resources.path("gemini3d.tests.data", "__init__.py") as fn:
        ref = gemini3d.web.download_and_extract(name, fn.parent)

    est = job.memory_estimate(ref)
    assert isinstance(est, int)

    assert est == bref


@pytest.mark.skipif(shutil.which("mpiexec") is None, reason="no Mpiexec available")
def test_mpiexec():

    gemini3d.setup()

    exe = job.get_gemini_exe()
    assert isinstance(exe, Path)

    # It's OK if MPIexec doesn't exist, but make the test assert consistent with that
    # there are numerous possibilities that MPIexec might not work
    # predicting the outcome of this test requires the function we're testing!
    mpiexec = job.check_mpiexec("mpiexec", exe)
    assert isinstance(mpiexec, str) or mpiexec is None


@pytest.mark.parametrize("name", ["mini2dew_eq"])
def test_dryrun(name, tmp_path):

    gemini3d.setup()

    with importlib.resources.path("gemini3d.tests.data", "__init__.py") as fn:
        ref = gemini3d.web.download_and_extract(name, fn.parent)

    params = {
        "config_file": ref,
        "out_dir": tmp_path,
        "dryrun": True,
    }

    job.runner(params)
