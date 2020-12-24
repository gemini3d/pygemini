""" test dryrun, that PyGemini can correctly invoke Gemini3D """

import pytest
from pathlib import Path

import gemini3d.run
import gemini3d.job as job
import gemini3d.web

R = Path(__file__).parents[1] / "tests/data"


@pytest.mark.parametrize("name,bref", [("2dew_eq", 1238112), ("3d_eq", 2323072)])
def test_memory(name, bref):

    try:
        ref = gemini3d.web.download_and_extract(name, R)
    except ConnectionError as e:
        pytest.skip(f"failed to download reference data {e}")

    est = job.memory_estimate(ref)

    assert est == bref


def test_mpiexec():

    exe = job.get_gemini_exe()
    assert isinstance(exe, Path)

    # It's OK if MPIexec doesn't exist, but make the test assert consistent with that
    # there are numerous possibilities that MPIexec might not work
    # predicting the outcome of this test requires the function we're testing!
    mpiexec = job.check_mpiexec("mpiexec", exe)
    assert isinstance(mpiexec, str) or mpiexec is None


@pytest.mark.parametrize("name", ["2dew_eq"])
def test_dryrun(name, tmp_path):

    try:
        ref = gemini3d.web.download_and_extract(name, R)
    except ConnectionError as e:
        pytest.skip(f"failed to download reference data {e}")

    params = {
        "config_file": ref,
        "out_dir": tmp_path,
        "dryrun": True,
    }

    job.runner(params)
