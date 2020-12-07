""" test dryrun, that PyGemini can correctly invoke Gemini3D """

import gemini3d.run
import gemini3d.job as job
import gemini3d.web
from pathlib import Path

R = Path(__file__).parents[1] / "tests/data"


def test_mpiexec():
    exe = job.get_gemini_exe()
    assert isinstance(exe, Path)

    # It's OK if MPIexec doesn't exist, but make the test assert consistent with that
    # there are numerous possibilities that MPIexec might not work
    # predicting the outcome of this test requires the function we're testing!
    mpiexec = job.check_mpiexec("mpiexec", exe)
    assert isinstance(mpiexec, str) or mpiexec is None


def test_dryrun(tmp_path):

    name = "2dew_eq"

    ref = gemini3d.web.download_and_extract(name, R)

    params = {
        "config_file": ref,
        "out_dir": tmp_path,
        "dryrun": True,
    }

    job.runner(params)
