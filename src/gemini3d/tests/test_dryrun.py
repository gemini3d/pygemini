""" test dryrun, that PyGemini can correctly invoke Gemini3D """

import gemini3d.run
import gemini3d.web
from pathlib import Path

R = Path(__file__).parents[1] / "tests/data"


def test_dryrun(tmp_path):

    name = "2dew_eq"

    ref = gemini3d.web.download_and_extract(name, R)

    params = {
        "config_file": ref,
        "out_dir": tmp_path,
        "mpiexec": None,
        "gemexe": None,
        "out_format": None,
        "dryrun": True,
    }

    gemini3d.run.gemini_run(params)
