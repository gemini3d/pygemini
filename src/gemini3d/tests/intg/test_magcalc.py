import pytest
import shutil
import os
import h5py

import gemini3d.web
import gemini3d.magcalc as gm


@pytest.mark.parametrize("name", ["mini2dew_fang", "mini2dns_fang", "mini3d_fang"])
def test_dryrun(name, tmp_path, monkeypatch, helpers):
    if not os.environ.get("GEMINI_CIROOT"):
        monkeypatch.setenv("GEMINI_CIROOT", str(tmp_path / "gemini_data"))

    ref = gemini3d.web.download_and_extract(name, helpers.get_test_datadir())

    shutil.copytree(ref, tmp_path, dirs_exist_ok=True)

    Ltheta = 40
    Lphi = 30

    gm.magcalc(tmp_path, 1.5, Ltheta, Lphi)

    # %% rudimentary check of file sizes
    file = tmp_path / "inputs/magfieldpoints.h5"
    with h5py.File(file, "r") as f:
        R = f["/r"][:]

    L = Ltheta * Lphi if "3d" in name else Ltheta

    assert R.size == L, f"mismatch grid size magcalc input fieldpoints {file}"
