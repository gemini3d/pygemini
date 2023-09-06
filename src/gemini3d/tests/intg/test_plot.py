import shutil
from datetime import datetime
import pytest
import os
import re

import matplotlib as mpl

import gemini3d.web
import gemini3d.plot


@pytest.mark.parametrize(
    "name",
    [
        "mini2dew_glow",
        "mini2dns_glow",
        "mini3d_glow",
    ],
)
def test_plot(name, tmp_path, monkeypatch, helpers):
    if not os.environ.get("GEMINI_CIROOT"):
        monkeypatch.setenv("GEMINI_CIROOT", str(tmp_path / "gemini_data"))

    # get files if needed
    test_dir = gemini3d.web.download_and_extract(name, helpers.get_test_datadir())

    shutil.copytree(
        test_dir, tmp_path, dirs_exist_ok=True, ignore=shutil.ignore_patterns("*.png")
    )

    fg = mpl.figure.Figure()

    gemini3d.plot.frame(fg, tmp_path, datetime(2013, 2, 20, 5), saveplot_fmt="png")

    plot_files = sorted((tmp_path / "plots").glob("*.png"))
    plot_names = [f.name for f in plot_files]

    for h in {"aurora", "J1", "J2", "J3", "v1", "v2", "v3", "ne", "Te", "Ti", "Phitop"}:
        pat = rf"^{h}-.*\.png"
        L = len(list(filter(re.compile(pat).match, plot_names)))
        assert L == 1, f"expected 1 {pat} plots under {tmp_path} but found {L} plots"
