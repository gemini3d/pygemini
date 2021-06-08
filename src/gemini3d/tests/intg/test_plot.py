import shutil
from datetime import datetime
import pytest
import importlib.resources
import sys

try:
    import matplotlib  # noqa: F401
except ImportError as e:
    pytest.skip(f"Matplotlib missing {e}", allow_module_level=True)

import gemini3d.web
import gemini3d.plot


@pytest.mark.skipif(sys.version_info < (3, 8), reason="Python >= 3.8 needed for copytree test")
@pytest.mark.parametrize(
    "name",
    [
        "mini2dew_glow",
        "mini2dns_glow",
        "mini3d_glow",
    ],
)
def test_plot(name, tmp_path):

    # get files if needed
    with importlib.resources.path("gemini3d.tests.data", "__init__.py") as fn:
        test_dir = gemini3d.web.download_and_extract(name, fn.parent)

    shutil.copytree(test_dir, tmp_path, dirs_exist_ok=True)
    gemini3d.plot.frame(tmp_path, datetime(2013, 2, 20, 5), saveplot_fmt="png")

    plot_files = sorted((tmp_path / "plots").glob("*.png"))

    assert len(plot_files) == 66
