import shutil
from datetime import datetime
import pytest
from pathlib import Path

try:
    import gemini3d.plot
except ImportError as e:
    pytest.skip(f"Matplotlib missing {e}", allow_module_level=True)

R = Path(__file__).parent / "data"


@pytest.mark.parametrize(
    "name",
    [
        "2dew_glow",
        "2dns_glow",
        "3d_glow",
    ],
)
def test_plot(name, tmp_path):

    # get files if needed
    try:
        test_dir = gemini3d.web.download_and_extract(name, R)
    except ConnectionError as e:
        pytest.skip(f"failed to download reference data {e}")

    shutil.copytree(test_dir, tmp_path)
    gemini3d.plot.frame(tmp_path, datetime(2013, 2, 20, 5), saveplot_fmt="png")

    plot_files = sorted((tmp_path / "plots").glob("*.png"))

    assert len(plot_files) == 10
