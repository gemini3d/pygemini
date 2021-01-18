import importlib.resources
import pytest
from datetime import datetime
from pathlib import Path

import gemini3d
import gemini3d.find as find
import gemini3d.web


def test_config(tmp_path):
    assert find.config(tmp_path) is None
    assert find.config(tmp_path / "not_exist") is None

    with importlib.resources.path("gemini3d.tests.config", "config_example.nml") as cfn:
        fn = find.config(cfn)
        assert fn == cfn


@pytest.mark.parametrize("name", ["2dew_fang"])
def test_grid(name, tmp_path):
    assert find.grid(tmp_path) is None
    assert find.grid(tmp_path / "not_exist") is None

    try:
        test_dir = gemini3d.web.download_and_extract(name, gemini3d.PYGEMINI_ROOT / "tests/data")
    except ConnectionError as e:
        pytest.skip(f"failed to download reference data {e}")

    fn = find.grid(test_dir)
    assert fn.name.endswith("simgrid.h5")


@pytest.mark.parametrize("name", ["2dew_fang"])
def test_simsize(name, tmp_path):
    assert find.simsize(tmp_path) is None
    assert find.simsize(tmp_path / "not_exist") is None

    R = Path(gemini3d.__path__[0])

    try:
        test_dir = gemini3d.web.download_and_extract(name, R / "tests/data")
    except ConnectionError as e:
        pytest.skip(f"failed to download reference data {e}")

    fn = find.simsize(test_dir)
    assert fn.name == "simsize.h5"


@pytest.mark.parametrize("name", ["2dew_fang"])
def test_frame(name, tmp_path):

    t = datetime(2013, 2, 20, 5)

    assert find.frame(tmp_path, t) is None
    assert find.frame(tmp_path / "not_exist", t) is None

    R = Path(gemini3d.__path__[0])

    try:
        test_dir = gemini3d.web.download_and_extract(name, R / "tests/data")
    except ConnectionError as e:
        pytest.skip(f"failed to download reference data {e}")

    fn = find.frame(test_dir, t)
    assert fn.name == "20130220_18000.000001.h5"
