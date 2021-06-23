import importlib.resources
import pytest
from datetime import datetime
from pathlib import Path

import gemini3d
import gemini3d.find as find
import gemini3d.web


def test_config(tmp_path):
    with pytest.raises(FileNotFoundError):
        find.config(tmp_path)

    with pytest.raises(FileNotFoundError):
        find.config(tmp_path / "not_exist")

    with importlib.resources.path("gemini3d.tests.config", "config_example.nml") as cfn:
        fn = find.config(cfn)
        assert fn == cfn


@pytest.mark.parametrize("name", ["mini2dew_fang"])
def test_grid(name, tmp_path):
    with pytest.raises(FileNotFoundError):
        find.grid(tmp_path)

    with pytest.raises(FileNotFoundError):
        find.grid(tmp_path / "not_exist")

    test_dir = gemini3d.web.download_and_extract(name, gemini3d.PYGEMINI_ROOT / "tests/data")

    fn = find.grid(test_dir)
    assert fn.name.endswith("simgrid.h5")


@pytest.mark.parametrize("name", ["mini2dew_fang"])
def test_simsize(name, tmp_path):
    with pytest.raises(FileNotFoundError):
        find.simsize(tmp_path)

    with pytest.raises(FileNotFoundError):
        find.simsize(tmp_path / "not_exist")

    R = Path(gemini3d.__path__[0])

    test_dir = gemini3d.web.download_and_extract(name, R / "tests/data")

    fn = find.simsize(test_dir)
    assert fn.name == "simsize.h5"


@pytest.mark.parametrize("name", ["mini2dew_fang"])
def test_frame(name, tmp_path):

    t = datetime(2013, 2, 20, 5)

    with pytest.raises(FileNotFoundError):
        find.frame(tmp_path, t)

    with pytest.raises(FileNotFoundError):
        find.frame(tmp_path / "not_exist", t)

    R = Path(gemini3d.__path__[0])

    test_dir = gemini3d.web.download_and_extract(name, R / "tests/data")

    fn = find.frame(test_dir, t)
    assert fn.name == "20130220_18000.000000.h5"
