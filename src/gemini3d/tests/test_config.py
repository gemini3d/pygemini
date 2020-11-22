import pytest
from datetime import datetime, timedelta
import importlib.resources

import gemini3d.config as config


def test_no_nml(tmp_path):
    p = config.read_nml(tmp_path)
    assert p == {}


def test_no_ini(tmp_path):
    p = config.read_ini(tmp_path)
    assert p == {}


def test_nml_bad(tmp_path):
    blank = tmp_path / "foo"
    blank.touch()
    with pytest.raises(KeyError):
        config.read_namelist(blank, "base")

    blank.write_text(
        """
&base
 t0 =
/
"""
    )
    with pytest.raises(KeyError):
        config.read_namelist(blank, "base")


@pytest.mark.parametrize("group", ["base", ("base", "flags", "files", "precip", "efield")])
def test_namelist_exists(group):

    with importlib.resources.path("gemini3d.tests.config", "config_example.nml") as cfn:
        assert config.namelist_exists(cfn, "base")


@pytest.mark.parametrize("namelist", ["base", "flags", "files", "precip", "efield"])
def test_nml_namelist(namelist):

    with importlib.resources.path("gemini3d.tests.config", "config_example.nml") as cfn:
        params = config.read_namelist(cfn, namelist)
    if "base" in namelist:
        assert params["t0"] == datetime(2013, 2, 20, 5)

    if "files" in namelist:
        assert params["format"] == "h5"

    if "precip" in namelist:
        assert params["dtprec"] == timedelta(seconds=5)

    if "efield" in namelist:
        assert params["dtE0"] == timedelta(seconds=1)


def test_read_config_nml():

    with importlib.resources.path("gemini3d.tests.config", "config_example.nml") as cfn:
        params = config.read_config(cfn)
    assert params["t0"] == datetime(2013, 2, 20, 5)


def test_read_config_ini():

    with importlib.resources.path("gemini3d.tests.config", "config_example.ini") as cfn:
        params = config.read_config(cfn)
    assert params["t0"] == datetime(2013, 2, 20, 5)
