import pytest
from datetime import datetime, timedelta
import importlib.resources

import gemini3d.config as config
import gemini3d.read as read
import gemini3d.model as model


def test_model_config(tmp_path):
    P = {
        "time": [
            datetime(2020, 1, 1, 5, 1, 3),
            datetime(2020, 1, 1, 5, 1, 4),
            datetime(2020, 1, 1, 5, 1, 5),
        ],
        "dtout": 1.0,
        "f107a": 108.9,
        "f107": 111.0,
        "Ap": 5,
        "glat": 67.11,
        "glon": 212.95,
        "x2dist": 200e3,
        "x3dist": 100e3,
        "alt_min": 80e3,
        "alt_max": 1000e3,
        "alt_scale": [13.75e3, 20e3, 200e3, 200e3],
        "lx2": 40,
        "lx3": 1,
        "Bincl": 90,
        "Nmf": 5e11,
        "Nme": 2e11,
    }
    model.config(P, tmp_path)


def test_datetime_range():
    # datetime_range is a closed interval
    t = config.datetime_range(datetime(2012, 1, 1), datetime(2010, 1, 1), timedelta(seconds=1))
    assert isinstance(t, list)
    assert len(t) == 0

    t = config.datetime_range(datetime(2012, 1, 1), datetime(2012, 1, 1), timedelta(seconds=1))
    assert t == [datetime(2012, 1, 1)]

    t = config.datetime_range(datetime(2012, 1, 1), datetime(2012, 1, 1), timedelta(seconds=-1))
    assert t == [datetime(2012, 1, 1)]

    # this is how pandas.date_range works
    t = config.datetime_range(
        datetime(2012, 1, 1),
        datetime(2012, 1, 1, 0, 0, 1, microsecond=500000),
        timedelta(seconds=1),
    )
    assert t == [datetime(2012, 1, 1, 0, 0, 0), datetime(2012, 1, 1, 0, 0, 1)]

    # this is how pandas.date_range works
    t = config.datetime_range(
        datetime(2012, 1, 1, 0, 0, 0), datetime(2012, 1, 1, 0, 1, 30), timedelta(minutes=1)
    )
    assert t == [datetime(2012, 1, 1, 0, 0, 0), datetime(2012, 1, 1, 0, 1, 0)]


def test_no_config(tmp_path):
    p = read.config(tmp_path)
    assert p == {}


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
        config.parse_namelist(blank, "base")

    blank.write_text(
        """
&base
 t0 =
/
"""
    )
    with pytest.raises(KeyError):
        config.parse_namelist(blank, "base")


@pytest.mark.parametrize("group", ["base", ("base", "flags", "files", "precip", "efield")])
def test_namelist_exists(group):

    with importlib.resources.path("gemini3d.tests.config", "config_example.nml") as cfn:
        assert config.namelist_exists(cfn, "base")


@pytest.mark.parametrize("namelist", ["base", "flags", "files", "precip", "efield"])
def test_nml_namelist(namelist):

    with importlib.resources.path("gemini3d.tests.config", "config_example.nml") as cfn:
        params = config.parse_namelist(cfn, namelist)
    if "base" in namelist:
        assert params["time"][0] == datetime(2013, 2, 20, 5)

    if "files" in namelist:
        assert params["file_format"] == "h5"

    if "precip" in namelist:
        assert params["dtprec"] == timedelta(seconds=5)

    if "efield" in namelist:
        assert params["dtE0"] == timedelta(seconds=1)


def test_read_config_nml():

    with importlib.resources.path("gemini3d.tests.config", "config_example.nml") as cfn:
        params = read.config(cfn)
    assert params["time"][0] == datetime(2013, 2, 20, 5)


def test_read_config_ini():

    with importlib.resources.path("gemini3d.tests.config", "config_example.ini") as cfn:
        params = read.config(cfn)
    assert params["time"][0] == datetime(2013, 2, 20, 5)
