import pytest
import os
from datetime import datetime, timedelta
from pathlib import Path

import gemini3d.config as config
import gemini3d.read as read
import gemini3d.model as model
from gemini3d.utils import get_pkg_file


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
    t = config.datetime_range(
        datetime(2012, 1, 1), datetime(2010, 1, 1), timedelta(seconds=1)
    )
    assert isinstance(t, list)
    assert len(t) == 0

    t = config.datetime_range(
        datetime(2012, 1, 1), datetime(2012, 1, 1), timedelta(seconds=1)
    )
    assert t == [datetime(2012, 1, 1)]

    t = config.datetime_range(
        datetime(2012, 1, 1), datetime(2012, 1, 1), timedelta(seconds=-1)
    )
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
        datetime(2012, 1, 1, 0, 0, 0),
        datetime(2012, 1, 1, 0, 1, 30),
        timedelta(minutes=1),
    )
    assert t == [datetime(2012, 1, 1, 0, 0, 0), datetime(2012, 1, 1, 0, 1, 0)]


def test_no_config(tmp_path):
    with pytest.raises(FileNotFoundError):
        read.config(tmp_path)


def test_no_nml(tmp_path):
    with pytest.raises(FileNotFoundError):
        config.read_nml(tmp_path)


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


@pytest.mark.parametrize("group", ["base", "flags", "files", "precip", "efield"])
def test_namelist_exists(group):
    assert config.namelist_exists(
        get_pkg_file("gemini3d.tests.config", "config_example.nml"), group
    )


def test_nml_gemini_env_root(monkeypatch, tmp_path):
    if not os.environ.get("GEMINI_CIROOT"):
        monkeypatch.setenv("GEMINI_CIROOT", str(tmp_path))

    cfg = config.parse_namelist(
        get_pkg_file("gemini3d.tests.config", "config_example.nml"), "setup"
    )

    assert isinstance(cfg["eq_dir"], Path)
    assert (
        cfg["eq_dir"] == Path(os.environ.get("GEMINI_CIROOT")).expanduser() / "test2d_eq"
    )


@pytest.mark.parametrize("namelist", ["base", "flags", "files", "precip", "efield"])
def test_nml_namelist(namelist):
    params = config.parse_namelist(
        get_pkg_file("gemini3d.tests.config", "config_example.nml"), namelist
    )

    if "base" in namelist:
        assert params["time"][0] == datetime(2013, 2, 20, 5)

    if "precip" in namelist:
        assert params["dtprec"] == timedelta(seconds=5)

    if "efield" in namelist:
        assert params["dtE0"] == timedelta(seconds=1)


@pytest.mark.parametrize("namelist", ["neutral_BG"])
def test_msis2_namelist(namelist):
    params = config.parse_namelist(
        get_pkg_file("gemini3d.tests.config", "config_msis2.nml"), namelist
    )

    if "neutral_BG" in namelist:
        msis_version = params["msis_version"]
        assert isinstance(msis_version, int)
        assert msis_version == 21


def test_read_config_nml(monkeypatch, tmp_path):
    if not os.environ.get("GEMINI_CIROOT"):
        monkeypatch.setenv("GEMINI_CIROOT", str(tmp_path))

    params = read.config(get_pkg_file("gemini3d.tests.config", "config_example.nml"))

    assert params["time"][0] == datetime(2013, 2, 20, 5)
    assert params["dtprec"] == timedelta(seconds=5)
    assert params["W0BG"] == pytest.approx(3000.0)
