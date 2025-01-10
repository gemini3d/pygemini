import pytest

import gemini3d.wsl as wsl


@pytest.mark.skipif(not wsl.wsl_available(), reason="WSL not available")
def test_wsl_path():
    wsl_temp = wsl.wsl_tempfile()
    assert wsl_temp, "could not get WSL tempfile"

    win_path = wsl.wsl_path2win_path(wsl_temp)
    assert wsl.is_wsl_path(win_path), "could not convert WSL path to Windows path"

    wsl_path = wsl.win_path2wsl_path(win_path)
    assert wsl_path == wsl_temp, "could not convert Windows path to WSL path"
