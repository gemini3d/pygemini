""" test Gemini3D build separately.
This can help diagnosis if the auto-building makes lots of errors

we override env vars so that we indeed build it.
"""

import gemini3d.cmake as cmake


def test_build_msis(tmp_path, monkeypatch):

    monkeypatch.setenv("GEMINI_ROOT", "")

    tgt = tmp_path / "build/msis_setup"

    msis_exe = cmake.build_gemini3d(tgt)

    assert msis_exe.is_file()
