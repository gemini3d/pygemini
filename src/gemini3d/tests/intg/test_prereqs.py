"""
test prereq build
"""

import pytest
import shutil

import gemini3d.prereqs as gcp


@pytest.mark.skipif(shutil.which("gfortran") is None, reason="test assumes GCC")
def test_find_gcc():
    comps = gcp.gcc_compilers()
    assert "gfortran" in comps["FC"]


@pytest.mark.skipif(shutil.which("ifort") is None, reason="test assumes Intel compilers")
def test_find_intel():
    comps = gcp.intel_compilers()
    assert "ifort" in comps["FC"]


@pytest.mark.skipif(shutil.which("mpiexec") is None, reason="no Mpiexec available")
@pytest.mark.parametrize("name", ["lapack", "scalapack"])
def test_libs(name, tmp_path):
    """test that exception isn't raised for dryrun
    skip on mpiexec because that means other stuff is missing too.
    """
    dirs = {"prefix": tmp_path / f"install/{name}", "workdir": tmp_path / f"build/{name}"}

    gcp.setup_libs(name, dirs, compiler="gcc", wipe=True, dryrun=True)
