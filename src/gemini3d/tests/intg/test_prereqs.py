"""
test prereq build
"""

import pytest
import shutil
import os

import gemini3d.prereqs as gcp


@pytest.mark.skipif(shutil.which("gfortran") is None, reason="test assumes GCC")
def test_find_gcc():
    comps = gcp.gcc_compilers()
    assert "gfortran" in comps["FC"]


@pytest.mark.skipif(shutil.which("ifort") is None, reason="test assumes Intel compilers")
def test_find_intel():
    comps = gcp.intel_compilers()
    assert "ifort" in comps["FC"]


@pytest.mark.parametrize("name", ["lapack", "scalapack"])
def test_libs(name, tmp_path):
    """test that exception isn't raised for dryrun"""
    dirs = {"prefix": tmp_path / f"install/{name}", "workdir": tmp_path / f"build/{name}"}

    if name in ("scalapack", "mumps") and not shutil.which("mpiexec"):
        pytest.skip("MPI not found")

    gcp.setup_libs(name, dirs, compiler="gcc", wipe=True, dryrun=True)


@pytest.mark.skipif(os.name != "nt", reason="these are windows-only tests")
@pytest.mark.parametrize("name", ["openmpi"])
def test_not_for_windows(name, tmp_path):

    dirs = {"prefix": tmp_path / f"install/{name}", "workdir": tmp_path / f"build/{name}"}

    with pytest.raises(EnvironmentError):
        gcp.setup_libs(name, dirs, compiler="gcc", wipe=True, dryrun=True)
