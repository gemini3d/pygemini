"""
test prereq build
"""

import pytest
import shutil
import os

import gemini3d.prereqs as gcp
import gemini3d.cmake as cmake


@pytest.mark.skipif(shutil.which("gfortran") is None, reason="test assumes GCC")
def test_find_gcc():
    comps = gcp.gcc_compilers()
    assert "gfortran" in comps["FC"]


@pytest.mark.skipif(shutil.which("ifort") is None, reason="test assumes Intel compilers")
def test_find_intel():
    comps = gcp.intel_compilers()
    assert "ifort" in comps["FC"]


@pytest.mark.parametrize("name", ["MPI COMPONENTS C Fortran", "LAPACK"])
def test_find_cmake(name):

    assert cmake.find_library(name, [], None), f"{name} not found"


def test_cmake_notfound():

    assert not cmake.find_library("abc123zxyb", [], None), "should not be found"


@pytest.mark.parametrize("name", ["lapack", "scalapack"])
def test_libs(name, tmp_path):
    """ test that exception isn't raised for dryrun """
    dirs = {"prefix": tmp_path / f"install/{name}", "workdir": tmp_path / f"build/{name}"}

    gcp.setup_libs(name, dirs, compiler="gcc", wipe=True, dryrun=True)


@pytest.mark.skipif(os.name != "nt", reason="these are windows-only tests")
@pytest.mark.parametrize("name", ["openmpi"])
def test_not_for_windows(name, tmp_path):

    dirs = {"prefix": tmp_path / f"install/{name}", "workdir": tmp_path / f"build/{name}"}

    with pytest.raises(NotImplementedError):
        gcp.setup_libs(name, dirs, compiler="gcc", wipe=True, dryrun=True)
