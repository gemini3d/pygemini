"""
test prereq build
"""

import pytest
import shutil

import gemini3d.compile_prereqs as gcp

GFORTRAN = shutil.which("gfortran")


@pytest.mark.skipif(GFORTRAN is None, reason="test assumes Gfortran")
def test_find_compilers():
    comps = gcp.gcc_compilers("GNU")
    assert "gfortran" in comps["FC"]


@pytest.mark.skipif(GFORTRAN is None, reason="test assumes Gfortran")
@pytest.mark.parametrize("name", ["MPI COMPONENTS C Fortran", "LAPACK"])
def test_find_cmake(name):
    gcp.cmake_find_library(name, [], None)


@pytest.mark.skipif(GFORTRAN is None, reason="test assumes Gfortran")
@pytest.mark.parametrize("name", ["lapack", "scalapack"])
def test_libs(name, tmp_path):
    dirs = {"prefix": tmp_path / f"install/{name}", "workdir": tmp_path / f"build/{name}"}
    gcp.setup_libs(name, dirs, compiler="gcc", wipe=True, dryrun=True)
