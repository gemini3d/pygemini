"""
Compile HDF5 library

Be sure environment variables are set for your desired compiler.
Use the full compiler path if it's not getting the right compiler.

* FC: Fortran compiler name or path
* CC: C compiler name or path
"""
import typing as T
import sys
import os
import subprocess
import shutil
import argparse
import tempfile
from pathlib import Path
import importlib.resources

from .utils import get_cpu_count
from .web import url_retrieve, extract_tar

# ========= user parameters ======================
BUILDDIR = "build"

NETCDF_C_TAG = "v4.7.4"
NETCDF_FORTRAN_TAG = "v4.5.3"
HDF5_TAG = "1.12/master"
MUMPS_TAG = "v5.3.3.8"
SCALAPACK_TAG = "v2.1.0.9"
LAPACK_TAG = "v3.9.0.2"

# Note: using OpenMPI 3.x because of legacy configured HPC
# that break for *any* OpenMPI 4.x app.
# https://www.open-mpi.org/software/ompi/major-changes.php
MPI_TAG = "3.1.6"

HDF5_DIR = "hdf5"
LAPACK_DIR = "lapack"
SCALAPACK_DIR = "scalapack"
MUMPS_DIR = "mumps"

# ========= end of user parameters ================

nice = ["nice"] if sys.platform == "linux" else []


def main():
    p = argparse.ArgumentParser(
        description="Compiles prerequisite libraries for Gemini (or other programs)"
    )
    p.add_argument(
        "compiler", help="compiler to build libraries for", choices=["gcc", "intel", "ibmxl"]
    )
    p.add_argument(
        "libs",
        help="libraries to compile",
        choices=["netcdf", "openmpi", "hdf5", "lapack", "scalapack", "mumps"],
        nargs="+",
    )
    p.add_argument("-prefix", help="top-level directory to install libraries under")
    p.add_argument(
        "-workdir",
        help="top-level directory to build under (can be deleted when done)",
        default=tempfile.gettempdir(),
    )
    p.add_argument("-reuse", help="reuse existing build (not usually done)", action="store_true")
    p.add_argument(
        "-n",
        "--dryrun",
        help="download code and configure but don't actually build (for testing only)",
        action="store_true",
    )
    P = p.parse_args()

    prefix = P.prefix if P.prefix else f"~/lib_{P.compiler}"

    dirs = {
        "prefix": Path(prefix).expanduser().resolve(),
        "workdir": Path(P.workdir).expanduser().resolve(),
    }

    setup_libs(P.libs, dirs, P.compiler, wipe=not P.reuse, dryrun=P.dryrun)


def setup_libs(
    libs: T.Sequence[str],
    dirs: T.Dict[str, Path],
    compiler: str,
    *,
    wipe: bool,
    dryrun: bool = False,
):

    if compiler == "gcc":
        env = gcc_compilers()
    elif compiler == "intel":
        env = intel_compilers()
    elif compiler == "ibmxl":
        env = ibmxl_compilers()
    else:
        raise ValueError(f"unknown compiler {compiler}")

    if "hdf5" in libs:
        hdf5(dirs, env=env)
    if "netcdf" in libs:
        netcdf_c(dirs, env=env, wipe=wipe, dryrun=dryrun)
        netcdf_fortran(dirs, env=env, wipe=wipe, dryrun=dryrun)

    # Note: OpenMPI needs to be before scalapack and mumps
    if "openmpi" in libs:
        openmpi(dirs, env=env, dryrun=dryrun)
    if "lapack" in libs:
        lapack(wipe, dirs, env=env, dryrun=dryrun)
    if "scalapack" in libs:
        scalapack(wipe, dirs, env=env, dryrun=dryrun)
    if "mumps" in libs:
        mumps(wipe, dirs, env=env, dryrun=dryrun)

    if not dryrun:
        print("Installed", libs, "under", dirs["prefix"])


def netcdf_c(
    dirs: T.Dict[str, Path], env: T.Mapping[str, str], wipe: bool = False, dryrun: bool = False
):
    """ build and install NetCDF-C
    """

    install_dir = dirs["prefix"] / "netcdf"
    source_dir = dirs["workdir"] / "netcdf-c"
    build_dir = source_dir / BUILDDIR

    git_url = "https://github.com/Unidata/netcdf-c.git"

    git_download(source_dir, git_url, NETCDF_C_TAG)

    hdf5_root = dirs["prefix"] / HDF5_DIR
    if hdf5_root.is_dir():
        lib_args = [f"-DHDF5_ROOT={hdf5_root.as_posix()}"]
    else:
        lib_args = []

    if not cmake_find_library("HDF5 COMPONENTS C Fortran", lib_args, env):
        raise RuntimeError("Please install HDF5 before NetCDF4")

    c_args = [
        f"-DCMAKE_INSTALL_PREFIX:PATH={install_dir}",
        "-DCMAKE_BUILD_TYPE:STRING=Release",
        "-DBUILD_SHARED_LIBS:BOOL=ON",
        "-DENABLE_PARALLEL4:BOOL=OFF",
        "-DENABLE_PNETCDF:BOOL=OFF",
        "-DBUILD_UTILITIES:BOOL=OFF",
        "-DENABLE_TESTS:BOOL=off",
        "-DBUILD_TESTING:BOOL=OFF",
        "-DENABLE_HDF4:BOOL=OFF",
        "-DUSE_DAP:BOOL=off",
        "-DENABLE_DAP:BOOL=OFF",
        "-DENABLE_DAP2:BOOL=OFF",
        "-DENABLE_DAP4:BOOL=OFF",
    ]
    cmake_build(
        c_args + lib_args, source_dir, build_dir, wipe, env=env, run_test=False, dryrun=dryrun
    )


def netcdf_fortran(
    dirs: T.Dict[str, Path], env: T.Mapping[str, str], wipe: bool = False, dryrun: bool = False
):
    """ build and install NetCDF-Fortran
    """

    install_dir = dirs["prefix"] / "netcdf"
    source_dir = dirs["workdir"] / "netcdf-fortran"
    build_dir = source_dir / BUILDDIR

    git_url = "https://github.com/Unidata/netcdf-fortran.git"

    git_download(source_dir, git_url, NETCDF_FORTRAN_TAG)

    # NetCDF-Fortran does not yet use NetCDF_ROOT
    if sys.platform == "linux":
        netcdf_c = install_dir / "lib/libnetcdf.so"
    elif sys.platform == "win32":
        print(
            "NetCDF4 on MSYS2 may not work, see https://github.com/Unidata/netcdf-c/issues/554",
            file=sys.stderr,
        )
        netcdf_c = install_dir / "bin/libnetcdf.dll"
    elif sys.platform == "darwin":
        netcdf_c = install_dir / "lib/libnetcdf.dylib"
    else:
        raise NotImplementedError(
            f"please open a GitHub Issue for your operating system {sys.platform}"
        )

    hdf5_root = dirs["prefix"] / HDF5_DIR
    if hdf5_root.is_dir():
        lib_args = [f"-DHDF5_ROOT={hdf5_root.as_posix()}"]
    else:
        lib_args = []

    if not cmake_find_library("HDF5 COMPONENTS C Fortran", lib_args, env):
        raise RuntimeError("Please install HDF5 before NetCDF4")

    patch = [
        f"-DNETCDF_C_LIBRARY:FILEPATH={netcdf_c}",
        f"-DNETCDF_INCLUDE_DIR:PATH={install_dir / 'include'}",
    ]
    f_args = patch + [
        f"-DNetCDF_ROOT:PATH={install_dir}",
        f"-DCMAKE_INSTALL_PREFIX:PATH={install_dir}",
        "-DCMAKE_BUILD_TYPE:STRING=Release",
        "-DBUILD_SHARED_LIBS:BOOL=ON",
        "-DBUILD_UTILITIES:BOOL=OFF",
        "-DENABLE_TESTS:BOOL=off",
        "-DBUILD_EXAMPLES:BOOL=OFF",
    ]
    cmake_build(f_args, source_dir, build_dir, wipe, env=env, run_test=False, dryrun=dryrun)


def hdf5(dirs: T.Dict[str, Path], env: T.Mapping[str, str]):
    """ build and install HDF5
    some systems have broken libz and so have trouble extracting tar.bz2 from Python.
    To avoid this, we git clone the release instead.
    """

    name = "hdf5"
    install_dir = dirs["prefix"] / name
    source_dir = dirs["workdir"] / name

    if os.name == "nt":
        if "ifort" in env["FC"]:
            msg = """
For Windows with Intel compiler, use HDF5 binaries from HDF Group.
https://www.hdfgroup.org/downloads/hdf5/
look for filename like hdf5-1.12.0-Std-win10_64-vs14-Intel.zip
            """
            raise NotImplementedError(msg)

        cmd0 = [
            "cmake",
            f"-S{source_dir}",
            f"-B{source_dir/BUILDDIR}",
            f"-DCMAKE_INSTALL_PREFIX={install_dir}",
            "-DBUILD_SHARED_LIBS:BOOL=false",
            "-DCMAKE_BUILD_TYPE=Release",
            "-DHDF5_BUILD_FORTRAN:BOOL=true",
            "-DHDF5_BUILD_CPP_LIB:BOOL=false",
            "-DHDF5_BUILD_TOOLS:BOOL=false",
            "-DBUILD_TESTING:BOOL=false",
            "-DHDF5_BUILD_EXAMPLES:BOOL=false",
        ]
        cmd1 = ["cmake", "--build", BUILDDIR, "--parallel"]
        cmd2 = ["cmake", "--install", BUILDDIR, "--parallel"]
    else:
        cmd0 = [
            "./configure",
            f"--prefix={install_dir}",
            "--enable-fortran",
            "--enable-build-mode=production",
        ]
        cmd1 = ["make", "-j"]
        cmd2 = ["make", "-j", "install"]

    git_url = "https://bitbucket.hdfgroup.org/scm/hdffv/hdf5.git"

    git_download(source_dir, git_url, HDF5_TAG)

    subprocess.check_call(nice + cmd0, cwd=source_dir, env=env)
    subprocess.check_call(nice + cmd1, cwd=source_dir)
    subprocess.check_call(nice + cmd2, cwd=source_dir)


def openmpi(dirs: T.Dict[str, Path], env: T.Mapping[str, str], dryrun: bool = False):
    """ build and install OpenMPI """
    if os.name == "nt":
        raise NotImplementedError(
            """
OpenMPI is not available in native Windows.
Other options on Windows:
* Windows Subsystem for Linux
* MS-MPI with MSYS2: https://www.scivision.dev/windows-mpi-msys2/
* Intel oneAPI with IntelMPI: https://www.scivision.dev/intel-oneapi-fortran-install/
* Cygwin
"""
        )

    mpi_dir = f"openmpi-{MPI_TAG}"
    install_dir = dirs["prefix"] / mpi_dir
    source_dir = dirs["workdir"] / mpi_dir

    tar_name = f"openmpi-{MPI_TAG}.tar.bz2"
    tarfn = dirs["workdir"] / tar_name
    url = f"https://download.open-mpi.org/release/open-mpi/v{MPI_TAG[:3]}/{tar_name}"
    url_retrieve(url, tarfn)
    extract_tar(tarfn, source_dir)

    cmd = [
        "./configure",
        f"--prefix={install_dir}",
        f"CC={env['CC']}",
        f"CXX={env['CXX']}",
        f"FC={env['FC']}",
    ]

    subprocess.check_call(nice + cmd, cwd=source_dir, env=env)

    Njobs = get_cpu_count()

    cmd = ["make", "-C", str(source_dir), "-j", str(Njobs), "install"]

    if dryrun:
        print("DRYRUN: would have run\n", " ".join(cmd))
        return None

    subprocess.check_call(nice + cmd)


def lapack(wipe: bool, dirs: T.Dict[str, Path], env: T.Mapping[str, str], dryrun: bool = False):
    """ build and insall Lapack """
    install_dir = dirs["prefix"] / LAPACK_DIR
    source_dir = dirs["workdir"] / LAPACK_DIR
    build_dir = source_dir / BUILDDIR

    git_url = "https://github.com/scivision/lapack.git"

    git_download(source_dir, git_url, LAPACK_TAG)

    args = ["-Dautobuild:BOOL=off", f"-DCMAKE_INSTALL_PREFIX:PATH={install_dir}"]
    cmake_build(args, source_dir, build_dir, wipe, env=env, dryrun=dryrun)


def scalapack(wipe: bool, dirs: T.Dict[str, Path], env: T.Mapping[str, str], dryrun: bool = False):
    """ build and install Scalapack """
    install_dir = dirs["prefix"] / SCALAPACK_DIR
    source_dir = dirs["workdir"] / SCALAPACK_DIR
    build_dir = source_dir / BUILDDIR

    git_url = "https://github.com/scivision/scalapack.git"

    git_download(source_dir, git_url, SCALAPACK_TAG)

    lapack_root = dirs["prefix"] / LAPACK_DIR
    lib_args = [f"-DLAPACK_ROOT={lapack_root.as_posix()}"]

    if not cmake_find_library("LAPACK", lib_args, env):
        lapack(wipe, dirs, env)

    args = [
        "-Dautobuild:BOOL=off",
        f"-DCMAKE_INSTALL_PREFIX:PATH={install_dir}",
    ]
    cmake_build(args + lib_args, source_dir, build_dir, wipe, env=env, dryrun=dryrun)


def mumps(wipe: bool, dirs: T.Dict[str, Path], env: T.Mapping[str, str], dryrun: bool = False):
    """ build and install Mumps """
    install_dir = dirs["prefix"] / MUMPS_DIR
    source_dir = dirs["workdir"] / MUMPS_DIR
    build_dir = source_dir / BUILDDIR

    scalapack_lib = dirs["prefix"] / SCALAPACK_DIR
    lapack_lib = dirs["prefix"] / LAPACK_DIR

    git_url = "https://github.com/scivision/mumps.git"

    git_download(source_dir, git_url, MUMPS_TAG)

    if env["FC"] == "ifort":
        lib_args = []
    else:
        lib_args = [
            f"-DSCALAPACK_ROOT:PATH={scalapack_lib.as_posix()}",
            f"-DLAPACK_ROOT:PATH={lapack_lib.as_posix()}",
        ]

    if not cmake_find_library("LAPACK", lib_args, env):
        lapack(wipe, dirs, env)
    if not cmake_find_library("SCALAPACK", lib_args, env):
        scalapack(wipe, dirs, env)

    args = ["-Dautobuild:BOOL=off", f"-DCMAKE_INSTALL_PREFIX:PATH={install_dir}"]
    cmake_build(args + lib_args, source_dir, build_dir, wipe, env=env, dryrun=dryrun)


def cmake_build(
    args: T.List[str],
    source_dir: Path,
    build_dir: Path,
    wipe: bool,
    env: T.Mapping[str, str],
    run_test: bool = True,
    dryrun: bool = False,
):
    """ build and install with CMake """
    cmake = get_cmake()

    cache_file = build_dir / "CMakeCache.txt"
    if wipe:
        if cache_file.is_file():
            cache_file.unlink()

    cmd = nice + [cmake] + args + ["-B", str(build_dir), "-S", str(source_dir)]
    subprocess.check_call(cmd, env=env)

    cmd = nice + [cmake, "--build", str(build_dir), "--parallel"]
    if dryrun:
        print("DRYRUN: would have run\n", " ".join(cmd))
        return None

    subprocess.check_call(cmd)

    Njobs = get_cpu_count()

    if run_test:
        subprocess.check_call(
            nice + ["ctest", "--parallel", str(Njobs), "--output-on-failure"], cwd=str(build_dir)
        )

    subprocess.check_call(nice + [cmake, "--install", str(build_dir)])


def cmake_find_library(lib_name: str, lib_path: T.List[str], env: T.Mapping[str, str]) -> bool:
    """
    check if library exists with CMake

    lib_name must have the appropriate upper and lower case letter as would be used
    directly in CMake.
    """

    cmake = get_cmake()

    if __file__:
        mod_path = Path(__file__).parent / "cmake"
    else:
        with importlib.resources.path("gemini3d.cmake", "FindLAPACK.cmake") as f:
            mod_path = Path(f).parent

    cmake_template = """
cmake_minimum_required(VERSION 3.15)
project(dummy LANGUAGES C Fortran)

"""

    if mod_path.is_dir():
        mod_str = mod_path.as_posix()
        cmake_template += f'list(APPEND CMAKE_MODULE_PATH "{mod_str}")\n'

    cmake_template += f"find_package({lib_name} REQUIRED)\n"

    build_dir = f"find-{lib_name.split(' ', 1)[0]}"

    # not context_manager to avoid Windows PermissionError on context exit for Git subdirs
    d = tempfile.TemporaryDirectory()
    r = Path(d.name)
    (r / "CMakeLists.txt").write_text(cmake_template)

    cmd = [cmake, "-S", str(r), "-B", str(r / build_dir)] + lib_path
    # use cwd= to avoid spilling temporary files into current directory if ancient CMake used
    # also avoids bugs if there is a CMakeLists.txt in the current directory
    ret = subprocess.run(cmd, env=env, cwd=str(r))

    try:
        d.cleanup()
    except PermissionError:
        pass

    return ret.returncode == 0


def get_cmake() -> str:

    cmake = shutil.which("cmake")
    if not cmake:
        raise FileNotFoundError("CMake not found.")

    cmake_version = (
        subprocess.check_output([cmake, "--version"], universal_newlines=True)
        .split("\n")[0]
        .split(" ")[2]
    )

    print("Using CMake", cmake_version)

    return cmake


def git_download(path: Path, repo: str, tag: str):
    """
    Use Git to download code repo.
    """
    GITEXE = shutil.which("git")

    if not GITEXE:
        raise FileNotFoundError("Git not found.")

    git_version = (
        subprocess.check_output([GITEXE, "--version"], universal_newlines=True).strip().split()[-1]
    )
    print("Using Git", git_version)

    if path.is_dir():
        # don't use "git -C" for old HPC
        ret = subprocess.run([GITEXE, "checkout", tag], cwd=str(path))
        if ret.returncode != 0:
            ret = subprocess.run([GITEXE, "fetch"], cwd=str(path))
            if ret.returncode != 0:
                raise RuntimeError(f"could not fetch {path}  Maybe try removing this directory.")
            subprocess.check_call([GITEXE, "checkout", tag], cwd=str(path))
    else:
        # shallow clone
        if tag:
            subprocess.check_call(
                [GITEXE, "clone", repo, "--branch", tag, "--single-branch", str(path)]
            )
        else:
            subprocess.check_call([GITEXE, "clone", repo, "--depth", "1", str(path)])


def get_compilers(compiler_name: str, **kwargs) -> T.Mapping[str, str]:
    """ get paths to compilers

    Parameters
    ----------

    compiler_name: str
        arbitrary string naming compiler--to give useful error message when compiler not found.
    """
    env = os.environ

    for k, v in kwargs.items():
        c = env.get(k, "")
        if v not in c:
            c = shutil.which(v)
        if not c:
            raise FileNotFoundError(
                f"Compiler {compiler_name} was not found: {k}."
                " Did you load the compiler shell environment first?"
            )
        env.update({k: c})

    return env


def gcc_compilers() -> T.Mapping[str, str]:
    return get_compilers("GNU", FC="gfortran", CC="gcc", CXX="g++")


def intel_compilers() -> T.Mapping[str, str]:
    return get_compilers(
        "Intel",
        FC="ifort",
        CC="icl" if os.name == "nt" else "icc",
        CXX="icl" if os.name == "nt" else "icpc",
    )


def ibmxl_compilers() -> T.Mapping[str, str]:
    return get_compilers("IBM XL", FC="xlf", CC="xlc", CXX="xlc++")
