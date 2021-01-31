"""
Compile HDF5 library

Be sure environment variables are set for your desired compiler.
Use the full compiler path if it's not getting the right compiler.

* FC: Fortran compiler name or path
* CC: C compiler name or path
"""

from __future__ import annotations
import typing as T
import sys
import os
import subprocess
import shutil
import argparse
import tempfile
import json
import importlib.resources
from pathlib import Path

from . import cmake
from .utils import get_cpu_count
from .web import url_retrieve, extract_tar, git_download

# ========= user parameters ======================
BUILDDIR = "build"

HDF5_DIR = "hdf5"
LAPACK_DIR = "lapack"
SCALAPACK_DIR = "scalapack"
MUMPS_DIR = "mumps"

# ========= end of user parameters ================


def cli():
    p = argparse.ArgumentParser(
        description="Compiles prerequisite libraries for Gemini (or other programs)"
    )
    p.add_argument(
        "compiler", help="compiler to build libraries for", choices=["gcc", "intel", "ibmxl"]
    )
    p.add_argument(
        "libs",
        help="libraries to compile",
        choices=["hdf5", "lapack", "mumps", "netcdf", "openmpi", "openmpi3", "scalapack"],
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

    main(P.compiler, P.libs, P.prefix, P.workdir, P.reuse, P.dryrun)


def main(compiler: str, libs: list[str], prefix: str, workdir: str, reuse: bool, dryrun: bool):

    prefix = prefix if prefix else f"~/lib_{compiler}"

    dirs = {
        "prefix": Path(prefix).expanduser().resolve(),
        "workdir": Path(workdir).expanduser().resolve(),
    }

    setup_libs(libs, dirs, compiler, wipe=not reuse, dryrun=dryrun)


def setup_libs(
    libs: list[str],
    dirs: dict[str, Path],
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
        openmpi(dirs, env=env, version="", dryrun=dryrun)
    elif "openmpi3" in libs:
        openmpi(dirs, env=env, version="3", dryrun=dryrun)

    if "lapack" in libs:
        lapack(wipe, dirs, env=env, dryrun=dryrun)
    if "scalapack" in libs:
        scalapack(wipe, dirs, env=env, dryrun=dryrun)
    if "mumps" in libs:
        mumps(wipe, dirs, env=env, dryrun=dryrun)

    if not dryrun:
        print("Installed", libs, "under", dirs["prefix"])


def netcdf_c(
    dirs: dict[str, Path], env: T.Mapping[str, str], wipe: bool = False, dryrun: bool = False
):
    """build and install NetCDF-C"""

    install_dir = dirs["prefix"] / "netcdf"
    source_dir = dirs["workdir"] / "netcdf-c"
    build_dir = source_dir / BUILDDIR

    git_json(source_dir, "netcdf-c")

    hdf5_root = dirs["prefix"] / HDF5_DIR
    if hdf5_root.is_dir():
        lib_args = [f"-DHDF5_ROOT={hdf5_root.as_posix()}"]
    else:
        lib_args = []

    if not cmake.find_library("HDF5 COMPONENTS C Fortran", lib_args, env):
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
    cmake.build(
        source_dir,
        build_dir,
        wipe=wipe,
        env=env,
        run_test=False,
        dryrun=dryrun,
        config_args=c_args + lib_args,
    )


def netcdf_fortran(
    dirs: dict[str, Path], env: T.Mapping[str, str], wipe: bool = False, dryrun: bool = False
):
    """build and install NetCDF-Fortran"""

    install_dir = dirs["prefix"] / "netcdf"
    source_dir = dirs["workdir"] / "netcdf-fortran"
    build_dir = source_dir / BUILDDIR

    git_json(source_dir, "netcdf-fortran")

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
        raise EnvironmentError(
            f"please open a GitHub Issue for your operating system {sys.platform}"
        )

    hdf5_root = dirs["prefix"] / HDF5_DIR
    if hdf5_root.is_dir():
        lib_args = [f"-DHDF5_ROOT={hdf5_root.as_posix()}"]
    else:
        lib_args = []

    if not cmake.find_library("HDF5 COMPONENTS C Fortran", lib_args, env):
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
    cmake.build(
        source_dir, build_dir, wipe=wipe, env=env, run_test=False, dryrun=dryrun, config_args=f_args
    )


def hdf5(dirs: dict[str, Path], env: T.Mapping[str, str]):
    """build and install HDF5
    some systems have broken libz and so have trouble extracting tar.bz2 from Python.
    To avoid this, we git clone the release instead.
    """

    use_cmake = True
    name = "hdf5"
    install_dir = dirs["prefix"] / name
    source_dir = dirs["workdir"] / name
    build_dir = source_dir / BUILDDIR

    git_json(source_dir, "hdf5")

    if use_cmake or os.name == "nt":
        # works for Intel oneAPI on Windows and many other systems/compilers.
        # works for Make or Ninja in general.
        cmd0 = [
            "cmake",
            f"-S{source_dir}",
            f"-DCMAKE_INSTALL_PREFIX={install_dir}",
            "-DHDF5_GENERATE_HEADERS:BOOL=false",
            "-DHDF5_DISABLE_COMPILER_WARNINGS:BOOL=true",
            "-DBUILD_SHARED_LIBS:BOOL=false",
            "-DCMAKE_BUILD_TYPE=Release",
            "-DHDF5_BUILD_FORTRAN:BOOL=true",
            "-DHDF5_BUILD_CPP_LIB:BOOL=false",
            "-DHDF5_BUILD_TOOLS:BOOL=false",
            "-DBUILD_TESTING:BOOL=false",
            "-DHDF5_BUILD_EXAMPLES:BOOL=false",
        ]

        cmd1 = ["cmake", "--build", str(build_dir), "--parallel"]

        cmd2 = ["cmake", "--install", str(build_dir)]

        # this old "cmake .." style command is necessary due to bugs with
        # HDF5 (including 1.10.7) CMakeLists:
        #   CMake Error at config/cmake/HDF5UseFortran.cmake:205 (file):
        #   file failed to open for reading (No such file or directory):
        #   C:/Users/micha/AppData/Local/Temp/hdf5/build/pac_fconftest.out.
        build_dir.mkdir(exist_ok=True)
        subprocess.check_call(cmd0, cwd=build_dir, env=env)
    else:
        cmd0 = [
            "./configure",
            f"--prefix={install_dir}",
            "--enable-fortran",
            "--enable-build-mode=production",
        ]
        cmd1 = ["make", "-j"]
        cmd2 = ["make", "-j", "install"]
        subprocess.check_call(cmd0, cwd=source_dir, env=env)

    subprocess.check_call(cmd1, cwd=source_dir)
    subprocess.check_call(cmd2, cwd=source_dir)


def openmpi(dirs: dict[str, Path], env: T.Mapping[str, str], version: str, dryrun: bool = False):
    """ build and install OpenMPI """
    if os.name == "nt":
        raise EnvironmentError(
            """
OpenMPI is not available in native Windows.
Use MPI on Windows via any of (choose one):

* Windows Subsystem for Linux
* MS-MPI with MSYS2: https://www.scivision.dev/windows-mpi-msys2/
* Intel oneAPI with IntelMPI: https://www.scivision.dev/intel-oneapi-fortran-install/
* Cygwin
"""
        )

    jmeta = json.loads(importlib.resources.read_text(__package__, "libraries.json"))
    version = jmeta[f"openmpi{version}"]["tag"]

    mpi_dir = f"openmpi-{version}"
    install_dir = dirs["prefix"] / mpi_dir
    source_dir = dirs["workdir"] / mpi_dir

    tar_name = f"openmpi-{version}.tar.bz2"
    tarfn = dirs["workdir"] / tar_name

    url = f"https://download.open-mpi.org/release/open-mpi/v{version[:3]}/{tar_name}"
    url_retrieve(url, tarfn)
    extract_tar(tarfn, source_dir)

    cmd = [
        "./configure",
        f"--prefix={install_dir}",
        f"CC={env['CC']}",
        f"CXX={env['CXX']}",
        f"FC={env['FC']}",
    ]

    subprocess.check_call(cmd, cwd=source_dir, env=env)

    Njobs = get_cpu_count()

    cmd = ["make", "-C", str(source_dir), "-j", str(Njobs), "install"]

    if dryrun:
        print("DRYRUN: would have run\n", " ".join(cmd))
        return None

    subprocess.check_call(cmd)


def lapack(wipe: bool, dirs: dict[str, Path], env: T.Mapping[str, str], dryrun: bool = False):
    """ build and insall Lapack """
    install_dir = dirs["prefix"] / LAPACK_DIR
    source_dir = dirs["workdir"] / LAPACK_DIR
    build_dir = source_dir / BUILDDIR

    git_json(source_dir, "lapack")

    args = ["-Dautobuild:BOOL=off", f"-DCMAKE_INSTALL_PREFIX:PATH={install_dir}"]
    cmake.build(source_dir, build_dir, wipe=wipe, env=env, dryrun=dryrun, config_args=args)


def scalapack(wipe: bool, dirs: dict[str, Path], env: T.Mapping[str, str], dryrun: bool = False):
    """ build and install Scalapack """
    install_dir = dirs["prefix"] / SCALAPACK_DIR
    source_dir = dirs["workdir"] / SCALAPACK_DIR
    build_dir = source_dir / BUILDDIR

    git_json(source_dir, "scalapack")

    lapack_root = dirs["prefix"] / LAPACK_DIR
    lib_args = [f"-DLAPACK_ROOT={lapack_root.as_posix()}"]

    if not cmake.find_library("LAPACK", lib_args, env):
        lapack(wipe, dirs, env)

    args = [
        "-Dautobuild:BOOL=off",
        f"-DCMAKE_INSTALL_PREFIX:PATH={install_dir}",
    ]
    cmake.build(
        source_dir, build_dir, wipe=wipe, env=env, dryrun=dryrun, config_args=args + lib_args
    )


def mumps(wipe: bool, dirs: dict[str, Path], env: T.Mapping[str, str], dryrun: bool = False):
    """ build and install Mumps """
    install_dir = dirs["prefix"] / MUMPS_DIR
    source_dir = dirs["workdir"] / MUMPS_DIR
    build_dir = source_dir / BUILDDIR

    scalapack_lib = dirs["prefix"] / SCALAPACK_DIR
    lapack_lib = dirs["prefix"] / LAPACK_DIR

    git_json(source_dir, "mumps")

    if env["FC"] == "ifort":
        lib_args = []
    else:
        lib_args = [
            f"-DSCALAPACK_ROOT:PATH={scalapack_lib.as_posix()}",
            f"-DLAPACK_ROOT:PATH={lapack_lib.as_posix()}",
        ]

    if not cmake.find_library("LAPACK", lib_args, env):
        lapack(wipe, dirs, env)
    if not cmake.find_library("SCALAPACK", lib_args, env):
        scalapack(wipe, dirs, env)

    args = ["-Dautobuild:BOOL=off", f"-DCMAKE_INSTALL_PREFIX:PATH={install_dir}"]
    cmake.build(
        source_dir, build_dir, wipe=wipe, env=env, dryrun=dryrun, config_args=args + lib_args
    )


def git_json(path: Path, name: str):
    jmeta = json.loads(importlib.resources.read_text(__package__, "libraries.json"))

    git_download(path, jmeta[name]["git"], tag=jmeta[name].get("tag"))


def get_compilers(compiler_name: str, **kwargs) -> T.Mapping[str, str]:
    """get paths to compilers

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


if __name__ == "__main__":
    cli()
