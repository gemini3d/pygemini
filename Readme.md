# PyGemini

![ci_python](https://github.com/gemini3d/pygemini/workflows/ci/badge.svg)

A Python interface to [Gemini3D](https://github.com/gemini3d/gemini).
In general, making multi-language programs with Python benefit significantly from Python &ge; 3.7.
In general, Python &ge; 3.8 provides additional robustness.

We kindly suggest using Python &ge; 3.7 for any Python project, including PyGemini.

## Setup

Setup PyGemini by:

```sh
git clone https://github.com/gemini3d/pygemini

pip install -e pygemini
```

PyGemini uses the "build on run" method developed by Michael Hirsch, which allows complex multi-language Python packages to install reliably across operating systems (MacOS, Linux, Windows).
Upon the first `import gemini3d`, the underlying C and Fortran Gemini code builds and installs, including all necessary libraries.

PyGemini requires that you have already installed:

* Fortran 2008 compiler, such as: GNU Gfortran, Intel Parallel Studio or Intel oneAPI
* MPI-2 capable library, such as: OpenMPI, MPICH, IntelMPI, MS-MPI

### Developers

For those working with GEMINI Fortran code itself or to work with non-release versions of GEMINI Fortran code:

1. install PyGemini in development mode as above
2. make a symbolic / soft link in Terminal from `pygemini/src/gemini3d/gemini-fortran/` to the top-level directory where you're prototyping Gemini Fortran code. For example, suppose you're in "~/code/" and have Gemini at "~/code/gemini" and PyGemini at "~/code/pygemini". T

   * MacOS / Linux:

        ```sh
        ln -s gemini/ pygemini/src/gemini3d/gemini-fortran/
        ````
    * Windows / PowerShell:

        ```posh
        New-Item -ItemType SymbolicLink -Path "pygemini/src/gemini3d/gemini-fortran/" -Target "gemini/"
        ```

## Run simulation

1. make a [config.nml](https://github.com/gemini3d/gemini/docs/Readme_input.md) with desired parameters for an equilibrium sim.
2. run the equilibrium sim:

    ```sh
    gemini_run /path_to/config_eq.nml /path_to/sim_eq/
    ```
3. create a new config.nml for the actual simulation and run

    ```sh
    gemini_run /path_to/config.nml /path_to/sim_out/
    ```

## Convert data files to HDF5

There is a a script to convert data to HDF5, and another to convert grids to HDF5.
The scripts convert from {raw, Matlab, NetCDF4} to HDF5.
The benefits of doing this are especially significant for raw data, and HDF5 may compress by 50% or more, and make the data self-describing.

```sh
python scripts/convert_data.py h5 ~/mysim
```

```sh
python scripts/convert_grid.py h5 ~/mysim/inputs/simgrid.dat
```
