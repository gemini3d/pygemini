# PyGemini

![ci_python](https://github.com/gemini3d/pygemini/workflows/ci/badge.svg)
[![codecov](https://codecov.io/gh/gemini3d/pygemini/branch/main/graph/badge.svg)](https://codecov.io/gh/gemini3d/pygemini)

A Python interface to [Gemini3D](https://github.com/gemini3d/gemini).

## Setup

Setup PyGemini by:

```sh
git clone https://github.com/gemini3d/pygemini

pip install -e pygemini
```

### build

Not all users need to run Gemini3D on the same device where PyGemini is installed.
PyGemini uses the "build on run" method developed by Michael Hirsch, which allows complex multi-language Python packages to install reliably across operating systems (MacOS, Linux, Windows).
Upon the first `gemini3d.run()`, the underlying Gemini3D code is built, including all necessary libraries.

```sh
python -m gemini3d.prereqs
```

allows manually installing those libraries to save rebuild time, but this is optional as Gemini3D automatically downloads and builds missing libraries.

### Developers

For those working with GEMINI Fortran code itself or to work with non-release versions of GEMINI Fortran code:

1. install PyGemini in development mode as above
2. set environment variable GEMINI_ROOT to the Gemini3D Fortran code directory, otherwise PyGemini will Git clone a new copy.

## Run simulation

1. make a [config.nml](https://github.com/gemini3d/gemini/docs/Readme_input.md) with desired parameters for an equilibrium sim.
2. run the equilibrium sim:

    ```sh
    python -m gemini3d.run /path_to/config_eq.nml /path_to/sim_eq/
    ```
3. create a new config.nml for the actual simulation and run

    ```sh
    python -m gemini3d.run /path_to/config.nml /path_to/sim_out/
    ```

## Plot simulation outputs

```sh
python -m gemini3d.plot /path_to/sim_data
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
