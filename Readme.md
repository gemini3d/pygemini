# PyGemini

![ci_python](https://github.com/gemini3d/pygemini/workflows/ci/badge.svg)

A Python interface to [Gemini3D](https://github.com/gemini3d/gemini)

## Setup

Setup Gemini and prereqs from the pygemini/ directory:

```sh
python3 setup.py develop --user
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
python scripts/convert_data_hdf5.py ~/mysim
```

```sh
python scripts/convert_grid_hdf5.py ~/mysim/inputs/simgrid.dat
```
