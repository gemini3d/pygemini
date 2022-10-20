# PyGemini

![ci](https://github.com/gemini3d/pygemini/workflows/ci/badge.svg)

A Python interface to [Gemini3D](https://github.com/gemini3d/gemini), funded in part by NASA HDEE.

Setup PyGemini by:

```sh
git clone https://github.com/gemini3d/pygemini

pip install -e pygemini
```

## Developers

For those working with GEMINI Fortran code itself or to work with non-release versions of GEMINI Fortran code:

1. install PyGemini in development mode as above
2. set environment variable GEMINI_ROOT to the Gemini3D Fortran code directory, otherwise PyGemini will Git clone a new copy.

## Run simulation

1. make a [config.nml](https://github.com/gemini3d/gemini/docs/Readme_input.md) with desired parameters for an equilibrium sim.
2. setup and/or run the equilibrium sim:

    ```sh
    python -m gemini3d.model /sim_equil/config.nml /path_to/sim_eq/
    # or
    python -m gemini3d.run /sim_equil/config.nml /path_to/sim_eq/
    ```
3. create a new config.nml for the actual simulation and run

    ```sh
    python -m gemini3d.model /sim/config.nml /path_to/sim_out/
    # or
    python -m gemini3d.run /sim/config.nml /path_to/sim_out/
    ```

## Plots

An important part of any simulation is viewing the output.
Because of the large data involved, most plotting functions automatically save PNG stacks to disk for quick flipping through with your preferred image viewing program.

### Grid

Help ensure the simulation grid is what you intended by the following, which can be used before or after running the simulation.

```python
import gemini3d.plot

gemini3d.plot.grid("path/to/sim")
```

### simulation outputs

These commands create plots and save to disk under the "plots/" directory under the specified data directory.

command line:

```sh
python -m gemini3d.plot path/to/data -save png
```

or from within Python:

```python
import gemini3d.plot as plot

plot.frame("path/to/data", datetime(2020, 1, 2, 1, 2, 3), saveplot_fmt="png")

# or

plot.plot_all("path/to/data", saveplot_fmt="png")
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
