# PyGemini

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
