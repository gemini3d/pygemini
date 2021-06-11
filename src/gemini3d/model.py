"""
setup a new simulation
"""

from __future__ import annotations
import logging
import argparse
from pathlib import Path
import typing as T
import shutil

from .config import read_nml
from .grid import cartesian, tilted_dipole
from .plasma import equilibrium_state, equilibrium_resample
from .efield import Efield_BCs
from .particles import particles_BCs
from .utils import str2func
from . import namelist
from . import write

__all__ = ["setup", "config"]


def config(params: dict[str, T.Any], out_dir: Path):
    """
    top-level API to create a new simulation.
    This is meant to be a front-end to the model.setup() function, creating
    files expected by model.setup(), especially config.nml

    PySat and other Python programs will normally use this function, perhaps via a shim
    to their internal data structures.

    The required namelists are written one-by-one for best modularity.
    The order of namelists does not matter, nor does the order of variables in a namelist.

    Parameters
    ----------

    params: dict
        location, time, and extents thereof
    """

    nml_file = out_dir / "inputs/config.nml"
    nml_file.parent.mkdir(parents=True, exist_ok=True)

    logging.info(f"Creating Gemini3D configuration file under {nml_file}")

    file_format = params.get("file_format", "h5")

    # %% base
    t0 = params["time"][0]
    tend = params["time"][-1]

    base = {
        "ymd": [t0.year, t0.month, t0.day],
        "UTsec0": t0.hour * 3600 + t0.minute * 60 + t0.second + t0.microsecond / 1e6,
        "tdur": (tend - t0).total_seconds(),
        "dtout": params["dtout"],
        "activ": [params["f107a"], params["f107"], params["Ap"]],
        "tcfl": params.get("tcfl", 0.9),
        "Teinf": params.get("Teinf", 1500.0),
    }
    namelist.write(nml_file, namelist="base", data=base)

    # %% flags
    flags = {"flagoutput": params.get("flagoutput", 1), "potsolve": params.get("potsolve", 1)}
    namelist.write(nml_file, "flags", flags)

    # %% files
    files = {
        "indat_size": f"inputs/simsize.{file_format}",
        "indat_grid": f"inputs/simgrid.{file_format}",
        "indat_file": f"inputs/initial_conditions.{file_format}",
    }
    namelist.write(nml_file, "files", files)

    # %% setup
    setup = {
        "glat": params["glat"],
        "glon": params["glon"],
        "xdist": params["x2dist"],
        "ydist": params["x3dist"],
        "alt_min": params["alt_min"],
        "alt_max": params["alt_max"],
        "lxp": params["lx2"],
        "lyp": params["lx3"],
        "Bincl": params["Bincl"],
        "nmf": params["Nmf"],
        "nme": params["Nme"],
    }

    for k in "eq_dir":
        if k in params:
            setup[k] = params[k]

    namelist.write(nml_file, "setup", setup)


def setup(path: Path | dict[str, T.Any], out_dir: Path):
    """
    top-level function to create a new simulation FROM A FILE config.nml

    Parameters
    ----------

    path: pathlib.Path
        path (directory or full path) to config.nml
    out_dir: pathlib.Path
        directory to write simulation artifacts to
    """

    # %% read config.nml
    if isinstance(path, dict):
        cfg = path
    elif isinstance(path, (str, Path)):
        cfg = read_nml(path)
    else:
        raise TypeError("expected Path to config.nml or dict with parameters")

    if not cfg:
        raise FileNotFoundError(f"no configuration found for {out_dir}")

    cfg["out_dir"] = Path(out_dir).expanduser().resolve()

    for k in ("indat_size", "indat_grid", "indat_file"):
        cfg[k] = cfg["out_dir"] / cfg[k]

    # FIXME: should use is_absolute() ?
    for k in ("eq_dir", "eq_archive", "E0dir", "precdir"):
        if cfg.get(k):
            cfg[k] = (cfg["out_dir"] / cfg[k]).resolve()

    # %% copy input config.nml to output dir
    input_dir = cfg["out_dir"] / "inputs"
    input_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(cfg["nml"], input_dir)

    # %% is this equilibrium or interpolated simulation
    if "eq_dir" in cfg:
        interp(cfg)
    else:
        equilibrium(cfg)


def equilibrium(cfg: dict[str, T.Any]):
    # %% GRID GENERATION

    if "lxp" in cfg and "lyp" in cfg:
        xg = cartesian.cart3d(cfg)
    elif "lq" in cfg and "lp" in cfg and "lphi" in cfg:
        xg = tilted_dipole.tilted_dipole3d(cfg)
    else:
        raise ValueError("grid does not seem to be cartesian or curvilinear")

    write.grid(cfg, xg)

    # %% Equilibrium input generation
    dat = equilibrium_state(cfg, xg)

    write.state(cfg["indat_file"], dat)


def interp(cfg: dict[str, T.Any]) -> None:

    if "lxp" in cfg and "lyp" in cfg:
        xg = cartesian.cart3d(cfg)
    elif "lq" in cfg and "lp" in cfg and "lphi" in cfg:
        xg = tilted_dipole.tilted_dipole3d(cfg)
    else:
        raise ValueError("grid does not seem to be cartesian or curvilinear")

    equilibrium_resample(cfg, xg)

    postprocess(cfg, xg)


def postprocess(cfg: dict[str, T.Any], xg: dict[str, T.Any]) -> None:
    """
    defaults to applying config.nml defined E-field and/or precipitation

    However, the user can also apply their own functions in config.nml
    &setup setup_functions
    """

    if "setup_functions" in cfg:
        # assume file to import from is in cwd or Python path
        funcs = (
            [cfg["setup_functions"]]
            if isinstance(cfg["setup_functions"], str)
            else cfg["setup_functions"]
        )

        for name in funcs:
            func = str2func(name)
            func(cfg, xg)

        return

    # %% potential boundary conditions
    if "E0dir" in cfg:
        Efield_BCs(cfg, xg)

    # %% aurora
    if "precdir" in cfg:
        particles_BCs(cfg, xg)


def cli():
    p = argparse.ArgumentParser()
    p.add_argument("config_file", help="path to config*.nml file")
    p.add_argument("out_dir", help="simulation output directory")
    P = p.parse_args()

    setup(P.config_file, P.out_dir)


if __name__ == "__main__":
    cli()
