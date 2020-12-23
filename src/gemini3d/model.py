"""
setup a new simulation
"""

import importlib
import argparse
from pathlib import Path
import typing as T
import shutil

from .config import read_nml
from . import grid
from .plasma import equilibrium_state, equilibrium_resample
from .efield import Efield_BCs
from .particles import particles_BCs
from . import write


def setup(p: T.Union[Path, T.Dict[str, T.Any]], out_dir: Path):
    """
    top-level function to create a new simulation

    Parameters
    ----------

    path: pathlib.Path
        path (directory or full path) to config.nml
    out_dir: pathlib.Path
        directory to write simulation artifacts to
    """

    # %% read config.nml
    if isinstance(p, dict):
        cfg = p
    elif isinstance(p, (str, Path)):
        cfg = read_nml(p)
    else:
        raise TypeError("expected Path to config.nml or dict with parameters")

    if not cfg:
        raise FileNotFoundError(f"no configuration found for {out_dir}")

    cfg["out_dir"] = Path(out_dir).expanduser().resolve()

    for k in ("indat_size", "indat_grid", "indat_file"):
        cfg[k] = cfg["out_dir"] / cfg[k]

    # FIXME: should use is_absolute() ?
    for k in ("eq_dir", "eqzip", "E0dir", "precdir"):
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


def equilibrium(p: T.Dict[str, T.Any]):
    # %% GRID GENERATION

    xg = grid.cart3d(p)

    write.grid(p, xg)

    # %% Equilibrium input generation
    [ns, Ts, vsx1] = equilibrium_state(p, xg)
    assert ns.shape == Ts.shape == vsx1.shape
    assert ns.shape[0] == 7
    assert ns.shape[1:] == tuple(xg["lx"])

    write.state(p["time"][0], ns, vsx1, Ts, p["indat_file"])


def interp(cfg: T.Dict[str, T.Any]) -> None:

    xg = grid.cart3d(cfg)

    equilibrium_resample(cfg, xg)

    postprocess(cfg, xg)


def postprocess(cfg: T.Dict[str, T.Any], xg: T.Dict[str, T.Any]) -> None:
    """
    defaults to applying config.nml defined E-field and/or precipitation

    However, the user can also apply their own functions in config.nml
    &setup setup_functions
    """

    if "setup_functions" in cfg:
        # assume file to import from is in cwd or Python path
        for name in cfg["setup_functions"]:
            mod_name = ".".join(name.split(".")[:-1])
            func_name = name.split(".")[-1]
            mod = importlib.import_module(mod_name)
            getattr(mod, func_name)(cfg, xg)

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
