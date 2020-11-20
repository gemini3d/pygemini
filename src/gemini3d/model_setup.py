"""
setup a new simulation
"""

from pathlib import Path
import typing as T
import shutil

from .config import read_nml
from .grid import makegrid_cart3d
from .plasma import equilibrium_state, equilibrium_resample
from .efield import Efield_BCs
from .particles import particles_BCs
from .fileio import write_state, write_grid


def model_setup(path: Path, out_dir: Path):
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
    p = read_nml(path)

    p["out_dir"] = Path(out_dir).expanduser().resolve()

    for k in ("indat_size", "indat_grid", "indat_file"):
        p[k] = p["out_dir"] / p[k]

    # FIXME: should use is_absolute() ?
    for k in ("eqdir", "eqzip", "E0dir", "precdir"):
        if p.get(k):
            p[k] = (p["out_dir"] / p[k]).resolve()

    # %% copy input config.nml to output dir
    shutil.copy2(p["nml"], p["out_dir"] / "inputs")

    # %% is this equilibrium or interpolated simulation
    if "eqdir" in p:
        model_setup_interp(p)
    else:
        model_setup_equilibrium(p)


def model_setup_equilibrium(p: T.Dict[str, T.Any]):
    # %% GRID GENERATION

    xg = makegrid_cart3d(p)

    write_grid(p, xg)

    # %% Equilibrium input generation
    [ns, Ts, vsx1] = equilibrium_state(p, xg)
    assert ns.shape == Ts.shape == vsx1.shape
    assert ns.shape[0] == 7
    assert ns.shape[1:] == tuple(xg["lx"])

    write_state(p["t0"], ns, vsx1, Ts, p["indat_file"])


def model_setup_interp(p: T.Dict[str, T.Any]):

    xg = makegrid_cart3d(p)

    equilibrium_resample(p, xg)

    # %% potential boundary conditions
    if "E0dir" in p:
        Efield_BCs(p, xg)

    # %% aurora
    if "precdir" in p:
        particles_BCs(p, xg)
