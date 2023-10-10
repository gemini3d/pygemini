from __future__ import annotations
import typing as T
import re
import os
import math
from pathlib import Path
from datetime import datetime, timedelta

from . import find
from . import namelist

NaN = math.nan

__all__ = ["datetime_range", "read_nml"]


def datetime_range(start: datetime, stop: datetime, step: timedelta) -> list[datetime]:
    """
    Generate range of datetime over a closed interval.
    pandas.date_range also defaults to a closed interval.
    That means that the start AND stop time are included.

    Parameters
    ----------
    start : datetime
        start time
    stop : datetime
        stop time
    step : timedelta
        time step

    Returns
    -------
    times : list of datetime
        times requested
    """

    return [start + i * step for i in range((stop - start) // step + 1)]


def read_nml(fn: Path) -> dict[str, T.Any]:
    """parse .nml file
    for now we don't use the f90nml package, though maybe we will in the future.
    Just trying to keep Python prereqs reduced for this simple parsing.
    """

    fn = find.config(fn)

    params = {"nml": fn}

    for k in {
        "base",
        "files",
        "flags",
        "setup",
        "neutral_BG",
        "neutral_perturb",
        "precip",
        "precip_BG",
        "efield",
        "glow",
    }:
        if namelist_exists(fn, k):
            params.update(parse_namelist(fn, k))

    return params


def namelist_exists(fn: Path, nml: str) -> bool:
    """
    Determines if a namelist exists in a file.
    Does not check for proper format / syntax.
    """

    pat = re.compile(r"^\s*&(" + nml + ")$")

    with fn.open("rt") as f:
        for line in f:
            if pat.match(line) is not None:
                return True

    return False


def parse_namelist(file: Path, nml: str) -> dict[str, T.Any]:
    """
    Parses Gemini-specific namelists in a file to dict.
    Does not resolve absolute paths here because that assumes same machine
    """

    r = namelist.read(file, nml)

    P = {}

    if nml == "base":
        P = parse_base(r)
    elif nml == "flags":
        P = _parse_flags(r)
    elif nml == "files":
        P = parse_files(r)
    elif nml == "setup":
        P = parse_setup(r)
    elif nml == "neutral_perturb":
        P = parse_neutral_perturb(r)
    elif nml == "neutral_BG":
        P = parse_neutral_BG(r)
    elif nml == "precip":
        P = {
            "dtprec": timedelta(seconds=float(r["dtprec"])),
            "precdir": r["prec_dir"],
        }
    elif nml == "precip_BG":
        for k in r:
            if k in {"W0BG", "PhiWBG"}:
                P[k] = float(r[k])
            else:
                P[k] = r[k]
    elif nml == "efield":
        P = {
            "dtE0": timedelta(seconds=float(r["dtE0"])),
            "E0dir": r["E0_dir"],
        }
    elif nml == "glow":
        P = {
            "aurmap_dir": r.get("aurmap_dir", "aurmaps"),
            "dtglow": timedelta(seconds=float(r["dtglow"])),
            "dtglowout": float(r["dtglowout"]),
        }
    else:
        raise ValueError(f"Not sure how to parse NML namelist {nml}")

    P = expand_envvar(P)

    return P


def parse_base(r: dict[str, T.Any]) -> dict[str, T.Any]:
    P: dict[str, T.Any] = {
        "tdur": timedelta(seconds=float(r["tdur"])),
        "dtout": timedelta(seconds=float(r["dtout"])),
        "f107a": float(r["activ"][0]),
        "f107": float(r["activ"][1]),
        "Ap": float(r["activ"][2]),
        "tcfl": float(r["tcfl"]),
        "Teinf": float(r["Teinf"]),
    }

    t0 = datetime(int(r["ymd"][0]), int(r["ymd"][1]), int(r["ymd"][2])) + timedelta(
        seconds=float(r["UTsec0"])
    )

    P["time"] = datetime_range(t0, t0 + P["tdur"], P["dtout"])

    return P


def _parse_flags(r: dict[str, T.Any]) -> dict[str, T.Any]:
    P = {}
    for k in r:
        P[k] = int(r[k])

    return P


def parse_files(r: dict[str, T.Any]) -> dict[str, T.Any]:
    P = {}

    for k in {"indat_file", "indat_grid", "indat_size"}:
        P[k] = r[k]

    if "realbits" in r:
        P["realbits"] = int(r["realbits"])
    else:
        P["realbits"] = 32

    return P


def expand_envvar(P: dict[str, T.Any]) -> dict[str, T.Any]:
    """
    Looks for text inbetween @ signs that is an environment variable.
    This helps avoid the need to hardcode paths.

    For example, if the path in the config.nml is like  @GEMINI_SIMROOT@/tohoku/inputs/
    and the user computer has previously set environment varaiable

        GEMINI_SIMROOT=~/data

    then the path will be expanded to ~/data/tohoku/inputs/
    """

    for k in {
        "indat_file",
        "indat_grid",
        "indat_size",
        "eq_dir",
        "eq_archive",
        "E0dir",
        "precdir",
        "sourcedir",
        "aurmap_dir",
    }:
        if k in P:
            if "@" in P[k]:
                # look for first pair @...@ -- following pairs would need to be handled by recursive calls
                i0 = P[k].find("@")
                i1 = P[k].find("@", i0 + 1)
                if i1 >= 0:
                    i1 += i0
                    envvar = P[k][i0 + 1 : i1]
                    envpath = os.environ.get(envvar)
                    if not envpath:
                        raise EnvironmentError(
                            f"Environment variable {envvar} not found, specified in {k}"
                        )
                    P[k] = P[k].replace(P[k][i0 : i1 + 1], envpath)

            P[k] = Path(P[k]).expanduser()

    return P


def parse_neutral_perturb(r: dict[str, T.Any]) -> dict[str, T.Any]:
    P = {
        "interptype": int(r["interptype"]),
        "sourcedir": r["source_dir"],
    }

    for k in {"sourcemlat", "sourcemlon", "dtneu", "dxn", "drhon", "dzn"}:
        try:
            P[k] = float(r[k])
        except KeyError:
            P[k] = NaN

    return P


def parse_neutral_BG(r: dict[str, T.Any]) -> dict[str, T.Any]:
    P: dict[str, T.Any] = {}

    P["flagneuBG"] = False
    if r.get("flagneuBG") and r["flagneuBG"].lower() in {".true.", ".t."}:
        P["flagneuBG"] = True

    for k in {
        "dtneuBG",
    }:
        try:
            P[k] = float(r[k])
        except KeyError:
            P[k] = NaN

    for k in {
        "msis_version",
    }:
        try:
            P[k] = int(r[k])
        except KeyError:
            P[k] = 0

    return P


def parse_setup(r: dict[str, T.Any]) -> dict[str, T.Any]:
    """
    r is str, list of str, or float
    """

    P: dict[str, T.Any] = {}

    for k in r:
        if k in {
            "lxp",
            "lyp",
            "lq",
            "lp",
            "lphi",
            "gridflag",
            "Efield_llon",
            "Efield_llat",
            "precip_llon",
            "precip_llat",
            "random_seed_init",
        }:
            P[k] = int(r[k])
        elif k in {
            "glat",
            "glon",
            "dtheta",
            "dphi",
            "Qprecip",
            "Qprecip_background",
            "E0precip",
            "Etarg",
            "Efield_latwidth",
            "Efield_latoffset",
            "Efield_lonwidth",
            "Efield_lonoffset",
            "altmin",
            "grid_openparm",
        }:
            P[k] = float(r[k])
        elif k == "eqdir":  # eqdir obsolete, should use eq_dir
            P["eq_dir"] = r[k]
        elif k == "setup_functions":
            P["setup_functions"] = [r[k]] if isinstance(r[k], str) else r[k]
        else:
            P[k] = r[k]

    return P
