from __future__ import annotations
import typing as T
import re
import os
import logging
import math
from pathlib import Path
from datetime import datetime, timedelta

from . import find
from . import namelist

NaN = math.nan


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

    for k in {"base", "files", "flags", "setup", "neutral_perturb", "precip", "efield", "glow"}:
        if namelist_exists(fn, k):
            params.update(parse_namelist(fn, k))

    return params


def namelist_exists(fn: Path, nml: str) -> bool:
    """determines if a namelist exists in a
    does not check for proper formatting etc."""

    pat = re.compile(r"^\s*&(" + nml + ")$")

    with fn.open("rt") as f:
        for line in f:
            if pat.match(line) is not None:
                return True

    return False


def parse_namelist(file: Path, nml: str) -> dict[str, T.Any]:
    """
    this is Gemini-specific
    don't resolve absolute paths here because that assumes same machine
    """

    r = namelist.read(file, nml)

    if nml == "base":
        P = parse_base(r)
    elif nml == "flags":
        P = parse_flags(r)
    elif nml == "files":
        P = parse_files(r)
    elif nml == "setup":
        P = parse_setup(r)
    elif nml == "neutral_perturb":
        P = parse_neutral(r)
    elif nml == "precip":
        P = {
            "dtprec": timedelta(seconds=float(r["dtprec"])),
            "precdir": r["prec_dir"],
        }
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

    P = expand_simroot(P)

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


def parse_flags(r: dict[str, T.Any]) -> dict[str, T.Any]:

    P = {}
    for k in r:
        P[k] = int(r[k])

    return P


def parse_files(r: dict[str, T.Any]) -> dict[str, T.Any]:

    P = {}

    for k in ("indat_file", "indat_grid", "indat_size"):
        P[k] = r[k]

    P["file_format"] = r.get("file_format", Path(P["indat_size"]).suffix)
    # defaults to type of input

    if "realbits" in r:
        P["realbits"] = int(r["realbits"])
    else:
        if P["file_format"] in ("raw", "dat"):
            P["realbits"] = 64
        else:
            P["realbits"] = 32

    return P


def expand_simroot(P: dict[str, T.Any]) -> dict[str, T.Any]:

    simroot_key = "@GEMINI_SIMROOT@"
    default_dir = "~/gemini_sims"

    for k in (
        "indat_file",
        "indat_grid",
        "indat_size",
        "eq_dir",
        "eq_archive",
        "E0dir",
        "precdir",
        "sourcedir",
        "aurmap_dir",
    ):
        if k in P:
            if P[k].startswith(simroot_key):
                root = os.environ.get(simroot_key[1:-1])
                if not root:
                    root = str(Path(default_dir).expanduser())
                    logging.warning(
                        f"{k} refers to undefined environment variable GEMINI_SIMROOT."
                        f"falling back to {root}"
                    )
                P[k] = P[k].replace(simroot_key, root, 1)
            P[k] = Path(P[k]).expanduser()

    return P


def parse_neutral(r: dict[str, T.Any]) -> dict[str, T.Any]:

    P = {
        "interptype": int(r["interptype"]),
        "sourcedir": r["source_dir"],
    }

    for k in ("sourcemlat", "sourcemlon", "dtneu", "dxn", "drhon", "dzn"):
        try:
            P[k] = float(r[k])
        except KeyError:
            P[k] = NaN

    return P


def parse_setup(r: dict[str, T.Any]) -> dict[str, T.Any]:
    """
    r is str, list of str, or float
    """

    P = {}

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
        }:
            P[k] = int(r[k])
        elif k == "eqdir":  # eqdir obsolete, should use eq_dir
            P["eq_dir"] = r[k]
        elif k == "setup_functions":
            P["setup_functions"] = [r[k]] if isinstance(r[k], str) else r[k]
        else:
            P[k] = r[k]

    return P


def read_ini(fn: Path) -> dict[str, T.Any]:
    """parse .ini file
    DEPRECATED
    """

    fn = find.config(fn)

    with fn.open("rt") as f:
        date = list(map(int, f.readline().split()[0].split(",")))[::-1]
        sec = float(f.readline().split()[0])
        t0 = datetime(date[0], date[1], date[2]) + timedelta(seconds=sec)

        P: dict[str, T.Any] = {
            "tdur": timedelta(seconds=float(f.readline().split()[0])),
            "dtout": timedelta(seconds=float(f.readline().split()[0])),
        }
        P["time"] = datetime_range(t0, t0 + P["tdur"], P["dtout"])

        P["f107a"], P["f107"], P["Ap"] = map(float, f.readline().split()[0].split(","))

        P["tcfl"] = float(f.readline().split()[0])
        P["Teinf"] = float(f.readline().split()[0])

        P["potsolve"] = int(f.readline().split()[0])
        P["flagperiodic"] = int(f.readline().split()[0])
        P["flagoutput"] = int(f.readline().split()[0])
        P["flagcap"] = int(f.readline().split()[0])

        for k in ("indat_size", "indat_grid", "indat_file"):
            P[k] = Path(f.readline().strip().replace("'", "").replace('"', "")).expanduser()

    return P
