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

    params: dict[str, T.Any] = {}

    fn = find.config(fn)
    if not fn:
        return params

    params["nml"] = fn
    for n in ("base", "files", "flags"):
        params.update(parse_namelist(fn, n))

    if namelist_exists(fn, "setup"):
        params.update(parse_namelist(fn, "setup"))
    if namelist_exists(fn, "neutral_perturb"):
        params.update(parse_namelist(fn, "neutral_perturb"))
    if namelist_exists(fn, "precip"):
        params.update(parse_namelist(fn, "precip"))
    if namelist_exists(fn, "efield"):
        params.update(parse_namelist(fn, "efield"))
    if namelist_exists(fn, "glow"):
        params.update(parse_namelist(fn, "glow"))

    return params


def namelist_exists(fn: Path, nml: str) -> bool:
    """determines if a namelist exists in a file"""

    pat = re.compile(r"^\s*&(" + nml + ")$")

    with fn.open("r") as f:
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

    P: dict[str, T.Any] = {}

    if nml == "base":
        t0 = datetime(int(r["ymd"][0]), int(r["ymd"][1]), int(r["ymd"][2])) + timedelta(
            seconds=float(r["UTsec0"])
        )
        P["tdur"] = timedelta(seconds=float(r["tdur"]))
        P["dtout"] = timedelta(seconds=float(r["dtout"]))
        P["time"] = datetime_range(t0, t0 + P["tdur"], P["dtout"])

        P["f107a"] = float(r["activ"][0])
        P["f107"] = float(r["activ"][1])
        P["Ap"] = float(r["activ"][2])
        P["tcfl"] = float(r["tcfl"])
        P["Teinf"] = float(r["Teinf"])
    elif nml == "flags":
        for k in r:
            P[k] = int(r[k])
    elif nml == "files":
        for k in ("indat_file", "indat_grid", "indat_size"):
            P[k] = r[k]

        P["file_format"] = r.get("file_format", Path(P["indat_size"]).suffix[1:])
        # defaults to type of input

        if "realbits" in r:
            P["realbits"] = int(r["realbits"])
        else:
            if P["file_format"] in ("raw", "dat"):
                P["realbits"] = 64
            else:
                P["realbits"] = 32
    elif nml == "setup":
        for k in r:
            if k in {"lxp", "lyp", "lq", "lp", "lphi", "gridflag"}:
                P[k] = int(r[k])
            elif k == "eqdir":  # old .nml
                P["eq_dir"] = r[k]
            elif k == "setup_functions":
                P["setup_functions"] = [r[k]] if isinstance(r[k], str) else r[k]
            else:
                P[k] = r[k]
    elif nml == "neutral_perturb":
        P["interptype"] = int(r["interptype"])
        P["sourcedir"] = r["source_dir"]

        for k in ("sourcemlat", "sourcemlon", "dtneu", "dxn", "drhon", "dzn"):
            try:
                P[k] = float(r[k])
            except KeyError:
                P[k] = NaN
    elif nml == "precip":
        P["dtprec"] = timedelta(seconds=float(r["dtprec"]))
        P["precdir"] = r["prec_dir"]
    elif nml == "efield":
        P["dtE0"] = timedelta(seconds=float(r["dtE0"]))
        P["E0dir"] = r["E0_dir"]
    elif nml == "glow":
        if r.get("aurmap_dir"):
            P["aurmap_dir"] = r["aurmap_dir"]
        else:
            # DEPRECATED: workaround for old hard-coded path
            P["aurmap_dir"] = "aurmaps"
        P["dtglow"] = timedelta(seconds=float(r["dtglow"]))
        P["dtglowout"] = float(r["dtglowout"])

    simroot_key = "@GEMINI_SIMROOT@"
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
                    raise EnvironmentError(
                        f"{k} refers to undefined environment variable GEMINI_SIMROOT."
                        "Set it to location to store/load Gemini3D simulations."
                    )
                P[k] = P[k].replace(simroot_key, root, 1)
            P[k] = Path(P[k]).expanduser()

    if not P:
        raise ValueError(f"Not sure how to parse NML namelist {nml}")

    return P


def read_ini(fn: Path) -> dict[str, T.Any]:
    """parse .ini file (legacy)"""

    P: dict[str, T.Any] = {}

    fn = find.config(fn)

    if not fn:
        return P

    with fn.open("r") as f:
        date = list(map(int, f.readline().split()[0].split(",")))[::-1]
        sec = float(f.readline().split()[0])
        t0 = datetime(date[0], date[1], date[2]) + timedelta(seconds=sec)
        P["tdur"] = timedelta(seconds=float(f.readline().split()[0]))
        P["dtout"] = timedelta(seconds=float(f.readline().split()[0]))
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
