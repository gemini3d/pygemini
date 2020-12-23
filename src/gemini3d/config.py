import typing as T
import re
import math
from pathlib import Path
from datetime import datetime, timedelta

from . import find

NaN = math.nan

__all__ = ["read_config"]


# do NOT use lru_cache--can have weird unexpected effects with complicated setups
def read_config(path: Path) -> T.Dict[str, T.Any]:
    """
    read simulation input configuration

    .nml is strongly preferred, .ini is legacy.

    Parameters
    ----------
    path: pathlib.Path
        config file path

    Returns
    -------
    params: dict
        simulation parameters from config file
    """

    file = find.config(path)
    if not file:
        return {}  # {} instead of None to work with .get()

    if file.suffix == ".ini":
        P = read_ini(file)
    else:
        P = read_nml(file)

    return P


def datetime_range(start: datetime, stop: datetime, step: timedelta) -> T.List[datetime]:

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


def read_nml(fn: Path) -> T.Dict[str, T.Any]:
    """parse .nml file
    for now we don't use the f90nml package, though maybe we will in the future.
    Just trying to keep Python prereqs reduced for this simple parsing.
    """

    params: T.Dict[str, T.Any] = {}

    fn = find.config(fn)
    if not fn:
        return params

    params["nml"] = fn
    for n in ("base", "files", "flags", "setup"):
        params.update(read_namelist(fn, n))

    if namelist_exists(fn, "neutral_perturb"):
        params.update(read_namelist(fn, "neutral_perturb"))
    if namelist_exists(fn, "precip"):
        params.update(read_namelist(fn, "precip"))
    if namelist_exists(fn, "efield"):
        params.update(read_namelist(fn, "efield"))
    if namelist_exists(fn, "glow"):
        params.update(read_namelist(fn, "glow"))

    return params


def namelist_exists(fn: Path, namelist: str) -> bool:
    """ determines if a namelist exists in a file """

    pat = re.compile(r"^\s*&(" + namelist + ")$")

    with fn.open("r") as f:
        for line in f:
            if pat.match(line) is not None:
                return True

    return False


def read_namelist(fn: Path, namelist: str) -> T.Dict[str, T.Any]:
    """ read a namelist from an .nml file """

    r: T.Dict[str, T.Sequence[str]] = {}
    nml_pat = re.compile(r"^\s*&(" + namelist + r")")
    end_pat = re.compile(r"^\s*/\s*$")
    val_pat = re.compile(r"^\s*(\w+)\s*=\s*([^!]*)")

    with fn.open("r") as f:
        for line in f:
            if not nml_pat.match(line):
                continue

            for line in f:
                if end_pat.match(line):
                    # end of namelist
                    return parse_namelist(r, namelist)
                val_mat = val_pat.match(line)
                if not val_mat:
                    continue

                key, vals = val_mat.group(1), val_mat.group(2).strip().split(",")
                vals = [v.strip().replace("'", "").replace('"', "") for v in vals]
                r[key] = vals[0] if len(vals) == 1 else vals

    raise KeyError(f"did not find Namelist {namelist} in {fn}")


def parse_namelist(r: T.Dict[str, T.Any], namelist: str) -> T.Dict[str, T.Any]:
    """
    this is Gemini-specific
    don't resolve absolute paths here because that assumes same machine
    """

    P: T.Dict[str, T.Any] = {}

    if namelist == "base":
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
    elif namelist == "flags":
        for k in r:
            P[k] = int(r[k])
    elif namelist == "files":
        for k in ("indat_file", "indat_grid", "indat_size"):
            P[k] = Path(r[k]).expanduser()

        if "file_format" in r:
            P["format"] = r["file_format"]
        else:
            # defaults to type of input
            P["format"] = P["indat_size"].suffix[1:]

        if "realbits" in r:
            P["realbits"] = int(r["realbits"])
        else:
            if P["format"] in ("raw", "dat"):
                P["realbits"] = 64
            else:
                P["realbits"] = 32
    elif namelist == "setup":
        P["alt_scale"] = list(map(float, r["alt_scale"]))

        if "setup_functions" in r:
            P["setup_functions"] = r["setup_functions"]

        for k in ("lxp", "lyp"):
            P[k] = int(r[k])
        for k in (
            "glat",
            "glon",
            "xdist",
            "ydist",
            "alt_min",
            "alt_max",
            "Bincl",
            "nmf",
            "nme",
            "precip_latwidth",
            "precip_lonwidth",
            "Qprecip",
            "Qprecip_background",
            "E0precip",
            "Etarg",
            "Jtarg",
            "Efield_latwidth",
            "Efield_lonwidth",
            # "Eflagdirich",  # future
        ):
            if k in r:
                P[k] = float(r[k])
        if "eqdir" in r:  # old .nml
            P["eq_dir"] = Path(r["eqdir"]).expanduser()
        if "eq_dir" in r:
            P["eq_dir"] = Path(r["eq_dir"]).expanduser()
    elif namelist == "neutral_perturb":
        P["interptype"] = int(r["interptype"])
        P["sourcedir"] = Path(r["source_dir"]).expanduser()

        for k in ("sourcemlat", "sourcemlon", "dtneu", "dxn", "drhon", "dzn"):
            try:
                P[k] = float(r[k])
            except KeyError:
                P[k] = NaN
    elif namelist == "precip":
        P["dtprec"] = timedelta(seconds=float(r["dtprec"]))
        P["precdir"] = Path(r["prec_dir"]).expanduser()
    elif namelist == "efield":
        P["dtE0"] = timedelta(seconds=float(r["dtE0"]))
        P["E0dir"] = Path(r["E0_dir"]).expanduser()
    elif namelist == "glow":
        P["dtglow"] = timedelta(seconds=float(r["dtglow"]))
        P["dtglowout"] = float(r["dtglowout"])

    if not P:
        raise ValueError(f"Not sure how to parse NML namelist {namelist}")

    return P


def read_ini(fn: Path) -> T.Dict[str, T.Any]:
    """ parse .ini file (legacy) """

    P: T.Dict[str, T.Any] = {}

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
