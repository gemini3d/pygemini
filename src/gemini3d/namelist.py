"""
read and write Fortran standard namelist
This is very basic--see f90nml Python package.
"""

from __future__ import annotations
import typing as T
from pathlib import Path
import re

import numpy as np


__all__ = ["read", "write"]


def read(file: Path, namelist: str) -> dict[str, T.Any]:
    """read a namelist from an .nml file, as strings

    Parameters
    ----------

    file: pathlib.Path
        Namelist file to read
    namelist: str
        Namelist to read from file

    Returns
    -------

    cfg: dict
        data contained in namelist
    """

    r: dict[str, list[str]] = {}
    nml_pat = re.compile(r"^\s*&(" + namelist + r")")
    end_pat = re.compile(r"^\s*/\s*$")
    val_pat = re.compile(r"^\s*(\w+)\s*=\s*([^!]*)")

    with file.open("rt") as f:
        for line in f:
            if not nml_pat.match(line):
                continue

            for line in f:
                if end_pat.match(line):
                    # end of namelist
                    return r
                val_mat = val_pat.match(line)
                if not val_mat:
                    continue

                key, vals = val_mat.group(1), val_mat.group(2).strip().split(",")
                values: list[T.Any] = []
                for v in vals:
                    v = v.strip().replace("'", "").replace('"', "")
                    try:
                        values.append(float(v))
                    except ValueError:
                        values.append(v)
                r[key] = values[0] if len(values) == 1 else values

    raise KeyError(f"did not find Namelist {namelist} in {file}")


def write(file: Path, namelist: str, data: dict[str, T.Any], overwrite: bool = False):
    """
    writes a basic Fortran namelist to a .nml text file

    Parameters
    ----------

    file: pathlib.Path
        Namelist file to read
    namelist: str
        Namelist to read from file
    data: dict
        data to write to namelist
    """

    def _write_scalar(f, key: str, value: T.Any):
        f.write(f"{key} = {value}\n")

    def _write_string(f, key: str, value: str):
        f.write(f'{key} = "{value}"\n')

    def _write_value(f, key: str, value: T.Any):
        if isinstance(value, (float, int)):
            _write_scalar(f, key, value)
        elif isinstance(value, str):
            _write_string(f, key, value)
        elif isinstance(value, (tuple, list)):
            if isinstance(value[0], str):
                s = ",".join([f'"{v}"' for v in value])
                _write_scalar(f, key, s)
                return
            s = ",".join(map(str, value))
            _write_scalar(f, key, s)
        elif isinstance(value, np.ndarray):
            _write_scalar(f, key, value.astype(str))
        else:
            raise TypeError(f"unsure how to handle {type(value)}")

    file = file.expanduser().resolve()

    if file.is_dir():
        raise OSError(f"give a filename, not a directory {file}")

    mode = "w" if overwrite else "a"

    with file.open(mode=mode) as f:
        if mode == "a":
            # arbitrary, for aesthetics
            f.write("\n")

        f.write(f"&{namelist}\n")

        for k, v in data.items():
            _write_value(f, k, v)

        f.write("/\n")
        # closes namelist
        # ensure a trailing blank line or file will fail to read in Fortran
