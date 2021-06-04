import sys
import argparse
from pathlib import Path
from dateutil.parser import parse

from . import compare_all
from .plot import plotdiff
from ..utils import to_datetime
from .. import read


def compare_cli(P):
    errs = compare_all(P.new_dir, P.ref_dir, file_format=P.file_format, only=P.only)

    if errs:
        for e, v in errs.items():
            print(f"{e} has {v} errors", file=sys.stderr)

        raise SystemExit(f"FAIL: compare {P.new_dir}")

    print(f"OK: Gemini comparison {P.new_dir} {P.ref_dir}")


def plot_cli(P):
    ref_path = Path(P.ref_dir).expanduser().resolve(strict=True)
    new_path = Path(P.new_dir).expanduser().resolve(strict=True)

    if P.time:
        time = parse(P.time)
        new = read.frame(new_path, time, var=P.name)
        ref = read.frame(ref_path, time, var=P.name)
    else:
        if not ref_path.is_file():
            raise FileNotFoundError(f"{ref_path} must be a file when not specifying time")
        if not new_path.is_file():
            raise FileNotFoundError(f"{new_path} must be a file when not specifying time")

        new = read.data(new_path, var=P.name)
        ref = read.data(ref_path, var=P.name)

        new_path = new_path.parent
        ref_path = ref_path.parent

    new = new[P.name]
    ref = ref[P.name]

    plotdiff(new, ref, P.name, to_datetime(new.time), new_path, ref_path)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Compare simulation file outputs and inputs")
    p.add_argument("new_dir", help="directory to compare")
    p.add_argument("ref_dir", help="reference directory")
    p.add_argument("-plot", help="plot instead of numeric compare", action="store_true")
    p.add_argument("-only", help="only check in or out", choices=["in", "out"])
    p.add_argument("-name", help="variable name")
    p.add_argument("-t", "--time", help="requested time (if directory given)")
    p.add_argument(
        "-file_format",
        help="specify file format to read from output dir",
        choices=["h5", "nc", "raw"],
    )
    P = p.parse_args()

    if P.plot:
        plot_cli(P)
    else:
        compare_cli(P)
