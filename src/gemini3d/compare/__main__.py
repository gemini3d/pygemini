import sys
import argparse

from . import compare_all


p = argparse.ArgumentParser(description="Compare simulation file outputs and inputs")
p.add_argument("new_dir", help="directory to compare")
p.add_argument("refdir", help="reference directory")
p.add_argument("-only", help="only check in or out", choices=["in", "out"])
p.add_argument(
    "-file_format",
    help="specify file format to read from output dir",
    choices=["h5", "nc", "raw"],
)
P = p.parse_args()

errs = compare_all(P.new_dir, P.refdir, file_format=P.file_format, only=P.only)

if errs:
    for e, v in errs.items():
        print(f"{e} has {v} errors", file=sys.stderr)

    raise SystemExit(f"FAIL: compare {P.new_dir}")

print(f"OK: Gemini comparison {P.new_dir} {P.refdir}")
