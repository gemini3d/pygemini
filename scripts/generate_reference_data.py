#!/usr/bin/env python
"""
Generate reference data from test scenario directory automatically

We choose to halt execution on error to avoid wasting CPU time.
The test scenarios should all generate perfectly.

Normally this scripts is run from the top-level gemini/ directory, outputting
the test reference data to gemini/test_data/:

    python scripts/generate_reference_data.py tests/data

If you wish to generate reference data for only one or a few test case(s), do like:

    python scripts/generate_reference_data.py test_data/ -only
"""

from pathlib import Path
import argparse

import gemini3d.job

TOP_DIR = Path(__file__).resolve().parents[1] / "src/unit_tests/config"


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Generate Gemini reference data")
    p.add_argument(
        "out_dir", help="reference file output directory (typically test_data/)"
    )
    p.add_argument("-only", help="test cases (pattern) to run (default all)")
    p.add_argument("-mpiexec", help="path to desired mpiexec executable")
    p.add_argument("-gemexe", help="path to Gemini3D executable")
    p.add_argument(
        "-f", "--force", help="force regeneration of simulation", action="store_true"
    )
    P = p.parse_args()

    if P.only:
        dirs = sorted([d for d in TOP_DIR.iterdir() if d.is_dir() and P.only in d.name])
    else:
        dirs = sorted([d for d in TOP_DIR.iterdir() if d.is_dir()])

    for d in dirs:
        params = {
            "config_file": d / "config.nml",
            "out_dir": Path(P.out_dir) / d.name,
            "mpiexec": P.mpiexec,
            "gemexe": P.gemexe,
            "force": P.force,
        }

        gemini3d.job.runner(params)
