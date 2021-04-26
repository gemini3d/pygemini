#!/usr/bin/env python3
"""
ZIP directories for Zenodo upload
"""

import os
import argparse
import zipfile
from pathlib import Path


def zip_dirs(path: Path, pattern: str):
    """
    recursively archive a directory
    """

    path = Path(path).expanduser().resolve()

    dlist = [d for d in path.glob(pattern) if d.is_dir()]
    if len(dlist) == 0:
        raise FileNotFoundError(f"no directories to archive under {path} with {pattern}")

    for d in dlist:
        arc_name = d.with_suffix(".zip")
        with zipfile.ZipFile(arc_name, mode="w", compression=zipfile.ZIP_LZMA) as z:
            for root, _, files in os.walk(d):
                for file in files:
                    fn = Path(root, file)
                    afn = fn.relative_to(path)
                    z.write(fn, arcname=afn)
        print("write", arc_name)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("rootdir", help="top-level directory to zip directories under")
    p.add_argument("pattern", help="glob pattern of directories to zip")
    P = p.parse_args()

    zip_dirs(P.rootdir, P.pattern)
