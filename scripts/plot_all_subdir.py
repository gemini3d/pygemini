#!/usr/bin/env python3
"""
Plot all subdirectories one level below the specified directory.
This is useful if making a change in PyGemini or Gemini3D, to plot many simulations
to visually check for something unexpected.
"""

from pathlib import Path
import argparse
import concurrent.futures
import itertools
import os

import gemini3d.plot as plot


def plot_dir(d: Path, resume: bool) -> None:
    if resume and (d / "plots").is_dir():
        print("SKIP already plotted ", d)
        return
    if not (d / "inputs").is_dir():
        print(f"SKIP no {d}/inputs dir", d)
        return
    print(d)
    plot.plot_all(d)


def print_dir(d: Path, resume: bool) -> None:
    print(d)


def main(top: Path, resume: bool) -> None:
    """
    With ThreadPoolExecutor and ProcessPoolExecutor:
    exception not thrown till iterator is iterated to the exception.
    """

    dirs = (d for d in top.iterdir() if d.is_dir())

    N = os.cpu_count()
    if N is None:
        N = 2
    else:
        N = min(N, 4)

    with concurrent.futures.ProcessPoolExecutor(max_workers=N) as executor:
        try:
            executor.map(plot_dir, dirs, itertools.repeat(resume))
        except KeyboardInterrupt:
            print("KeyboardInterrupt")
            executor.shutdown(wait=True, cancel_futures=True)


if __name__ == "__main__":
    # if __name__ == __main__ is needed for ProcessPoolExecutor to work
    p = argparse.ArgumentParser()
    p.add_argument("top_dir", help="Top level directory to look one level below")
    p.add_argument(
        "-r",
        "--resume",
        help="skip directories that already have plots",
        action="store_true",
    )
    P = p.parse_args()

    top_dir = Path(P.top_dir).expanduser()

    if not top_dir.is_dir():
        raise NotADirectoryError(top_dir)

    main(top_dir, P.resume)
