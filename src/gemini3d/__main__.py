"""
CLI scripts
"""

import argparse

from .model_setup import model_setup


def gemini_setup():
    p = argparse.ArgumentParser()
    p.add_argument("config_file", help="path to config*.nml file")
    p.add_argument("out_dir", help="simulation output directory")
    P = p.parse_args()

    model_setup(P.config_file, P.out_dir)
