#!/usr/bin/env python3

import gemini3d.fileio
import argparse


p = argparse.ArgumentParser()
p.add_argument("fn", help="path to simsize")
p = p.parse_args()

print(gemini3d.fileio.get_simsize(p.fn))
