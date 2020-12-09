#!/usr/bin/env python3

import gemini3d.read as read
import argparse


p = argparse.ArgumentParser()
p.add_argument("fn", help="path to simsize")
p = p.parse_args()

print(read.simsize(p.fn))
