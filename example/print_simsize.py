#!/usr/bin/env python3

import argparse

import gemini3d.read as read


p = argparse.ArgumentParser()
p.add_argument("fn", help="path to simsize")
args = p.parse_args()

print(read.simsize(args.fn))
