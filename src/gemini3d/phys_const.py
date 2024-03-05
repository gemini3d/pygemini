#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 16 21:12:57 2024

Physical constants used in many calculations

@author: zettergm
"""

import numpy as np

kB = 1.3806503e-23
elchrg = 1.60217646e-19
amu = 1.66053886e-27
ms = np.array([16.0, 30.0, 28.0, 32.0, 14.0, 1.0, 9.1e-31 / amu]) * amu
gammas = np.array([5 / 3, 7 / 5, 7 / 5, 7 / 5, 5 / 3, 5 / 3, 5 / 3])
qs = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, -1.0]) * elchrg
