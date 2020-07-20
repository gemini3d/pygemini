#!/usr/bin/env python3
import site
import setuptools
import sys

# PEP517 workaround
site.ENABLE_USER_SITE = True

# Python 3.6 workaround
install_requires = ["numpy >= 1.16.0", "scipy", "h5py"]
if sys.version_info < (3, 7):
    install_requires.append("importlib-resources")

setuptools.setup(install_requires=install_requires)
