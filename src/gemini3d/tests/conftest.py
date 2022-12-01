from __future__ import annotations
from pathlib import Path
import sys
import importlib.resources as pkgr

import pytest


class Helpers:
    @staticmethod
    def get_test_datadir() -> Path:
        return Path(__file__).parent / "data"

    @staticmethod
    def get_pkg_file(package: str, filename: str) -> Path:
        """Get a file from a package.
        This function works for 3.7, 3.8 using a deprecated method,
        and uses the recommended method for Python >= 3.9

        Parameters
        ----------
        package : str
            Package name.
        filename : str
            File name.

        Returns
        -------
        Path
            Path to the file.

        NOTE: this probably assumes the install is Zip safe
        """

        if sys.version_info < (3, 9):
            with pkgr.path(package, filename) as f:
                return f
        else:
            with pkgr.as_file(pkgr.files(package).joinpath(filename)) as f:
                return f


@pytest.fixture
def helpers():
    return Helpers
