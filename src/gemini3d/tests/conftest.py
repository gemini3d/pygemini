from __future__ import annotations
from pathlib import Path

import pytest


class Helpers:
    @staticmethod
    def get_test_datadir() -> Path:
        return Path(__file__).parent / "data"


@pytest.fixture
def helpers():
    return Helpers
