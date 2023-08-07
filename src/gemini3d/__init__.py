"""
PyGemini is the Python interface to the Gemini3D ionospheric model.
"""

from __future__ import annotations
from pathlib import Path

__version__ = "1.8.0"

PYGEMINI_ROOT = Path(__path__[0])  # type: ignore

WAVELEN = [
    "3371",
    "4278",
    "5200",
    "5577",
    "6300",
    "7320",
    "10400",
    "3466",
    "7774",
    "8446",
    "3726",
    "LBH",
    "1356",
    "1493",
    "1304",
]

SPECIES = ["O+", "ns1", "ns2", "ns3", "N+", "protons", "electrons"]
# must be list, not tuple
LSP = len(SPECIES)
