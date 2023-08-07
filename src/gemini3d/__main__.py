"""
Friendly help if user types

python -m gemini3d
"""

from . import __version__, plot, model, compare
from inspect import getdoc

print(
    f"PyGemini {__version__} is the Python interface to the Gemini3D ionospheric model."
)

print("many PyGemini functions can be used from the command line:")

print("-------\n", getdoc(plot))

print("-------\n", getdoc(model))

print("-------\n", getdoc(compare))
