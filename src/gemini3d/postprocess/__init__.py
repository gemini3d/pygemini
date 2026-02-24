"""
postprocessing module

Things to add based on mat-gemini:
(https://github.com/gemini3d/mat_gemini-scripts/tree/main/%2Bgemscr/%2Bpostprocess)
- Efield
- GDIgrowth
- GDIgrowth_intertial
- Poynting
- collisions3D
- conductivity3D
- conductivity_reconstruct
- current_decompose
- plot2field
- vfield
"""

from .collisions3D import collisionfrequency
from .conduct_all_points import conductivity
