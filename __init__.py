bl_info = {
    "name": "PointsTracker",
    "author": "Broussel",
    "description": "Advanced point tracking add-on for Blender",
    "blender": (4, 2, 0),
    "version": (1, 0, 0),
    "location": "Movie Clip Editor > N Panel > 'PointsTracker'",
    "warning": "",
    "category": "Tracking",
}

import bpy
import os
import sys


addon_dir = os.path.dirname(__file__)
lib_dir = os.path.join(addon_dir, "lib")
if os.path.exists(lib_dir) and lib_dir not in sys.path:
    sys.path.append(lib_dir)
else:
    print("Warning: 'lib' directory not found. Dependencies may be missing.")

from . import point_tracking, models

modules = {point_tracking}


def register():
    for module in modules:
        module.register()


def unregister():
    for module in modules:
        module.unregister()
