# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTIBILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.

bl_info = {
    "name": "Pytorch_test",
    "author": "Benoit",
    "description": "",
    "blender": (4, 2, 0),
    "version": (1, 0, 0),
    "location": "",
    "warning": "",
    "category": "Generic",
}

import bpy
import os
import sys

# Add lib directory to Python path
# To install a lib do : pip install tree -t .\lib\ ? (not sure)
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
