"""
Wrapper so gen_coll_comm_a2a_2d_sym can do: from symmetry_2d import Symmetry2D
(Module name 2d_symmetry is not importable with plain import because of leading digit.)
"""
import os
import importlib.util

_this_dir = os.path.dirname(os.path.abspath(__file__))
_path = os.path.join(_this_dir, "2d_symmetry.py")
_spec = importlib.util.spec_from_file_location("symmetry_2d_mod", _path)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
Symmetry2D = _mod.Symmetry2D
