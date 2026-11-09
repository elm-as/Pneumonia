"""Root conftest for test initialization.

Why: On Windows, PyTorch DLLs (c10.dll) must be initialized before any Qt / PySide
dynamic libraries are loaded by third-party plugins.
"""

import torch  # Pre-loads PyTorch DLLs into memory cleanly

