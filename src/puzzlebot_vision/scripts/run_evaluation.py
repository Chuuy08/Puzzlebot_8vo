#!/usr/bin/env python3
"""Convenience wrapper — run from anywhere inside the workspace."""
import os, sys

_pkg = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, _pkg)

from puzzlebot_vision.evaluate_model import main
main()
