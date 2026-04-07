"""
server/app.py — Re-exports the FastAPI app for OpenEnv multi-mode deployment compatibility.
The main application lives in server.py at the repo root.
"""
import sys
import os

# Ensure repo root is on the path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from server import app, main  # noqa: F401

__all__ = ["app", "main"]
