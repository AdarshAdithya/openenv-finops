"""
server/app.py — OpenEnv multi-mode deployment entry point.

This file is required by the OpenEnv validator. It defines the FastAPI app
and a callable main() entry point as specified in pyproject.toml [project.scripts].
"""

import sys
import os

# Ensure repo root is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Re-export the FastAPI app from the root server module
from server import app  # noqa: F401


def main() -> None:
    """Start the uvicorn server — entry point for `server` console script."""
    import uvicorn
    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 7860)),
        reload=False,
    )


if __name__ == "__main__":
    main()
