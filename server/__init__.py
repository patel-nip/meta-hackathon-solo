"""
Server sub-package for the ContextAwareEnv environment.
=======================================================

This package contains:

* :mod:`server.app` – FastAPI application with HTTP and WebSocket endpoints.
* :mod:`server.environment` – Core RL environment logic (task presets,
  reward computation, state management).
"""

try:
    from server.environment import ContextAwareEnvironment  # noqa: F401
except ImportError:
    pass  # Fine during import resolution or pip install

__all__ = ["ContextAwareEnvironment"]
