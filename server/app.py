"""
FastAPI application for the ContextAwareEnv environment.
========================================================

Exposes the environment over HTTP / WebSocket endpoints that
:class:`EnvClient` (and openenv tooling) can consume.

Endpoints
---------
``GET  /health``   – Liveness / readiness probe (for Docker, K8s, and
                     the inference script's startup check).
``WS   /ws``       – WebSocket endpoint managed by ``openenv.create_fastapi_app``.
``GET  /``         – Root endpoint with service metadata.

Usage
-----
.. code-block:: bash

    uvicorn server.app:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import logging
import sys
import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# ---------------------------------------------------------------------------
# Ensure parent directory is on sys.path so `models` can be imported
# when the server is launched via `uvicorn server.app:app` from the
# context_aware_env/ directory.  This is the ONE place we tolerate
# a sys.path adjustment (the entry-point bootstrap).
# ---------------------------------------------------------------------------
_parent_dir = os.path.join(os.path.dirname(__file__), "..")
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)

from openenv.core.env_server import create_fastapi_app

# ---------------------------------------------------------------------------
# Import models — try proper package path first, fall back to local.
# ---------------------------------------------------------------------------
try:
    from context_aware_env.models import ContextAction, ContextObservation
    from context_aware_env.server.environment import ContextAwareEnvironment
except ImportError:
    from models import ContextAction, ContextObservation  # type: ignore[no-redef]
    from server.environment import ContextAwareEnvironment  # type: ignore[no-redef]


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Application factory (create_fastapi_app expects a callable / class)
# ---------------------------------------------------------------------------
app: FastAPI = create_fastapi_app(
    ContextAwareEnvironment,
    ContextAction,
    ContextObservation,
)


# ---------------------------------------------------------------------------
# CORS middleware — allows browser-based UIs to connect (development).
# In production you would restrict `allow_origins` to your domain.
# ---------------------------------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Log configuration at import time (server module load = startup)
# ---------------------------------------------------------------------------
logger.info("ContextAwareEnv server module loaded — tasks: easy, medium, hard")


# ---------------------------------------------------------------------------
# Custom endpoints (on top of what openenv provides)
# ---------------------------------------------------------------------------

@app.get("/health", tags=["ops"])
async def health_check() -> dict:
    """Liveness / readiness probe.

    Returns ``{"status": "healthy"}`` as long as the process is alive
    and the FastAPI event loop is responsive.  Used by:

    * **inference.py** – to wait for the server to become ready.
    * **Docker HEALTHCHECK** – to auto-restart unhealthy containers.
    * **Kubernetes** – for liveness / readiness probes.
    """
    return {"status": "healthy", "service": "context-aware-env", "version": "1.0.0"}


@app.get("/", tags=["info"])
async def root_info() -> dict:
    """Root endpoint — returns service metadata.

    Useful for quick verification that the server is running and for
    automated discovery.
    """
    return {
        "service": "ContextAwareEnv",
        "version": "1.0.0",
        "description": (
            "Social-awareness RL environment — evaluates whether an LLM "
            "agent knows when to stay silent vs. proactively help."
        ),
        "endpoints": {
            "health": "/health",
            "websocket": "/ws",
        },
        "tasks": ["easy", "medium", "hard"],
    }


# ---------------------------------------------------------------------------
# Direct-run entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Entry point for direct execution via ``uv run`` or ``python -m``."""
    import uvicorn

    logger.info("Starting server via main() entry point …")
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info",
    )


if __name__ == "__main__":
    main()
