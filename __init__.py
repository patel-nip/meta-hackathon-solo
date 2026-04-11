"""
ContextAwareEnv – Social Awareness RL Environment
===================================================

A reinforcement-learning environment that evaluates whether an LLM agent
possesses social awareness — specifically, knowing **when to stay silent**
and **when to proactively interrupt** to help.

The package exposes three tightly-typed Pydantic models (ContextAction,
ContextObservation, ContextState) plus a typed WebSocket client
(ContextEnvClient) for interacting with the FastAPI server.

Quick Start
-----------
>>> from models import ContextAction
>>> from client import ContextEnvClient
>>> import asyncio
>>> async def demo():
...     async with ContextEnvClient(base_url="http://localhost:8000") as env:
...         obs = await env.reset(task_name="easy")
...         result = await env.step(ContextAction(action_type="summarize_screen"))
...         print(result.reward)
>>> asyncio.run(demo())
"""

# ---------------------------------------------------------------------------
# Package metadata
# ---------------------------------------------------------------------------
__version__ = "1.0.0"
__author__ = "hackathon-team"

# ---------------------------------------------------------------------------
# Public re-exports
# ---------------------------------------------------------------------------
# We use try/except to handle both execution modes:
#   1. `pip install -e .` → package is importable as `context_aware_env.models`
#   2. Running from the project directory → `from models import ...`
# ---------------------------------------------------------------------------
try:
    from context_aware_env.models import (
        ContextAction,
        ContextObservation,
        ContextState,
    )
    from context_aware_env.client import ContextEnvClient
    from context_aware_env.utils import (
        clamp_score,
        fuzzy_match_score,
        weighted_keyword_score,
        SCORE_EPSILON,
    )
except ImportError:
    try:
        from models import (  # type: ignore[no-redef]
            ContextAction,
            ContextObservation,
            ContextState,
        )
        from client import ContextEnvClient  # type: ignore[no-redef]
        from utils import (  # type: ignore[no-redef]
            clamp_score,
            fuzzy_match_score,
            weighted_keyword_score,
            SCORE_EPSILON,
        )
    except ImportError:
        # If we can't import at all (e.g. during pip install resolution),
        # don't crash the package init.
        pass

__all__ = [
    # Metadata
    "__version__",
    "__author__",
    # Models
    "ContextAction",
    "ContextObservation",
    "ContextState",
    # Client
    "ContextEnvClient",
    # Utilities
    "clamp_score",
    "fuzzy_match_score",
    "weighted_keyword_score",
    "SCORE_EPSILON",
]
