"""
ContextEnvClient – typed async WebSocket client for the ContextAwareEnv.
========================================================================

Implements the three abstract methods required by
:class:`openenv.core.env_client.EnvClient`:

* ``_step_payload``  – serialise a :class:`ContextAction` for the server.
* ``_parse_result``  – deserialise a server response into
  ``StepResult[ContextObservation]``.
* ``_parse_state``   – deserialise a server response into
  :class:`ContextState`.

Usage
-----
>>> import asyncio
>>> from context_aware_env.client import ContextEnvClient
>>> from context_aware_env.models import ContextAction
>>>
>>> async def demo():
...     async with ContextEnvClient(base_url="http://localhost:8000") as env:
...         obs = await env.reset(task_name="easy")
...         result = await env.step(
...             ContextAction(action_type="summarize_screen")
...         )
...         print(f"reward={result.reward}, done={result.done}")
>>>
>>> asyncio.run(demo())
"""

from __future__ import annotations

import logging
from typing import Any, Dict

from openenv.core.env_client import EnvClient
from openenv.core.client_types import StepResult

# ---------------------------------------------------------------------------
# Import models using proper package path, with a local-fallback so the
# file still works when executed directly from the project root.
# ---------------------------------------------------------------------------
try:
    from context_aware_env.models import (
        ContextAction,
        ContextObservation,
        ContextState,
    )
except ImportError:
    from models import ContextAction, ContextObservation, ContextState  # type: ignore[no-redef]

__all__ = ["ContextEnvClient"]

logger = logging.getLogger(__name__)


# ── Client implementation ─────────────────────────────────────────────────────


class ContextEnvClient(EnvClient[ContextAction, ContextObservation, ContextState]):
    """Typed async client for the ContextAwareEnv server.

    This client handles the serialisation/deserialisation layer between
    Python-typed models and the JSON that flows over the WebSocket.
    All heavy lifting (connection management, reconnection) is handled
    by the parent :class:`EnvClient`.
    """

    # ── serialize outbound action ─────────────────────────────────────────

    def _step_payload(self, action: ContextAction) -> Dict[str, Any]:
        """Convert a :class:`ContextAction` to the JSON dict expected by the server.

        Parameters
        ----------
        action : ContextAction
            The agent's chosen action for the current step.

        Returns
        -------
        dict
            JSON-serialisable dictionary representation of the action.
        """
        return action.model_dump()

    # ── deserialize inbound step result ───────────────────────────────────

    def _parse_result(self, payload: Dict[str, Any]) -> StepResult[ContextObservation]:
        """Parse a server JSON response into a typed :class:`StepResult`.

        The server wraps the observation fields in an ``"observation"``
        key and places ``reward`` / ``done`` at the top level.

        Parameters
        ----------
        payload : dict
            Raw JSON dict received from the server.

        Returns
        -------
        StepResult[ContextObservation]
            Typed result containing the observation, reward, and done flag.

        Raises
        ------
        KeyError, pydantic.ValidationError
            If required fields are missing from the payload.
        """
        obs_data: Dict[str, Any] = payload.get("observation", payload)

        # Merge top-level reward/done into the observation dict so the
        # Pydantic model receives all required fields.
        obs_data_complete = dict(obs_data)
        if "reward" not in obs_data_complete:
            obs_data_complete["reward"] = payload.get("reward", 0.0)
        if "done" not in obs_data_complete:
            obs_data_complete["done"] = payload.get("done", False)

        observation = ContextObservation(**obs_data_complete)

        return StepResult(
            observation=observation,
            reward=payload.get("reward", observation.reward),
            done=payload.get("done", observation.done),
        )

    # ── deserialize inbound state ─────────────────────────────────────────

    def _parse_state(self, payload: Dict[str, Any]) -> ContextState:
        """Parse a server JSON response into a typed :class:`ContextState`.

        Parameters
        ----------
        payload : dict
            Raw JSON dict received from the server.

        Returns
        -------
        ContextState
            The current internal state of the environment.
        """
        return ContextState(**payload)