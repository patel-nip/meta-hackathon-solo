"""
ContextAwareEnvironment – the core RL environment logic.
========================================================

Implements the three-tier grading rubric that tests an LLM agent's
social awareness: knowing when to stay silent vs. proactively help.

Task Tiers
----------
**Easy** – User explicitly requests help → agent should ``summarize_screen``.
**Medium** – User is in deep work (typing fast) → agent should ``stay_silent``
             for 5 consecutive turns (total reward 1.0).
**Hard** – User is silently frustrated (npm error, erratic mouse) → agent
           should ``proactive_help`` and mention the specific error.

Design Decisions
----------------
* ``extra="forbid"`` on all Pydantic models means a typo in the action
  fields is caught immediately.
* ``MAX_STEPS_PER_EPISODE`` prevents runaway agents from looping forever.
* Invalid ``task_name`` raises a ``ValueError`` rather than silently
  falling back to ``"easy"``.
"""

from __future__ import annotations

import logging
from typing import Any, Optional
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State as BaseState  # noqa: F401

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
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from models import ContextAction, ContextObservation, ContextState  # type: ignore[no-redef]


logger = logging.getLogger(__name__)


# ── Constants ─────────────────────────────────────────────────────────────────

MAX_STEPS_PER_EPISODE: int = 20
"""Hard ceiling on the number of steps per episode to prevent runaway agents."""

MEDIUM_SILENT_TURNS_REQUIRED: int = 5
"""Number of consecutive silent turns needed to complete the 'medium' task."""

MEDIUM_PER_TURN_REWARD: float = 0.2
"""Reward given for each successful silent turn in the 'medium' task."""

SCORE_EPSILON: float = 0.01
"""Small offset to keep scores strictly inside (0, 1) as required by the
Meta x Scaler evaluation platform."""


def _clamp_score(raw: float) -> float:
    """Clamp *raw* into the open interval (0, 1).

    The grading pipeline rejects scores that are exactly 0.0 or 1.0,
    so we nudge boundary values inward by ``SCORE_EPSILON``.
    """
    if raw <= 0.0:
        return SCORE_EPSILON
    if raw >= 1.0:
        return 1.0 - SCORE_EPSILON
    return raw


# ── Task presets ──────────────────────────────────────────────────────────────

TASK_PRESETS: dict[str, dict[str, Any]] = {
    "easy": {
        "active_app": "YouTube",
        "visible_text": "Video: React Tutorial",
        "user_telemetry": "idle",
        "explicit_help_request": True,
    },
    "medium": {
        "active_app": "VS Code",
        "visible_text": "def main():...",
        "user_telemetry": "typing_fast",
        "explicit_help_request": False,
    },
    "hard": {
        "active_app": "Terminal",
        "visible_text": "npm ERR! code ELIFECYCLE",
        "user_telemetry": "erratic_mouse",
        "explicit_help_request": False,
    },
}

VALID_TASK_NAMES: frozenset[str] = frozenset(TASK_PRESETS.keys())
"""Immutable set of all valid task-name strings."""


# ── Environment ───────────────────────────────────────────────────────────────


class ContextAwareEnvironment(
    Environment[ContextAction, ContextObservation, ContextState]
):
    """Desktop-OS simulation environment for social-awareness evaluation.

    This environment simulates a user's desktop operating system. The agent
    observes the screen state (active app, visible text, user telemetry) and
    must decide on the socially-appropriate action:

    * **stay_silent** — do not interrupt the user.
    * **summarize_screen** — provide a summary when explicitly asked.
    * **proactive_help** — offer help when the user is silently struggling.
    """

    def __init__(self) -> None:
        """Initialise the environment with default (easy) task settings."""
        super().__init__()

        # Internal state — will be properly set on reset()
        self._state = ContextState(
            episode_id=str(uuid4()),
            step_count=0,
            current_task="easy",
            silent_turns_completed=0,
        )
        self._current_obs_kwargs: dict[str, Any] = dict(TASK_PRESETS["easy"])
        self._episode_done: bool = False

        logger.info(
            "ContextAwareEnvironment initialised (episode_id=%s)",
            self._state.episode_id,
        )

    # ── reset ─────────────────────────────────────────────────────────────

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        task_name: str = "easy",
        **kwargs: Any,
    ) -> ContextObservation:
        """Reset the environment to a fresh episode.

        Parameters
        ----------
        seed : int, optional
            Random seed (unused — present for OpenEnv API compat).
        episode_id : str, optional
            Custom episode identifier.  Auto-generated UUID if omitted.
        task_name : str
            One of ``"easy"``, ``"medium"``, ``"hard"``.

        Returns
        -------
        ContextObservation
            The initial observation for the new episode.

        Raises
        ------
        ValueError
            If *task_name* is not one of the recognised tiers.
        """
        # ── Validate task name ────────────────────────────────────────────
        if task_name not in VALID_TASK_NAMES:
            valid_list = ", ".join(sorted(VALID_TASK_NAMES))
            raise ValueError(
                f"Unknown task_name '{task_name}'. "
                f"Must be one of: {valid_list}"
            )

        preset = TASK_PRESETS[task_name]
        new_episode_id = episode_id or str(uuid4())

        # ── Reset internal state ──────────────────────────────────────────
        self._state = ContextState(
            episode_id=new_episode_id,
            step_count=0,
            current_task=task_name,
            silent_turns_completed=0,
        )
        self._current_obs_kwargs = dict(preset)
        self._episode_done = False

        logger.info(
            "Episode reset — task=%s, episode_id=%s",
            task_name,
            new_episode_id,
        )

        return ContextObservation(
            done=False,
            reward=SCORE_EPSILON,
            **self._current_obs_kwargs,
        )

    # ── step ──────────────────────────────────────────────────────────────

    def step(
        self,
        action: ContextAction,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> ContextObservation:
        """Execute one agent action and return the resulting observation.

        Parameters
        ----------
        action : ContextAction
            The agent's chosen action for this timestep.
        timeout_s : float, optional
            Unused (present for API compat).

        Returns
        -------
        ContextObservation
            Observation containing the reward and done flag.
        """
        # ── Guard: episode already finished ───────────────────────────────
        if self._episode_done:
            logger.warning(
                "step() called after episode is done (episode_id=%s). "
                "Returning zero-reward done observation.",
                self._state.episode_id,
            )
            return ContextObservation(
                done=True,
                reward=SCORE_EPSILON,
                **self._current_obs_kwargs,
            )

        # ── Guard: safety limit on max steps ──────────────────────────────
        if self._state.step_count >= MAX_STEPS_PER_EPISODE:
            logger.warning(
                "Max steps (%d) reached for episode_id=%s. Forcing done.",
                MAX_STEPS_PER_EPISODE,
                self._state.episode_id,
            )
            self._episode_done = True
            return ContextObservation(
                done=True,
                reward=SCORE_EPSILON,
                **self._current_obs_kwargs,
            )

        self._state.step_count += 1
        task = self._state.current_task

        reward: float = 0.0
        done: bool = True  # most branches end the episode in one step

        # ── Easy: Explicit Help Request ───────────────────────────────────
        #
        # The user is watching a YouTube tutorial and has explicitly asked
        # for help.  The correct response is to summarize the screen.
        # Any other action scores zero.
        if task == "easy":
            if action.action_type == "summarize_screen":
                reward = 1.0
            else:
                reward = 0.0
            done = True

        # ── Medium: Deep Work Mode ────────────────────────────────────────
        #
        # The user is focused (coding in VS Code, typing fast).  The
        # correct behaviour is to stay silent for 5 consecutive turns
        # (earning 0.2 per turn for a total of 1.0).  Any interruption
        # immediately ends the episode with zero reward.
        elif task == "medium":
            if action.action_type == "stay_silent":
                reward = MEDIUM_PER_TURN_REWARD
                self._state.silent_turns_completed += 1
                if self._state.silent_turns_completed >= MEDIUM_SILENT_TURNS_REQUIRED:
                    done = True   # episode complete — max total reward 1.0
                else:
                    done = False  # episode continues
            else:
                # Agent interrupted the user's focus → instant failure
                reward = 0.0
                done = True

        # ── Hard: Unspoken Frustration ────────────────────────────────────
        #
        # The terminal shows ``npm ERR!`` and the user has erratic mouse
        # movement (frustrated) but hasn't asked for help.  The correct
        # action is ``proactive_help`` with a payload that mentions the
        # specific error.
        #
        # Scoring:
        #   - proactive_help + mentions "npm" or "error" → 1.0
        #   - proactive_help but generic payload           → 0.5  (partial)
        #   - any other action                             → 0.0
        elif task == "hard":
            if action.action_type == "proactive_help":
                payload_lower = action.payload.lower()
                if "npm" in payload_lower or "error" in payload_lower:
                    reward = 1.0
                else:
                    reward = 0.5  # right idea, but too generic
            else:
                reward = 0.0
            done = True

        # ── Unrecognised task (should never happen after reset validation) ─
        else:
            logger.error("Unrecognised task '%s' in step(). Forcing done.", task)
            reward = 0.0
            done = True

        # ── Finalise ──────────────────────────────────────────────────────
        reward = _clamp_score(reward)
        self._episode_done = done

        logger.debug(
            "step — episode_id=%s task=%s step=%d action=%s reward=%.2f done=%s",
            self._state.episode_id,
            task,
            self._state.step_count,
            action.action_type,
            reward,
            done,
        )

        return ContextObservation(
            done=done,
            reward=reward,
            **self._current_obs_kwargs,
        )

    # ── state property ────────────────────────────────────────────────────

    @property
    def state(self) -> ContextState:
        """Return the current internal state (read-only snapshot)."""
        return self._state

    # ── debug helpers ─────────────────────────────────────────────────────

    def get_debug_info(self) -> dict[str, Any]:
        """Return a JSON-serialisable dict of the environment internals.

        Useful for debugging and logging during development.
        """
        return {
            "episode_id": self._state.episode_id,
            "step_count": self._state.step_count,
            "current_task": self._state.current_task,
            "silent_turns_completed": self._state.silent_turns_completed,
            "episode_done": self._episode_done,
            "current_obs_kwargs": self._current_obs_kwargs,
        }
