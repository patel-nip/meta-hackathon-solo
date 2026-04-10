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

import difflib
import hashlib
import logging
import random
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

SCORE_EPSILON: float = 0.01
"""Small offset to keep scores strictly inside (0, 1) as required by the
Meta x Scaler evaluation platform."""


def fuzzy_match_score(predicted: str, reference: str) -> float:
    """Calculate a continuous string similarity score between 0.0 and 1.0."""
    if not predicted:
        return 0.0
    return difflib.SequenceMatcher(None, predicted.lower(), reference.lower()).ratio()


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
        "visible_text": "def main():\n    config = load_config()",
        "user_telemetry": "typing_moderate",
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

# ── Medium task: evolving coding-session contexts ─────────────────────────────
#
# Each turn simulates a progression through a real coding session.
# The observation changes each step — different code visible, different
# typing behaviour — so the reward is computed from *analysing* the
# current context rather than from a fixed formula.

MEDIUM_STEP_CONTEXTS: list[dict[str, Any]] = [
    {   # Turn 1 — session just started, user warming up
        "active_app": "VS Code",
        "visible_text": "def main():\n    config = load_config()",
        "user_telemetry": "typing_moderate",
        "explicit_help_request": False,
    },
    {   # Turn 2 — user writing a class, picking up speed
        "active_app": "VS Code",
        "visible_text": "class DataProcessor:\n    def __init__(self, config):\n        self.pipeline = []",
        "user_telemetry": "typing_fast",
        "explicit_help_request": False,
    },
    {   # Turn 3 — brief pause to think, then resumes
        "active_app": "VS Code",
        "visible_text": "# TODO: refactor this section\ndef process(data, pipeline):\n    results = []",
        "user_telemetry": "brief_pause",
        "explicit_help_request": False,
    },
    {   # Turn 4 — deep in a loop, high focus
        "active_app": "VS Code",
        "visible_text": "for item in dataset:\n    result = transform(item, weights)\n    validated.append(result)",
        "user_telemetry": "typing_fast",
        "explicit_help_request": False,
    },
    {   # Turn 5 — peak flow, complex async code
        "active_app": "VS Code",
        "visible_text": "async def fetch_batch(urls: list[str]) -> list[dict]:\n    async with aiohttp.ClientSession() as session:",
        "user_telemetry": "typing_burst",
        "explicit_help_request": False,
    },
]

# Telemetry → normalised typing-intensity score  (used by reward computation)
TYPING_INTENSITY_SCORES: dict[str, float] = {
    "idle":            0.05,
    "typing_moderate": 0.40,
    "typing_fast":     0.70,
    "brief_pause":     0.50,   # ambiguous — harder to stay silent
    "typing_burst":    0.90,
}


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
            reward=0.0,
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
                reward=0.0,
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
                reward=0.0,
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
                base_reward = 0.4
                # similarity gives up to 0.4
                sim_score = fuzzy_match_score(
                    action.payload, 
                    "watching react tutorial video on youtube"
                ) * 0.4
                
                # word count penalty if too short or too long
                length = len(action.payload.split())
                len_bonus = 0.2 if 5 <= length <= 25 else 0.0
                
                reward = base_reward + sim_score + len_bonus
            else:
                reward = 0.0
            done = True

        # ── Medium: Deep Work Mode ────────────────────────────────────────
        #
        # The user is coding in VS Code during a focused session.
        # Each turn presents an evolving snapshot of the session:
        # different code on screen, varying typing speed, etc.
        # The agent must stay silent for 5 consecutive turns.
        #
        # Reward is computed by *analysing* the current observation:
        #   - typing intensity   (faster typing → deeper focus)
        #   - code complexity    (more complex code → more critical)
        #   - session momentum   (deeper into flow → higher reward)
        #   - stochastic noise   (natural variance between runs)
        elif task == "medium":
            if action.action_type == "stay_silent":
                self._state.silent_turns_completed += 1
                turn = self._state.silent_turns_completed

                # Advance the observation to the current turn's context
                ctx_idx = min(turn - 1, len(MEDIUM_STEP_CONTEXTS) - 1)
                self._current_obs_kwargs = dict(MEDIUM_STEP_CONTEXTS[ctx_idx])

                # ── Analyse the observation to compute reward ─────────
                reward = self._compute_medium_step_reward(
                    self._current_obs_kwargs, turn
                )

                if self._state.silent_turns_completed >= MEDIUM_SILENT_TURNS_REQUIRED:
                    done = True   # episode complete
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
        # Scoring is dynamic for continuous variance:
        #   - base score + fuzzy match + keyword checks
        elif task == "hard":
            if action.action_type == "proactive_help":
                base_reward = 0.4
                sim_score = fuzzy_match_score(
                    action.payload, 
                    "npm error code elifecycle in terminal"
                ) * 0.4
                
                payload_lower = action.payload.lower()
                kwd_bonus = 0.0
                if "npm" in payload_lower: 
                    kwd_bonus += 0.1
                if "error" in payload_lower or "err" in payload_lower: 
                    kwd_bonus += 0.1
                
                reward = base_reward + sim_score + kwd_bonus
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

    # ── medium task context-analysis reward ───────────────────────────────

    def _compute_medium_step_reward(self, obs_kwargs: dict[str, Any], turn: int) -> float:
        """Compute the reward for a medium step by analysing the context.

        Instead of hardcoding standard rewards per turn, we evaluate:
          1. Base momentum: deeper into the session = more value in staying silent.
          2. Typing intensity: high focus (fast/burst) increases the reward.
          3. Code complexity heuristcs: longer/nested code indicates higher cognitive load.
          4. Small deterministic pseudo-random noise for natural scoring variance.
        """
        # 1. Base momentum from how long they've been working (turn index 1-5)
        momentum_score = turn * 0.02
        
        # 2. Typing intensity from telemetry
        telemetry = obs_kwargs.get("user_telemetry", "idle")
        intensity = TYPING_INTENSITY_SCORES.get(telemetry, 0.1) * 0.05
        
        # 3. Code complexity heuristics (lines, indentations)
        code = obs_kwargs.get("visible_text", "")
        lines = code.count("\n") + 1
        indents = code.count("    ")
        complexity = min((lines * 0.01) + (indents * 0.015), 0.08)
        
        # 4. Small pseudo-random noise based on episode ID (deterministic variance)
        # Using md5 hash of episode_id to get a float between 0.0 and 0.01
        hash_digest = hashlib.md5(f"{self._state.episode_id}_{turn}".encode()).hexdigest()
        noise = (int(hash_digest[:4], 16) / 65535.0) * 0.02
        
        # Combine metrics into final reward
        # Base starts around 0.10, scales up to ~0.25 based on momentum and complexity
        base_reward = 0.08
        final_reward = base_reward + momentum_score + intensity + complexity + noise
        
        logger.debug(
            "Computed medium reward: turn=%d -> momentum=%.3f intensity=%.3f complexity=%.3f noise=%.3f -> %.3f",
            turn, momentum_score, intensity, complexity, noise, final_reward
        )
        return final_reward

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
