"""
ContextAwareEnvironment – the core RL environment logic.
========================================================

Implements the three-tier grading rubric that tests an LLM agent's
social awareness: knowing when to stay silent vs. proactively help.

Task Tiers
----------
**Easy** – User explicitly requests help → agent should ``summarize_screen``.
**Medium** – User is in deep work (typing fast) → agent should ``stay_silent``
             for 5 consecutive turns (total reward ~0.95).
**Hard** – User is silently frustrated (npm build error, erratic mouse) → agent
           should ``proactive_help`` and mention the specific error details.

Design Decisions
----------------
* ``extra="forbid"`` on all Pydantic models means a typo in the action
  fields is caught immediately.
* ``MAX_STEPS_PER_EPISODE`` prevents runaway agents from looping forever.
* Invalid ``task_name`` raises a ``ValueError`` rather than silently
  falling back to ``"easy"``.
* Scoring uses a combination of fuzzy matching and weighted keyword
  detection for fine-grained, continuous reward signals.
"""

from __future__ import annotations

import hashlib
import logging
import random
from typing import Any, Optional
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State as BaseState  # noqa: F401

# ---------------------------------------------------------------------------
# Import models and utilities using proper package path, with a local-fallback
# so the file still works when executed directly from the project root.
# ---------------------------------------------------------------------------
try:
    from context_aware_env.models import (
        ContextAction,
        ContextObservation,
        ContextState,
    )
    from context_aware_env.utils import (
        SCORE_EPSILON,
        clamp_score,
        fuzzy_match_score,
        weighted_keyword_score,
    )
except ImportError:
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from models import ContextAction, ContextObservation, ContextState  # type: ignore[no-redef]
    from utils import (  # type: ignore[no-redef]
        SCORE_EPSILON,
        clamp_score,
        fuzzy_match_score,
        weighted_keyword_score,
    )


logger = logging.getLogger(__name__)


# ── Constants ─────────────────────────────────────────────────────────────────

MAX_STEPS_PER_EPISODE: int = 20
"""Hard ceiling on the number of steps per episode to prevent runaway agents."""

MEDIUM_SILENT_TURNS_REQUIRED: int = 5
"""Number of consecutive silent turns needed to complete the 'medium' task."""


# ── Task presets ──────────────────────────────────────────────────────────────
#
# Each preset simulates a realistic desktop scenario.  The visible_text
# contains actual content one would see on screen, not placeholder strings.

TASK_PRESETS: dict[str, dict[str, Any]] = {
    "easy": {
        "active_app": "YouTube — Google Chrome",
        "visible_text": (
            "React Tutorial for Beginners — Full Course 2024\n"
            "freeCodeCamp.org · 1.2M views · 8 months ago\n"
            "Learn React JS in this comprehensive tutorial for beginners.\n"
            "Covers components, hooks, state management, and routing.\n"
            "Timestamps: 0:00 Intro | 5:30 Setup | 12:00 Components | 25:00 Hooks"
        ),
        "user_telemetry": "idle",
        "explicit_help_request": True,
        "mouse_activity": "stationary",
        "recent_keystrokes_per_minute": 0,
        "error_count": 0,
    },
    "medium": {
        "active_app": "VS Code — project/src/main.py",
        "visible_text": (
            "import logging\n"
            "from pathlib import Path\n"
            "\n"
            "logger = logging.getLogger(__name__)\n"
            "\n"
            "def load_config(path: str) -> dict:\n"
            '    """Load configuration from a YAML file."""\n'
            "    config_path = Path(path)\n"
            "    if not config_path.exists():\n"
            '        raise FileNotFoundError(f"Config not found: {path}")'
        ),
        "user_telemetry": "typing_moderate",
        "explicit_help_request": False,
        "mouse_activity": "minimal",
        "recent_keystrokes_per_minute": 45,
        "error_count": 0,
    },
    "hard": {
        "active_app": "Terminal — zsh",
        "visible_text": (
            "$ npm run build\n"
            "\n"
            "> my-app@2.1.0 build\n"
            "> react-scripts build\n"
            "\n"
            "Creating an optimized production build...\n"
            "Failed to compile.\n"
            "\n"
            "Module not found: Error: Can't resolve './components/Dashboard'\n"
            "  in '/home/user/my-app/src/pages'\n"
            "\n"
            "npm ERR! code ELIFECYCLE\n"
            "npm ERR! errno 1\n"
            "npm ERR! my-app@2.1.0 build: `react-scripts build`\n"
            "npm ERR! Exit status 1\n"
            "npm ERR!\n"
            "npm ERR! Failed at the my-app@2.1.0 build script.\n"
            "npm ERR! This is probably not a problem with npm.\n"
            "npm ERR! There is likely additional logging output above."
        ),
        "user_telemetry": "erratic_mouse",
        "explicit_help_request": False,
        "mouse_activity": "erratic_clicking",
        "recent_keystrokes_per_minute": 5,
        "error_count": 3,
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
    {   # Turn 1 — session starting, setting up imports and config loader
        "active_app": "VS Code — project/src/main.py",
        "visible_text": (
            "import logging\n"
            "from pathlib import Path\n"
            "\n"
            "logger = logging.getLogger(__name__)\n"
            "\n"
            "def load_config(path: str) -> dict:\n"
            '    """Load configuration from a YAML file."""\n'
            "    config_path = Path(path)\n"
            "    if not config_path.exists():\n"
            '        raise FileNotFoundError(f"Config not found: {path}")'
        ),
        "user_telemetry": "typing_moderate",
        "explicit_help_request": False,
        "mouse_activity": "minimal",
        "recent_keystrokes_per_minute": 45,
        "error_count": 0,
    },
    {   # Turn 2 — writing a data processor class, picking up speed
        "active_app": "VS Code — project/src/processor.py",
        "visible_text": (
            "class DataProcessor:\n"
            '    """Process and validate incoming data records."""\n'
            "\n"
            "    def __init__(self, config: dict, batch_size: int = 32):\n"
            "        self.config = config\n"
            "        self.batch_size = batch_size\n"
            "        self._pipeline: list[Callable] = []\n"
            "        self._error_count = 0\n"
            "\n"
            "    def add_stage(self, fn: Callable) -> 'DataProcessor':\n"
            "        self._pipeline.append(fn)\n"
            "        return self"
        ),
        "user_telemetry": "typing_fast",
        "explicit_help_request": False,
        "mouse_activity": "none",
        "recent_keystrokes_per_minute": 78,
        "error_count": 0,
    },
    {   # Turn 3 — writing transform logic, brief pause to think
        "active_app": "VS Code — project/src/transforms.py",
        "visible_text": (
            "def normalize_record(record: dict) -> dict:\n"
            '    """Normalize field names and validate types."""\n'
            "    normalized = {}\n"
            "    for key, value in record.items():\n"
            "        clean_key = key.strip().lower().replace(' ', '_')\n"
            "        if isinstance(value, str):\n"
            "            value = value.strip()\n"
            "        normalized[clean_key] = value\n"
            "    return normalized\n"
            "\n"
            "# TODO: add schema validation step"
        ),
        "user_telemetry": "brief_pause",
        "explicit_help_request": False,
        "mouse_activity": "scrolling",
        "recent_keystrokes_per_minute": 32,
        "error_count": 0,
    },
    {   # Turn 4 — deep in async processing loop, high focus
        "active_app": "VS Code — project/src/processor.py",
        "visible_text": (
            "    async def process_batch(self, records: list[dict]) -> BatchResult:\n"
            '        """Process a batch through the pipeline."""\n'
            "        results: list[dict] = []\n"
            "        errors: list[ProcessingError] = []\n"
            "        for record in records:\n"
            "            try:\n"
            "                for stage in self._pipeline:\n"
            "                    record = await stage(record)\n"
            "                results.append(record)\n"
            "            except ValidationError as exc:\n"
            "                errors.append(ProcessingError(record=record, error=exc))\n"
            "        return BatchResult(results=results, errors=errors)"
        ),
        "user_telemetry": "typing_fast",
        "explicit_help_request": False,
        "mouse_activity": "none",
        "recent_keystrokes_per_minute": 92,
        "error_count": 0,
    },
    {   # Turn 5 — peak flow, complex orchestration with concurrency control
        "active_app": "VS Code — project/src/orchestrator.py",
        "visible_text": (
            "async def run_pipeline(\n"
            "    source: AsyncIterator[dict],\n"
            "    processor: DataProcessor,\n"
            "    sink: DataSink,\n"
            "    max_concurrency: int = 4,\n"
            ") -> PipelineResult:\n"
            '    """Execute the full ETL pipeline with controlled concurrency."""\n'
            "    semaphore = asyncio.Semaphore(max_concurrency)\n"
            "    async def bounded_process(batch):\n"
            "        async with semaphore:\n"
            "            return await processor.process_batch(batch)\n"
            "    tasks = [bounded_process(b) async for b in batched(source, 32)]"
        ),
        "user_telemetry": "typing_burst",
        "explicit_help_request": False,
        "mouse_activity": "none",
        "recent_keystrokes_per_minute": 110,
        "error_count": 0,
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


# ── Easy task: keyword relevance weights ──────────────────────────────────────
#
# Used alongside fuzzy matching to give fine-grained partial credit
# when the agent mentions relevant terms from the screen context.

EASY_KEYWORDS: dict[str, float] = {
    "react": 0.04,
    "tutorial": 0.04,
    "youtube": 0.03,
    "video": 0.02,
    "help": 0.02,
}

# ── Hard task: multi-signal scoring weights ───────────────────────────────────
#
# Critical keywords: mentioning the technology and error type
HARD_CRITICAL_KEYWORDS: dict[str, float] = {
    "npm": 0.05,
    "error": 0.05,
    "elifecycle": 0.06,
    "build": 0.04,
}

# Specificity keywords: mentioning the exact root cause
HARD_SPECIFICITY_KEYWORDS: dict[str, float] = {
    "dashboard": 0.05,
    "module": 0.04,
    "resolve": 0.03,
    "component": 0.03,
    "import": 0.02,
    "path": 0.02,
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
        # Scoring combines fuzzy similarity + keyword relevance + length.
        if task == "easy":
            if action.action_type == "summarize_screen":
                base_reward = 0.40

                # Fuzzy similarity to ideal summary (up to 0.30)
                sim_score = fuzzy_match_score(
                    action.payload,
                    "user watching react tutorial video on youtube needs help understanding"
                ) * 0.30

                # Keyword relevance bonuses (up to 0.15)
                kwd_score = weighted_keyword_score(action.payload, EASY_KEYWORDS)
                kwd_score = min(kwd_score, 0.15)

                # Length quality — prefer 5–30 word responses
                length = len(action.payload.split())
                len_bonus = 0.15 if 5 <= length <= 30 else 0.0

                reward = base_reward + sim_score + kwd_score + len_bonus
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
        # The terminal shows a real ``npm run build`` failure with an
        # ``ELIFECYCLE`` error and a missing ``Dashboard`` component.
        # The user has erratic mouse movement (frustrated) but hasn't
        # asked for help.
        #
        # Scoring uses multi-signal analysis:
        #   - base score
        #   - fuzzy similarity to ideal response
        #   - critical keyword detection (npm, error, elifecycle, build)
        #   - specificity bonus (dashboard, module, resolve, component)
        #
        # This makes the task genuinely challenging for frontier models:
        # they must parse noisy terminal output and craft a specific
        # response referencing the root cause, not just generic help.
        elif task == "hard":
            if action.action_type == "proactive_help":
                base_reward = 0.35

                # Fuzzy similarity to ideal response (up to 0.25)
                sim_score = fuzzy_match_score(
                    action.payload,
                    "npm build failed with ELIFECYCLE error because module "
                    "Dashboard cannot be resolved check import path"
                ) * 0.25

                # Critical keyword detection (up to 0.20)
                kwd_score = weighted_keyword_score(
                    action.payload, HARD_CRITICAL_KEYWORDS
                )
                kwd_score = min(kwd_score, 0.20)

                # Specificity bonus — mentioning the exact root cause (up to 0.15)
                specificity = weighted_keyword_score(
                    action.payload, HARD_SPECIFICITY_KEYWORDS
                )
                specificity = min(specificity, 0.15)

                reward = base_reward + sim_score + kwd_score + specificity
            else:
                reward = 0.0
            done = True

        # ── Unrecognised task (should never happen after reset validation) ─
        else:
            logger.error("Unrecognised task '%s' in step(). Forcing done.", task)
            reward = 0.0
            done = True

        # ── Finalise ──────────────────────────────────────────────────────
        reward = clamp_score(reward)
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
          3. Code complexity heuristics: longer/nested code indicates higher
             cognitive load.
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
        complexity = min((lines * 0.01) + (indents * 0.015), 0.07)

        # 4. Small pseudo-random noise based on episode ID (deterministic variance)
        # Using md5 hash of episode_id to get a float between 0.0 and 0.01
        hash_digest = hashlib.md5(f"{self._state.episode_id}_{turn}".encode()).hexdigest()
        noise = (int(hash_digest[:4], 16) / 65535.0) * 0.01

        # Combine metrics into final reward
        base_reward = 0.06
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
