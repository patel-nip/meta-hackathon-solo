"""
Pydantic data models for the ContextAwareEnv environment.
=========================================================

Defines the strict data contracts (Action, Observation, State) that flow
between the agent, the client, and the server.  Every model uses
``extra="forbid"`` so that typos in field names are caught immediately
rather than silently ignored.

Models
------
ContextAction
    The action the agent can take: stay_silent, summarize_screen, or proactive_help.
ContextObservation
    The observation returned by the environment after each step / reset.
ContextState
    Internal episode state tracked by the environment on the server side.
"""

from typing import Literal

from pydantic import Field, model_validator

from openenv.core.env_server.types import Action, Observation, State


# ---------------------------------------------------------------------------
# All valid action types in one place for reuse and documentation
# ---------------------------------------------------------------------------
VALID_ACTION_TYPES = ("stay_silent", "summarize_screen", "proactive_help")

__all__ = [
    "VALID_ACTION_TYPES",
    "ContextAction",
    "ContextObservation",
    "ContextState",
]


# ── Action ────────────────────────────────────────────────────────────────────


class ContextAction(Action):
    """Action the agent can take in the ContextAwareEnv.

    Attributes
    ----------
    action_type : str
        One of ``"stay_silent"``, ``"summarize_screen"``, or ``"proactive_help"``.
    payload : str
        Free-text payload the agent uses when it speaks (e.g. a help
        message).  Defaults to the empty string for silent actions.

    Validation
    ----------
    * ``extra="forbid"`` – unknown fields raise an error immediately.
    * If ``action_type`` is ``"stay_silent"`` the ``payload`` is forced to
      an empty string (the agent should not speak when silent).
    """

    model_config = {"extra": "forbid"}

    action_type: Literal["stay_silent", "summarize_screen", "proactive_help"] = Field(
        ...,
        description="The type of action the agent chooses to take.",
    )
    payload: str = Field(
        default="",
        description="Optional text payload when the agent decides to speak.",
    )

    # ----- validators -----

    @model_validator(mode="after")
    def _enforce_silent_payload(self) -> "ContextAction":
        """When the agent stays silent it must not send a payload."""
        if self.action_type == "stay_silent" and self.payload:
            # Auto-correct rather than reject – keeps inference robust.
            object.__setattr__(self, "payload", "")
        return self


# ── Observation ───────────────────────────────────────────────────────────────


class ContextObservation(Observation):
    """Observation returned by the environment after each step / reset.

    Inherits ``done`` (bool) and ``reward`` (float | None) from the base
    :class:`Observation` class.

    Attributes
    ----------
    active_app : str
        Name of the currently active application on the user's desktop.
    visible_text : str
        Text content currently visible on the user's screen.
    user_telemetry : str
        Behavioural signal from the user (e.g. ``"idle"``, ``"typing_fast"``).
    explicit_help_request : bool
        Whether the user has explicitly asked for help.
    """

    model_config = {"extra": "forbid"}

    active_app: str = Field(
        ...,
        description="Name of the currently active application on the user's desktop.",
    )
    visible_text: str = Field(
        ...,
        description="Text content currently visible on the user's screen.",
    )
    user_telemetry: str = Field(
        ...,
        description="Behavioural signal from the user (e.g. 'idle', 'typing_fast').",
    )
    explicit_help_request: bool = Field(
        default=False,
        description="Whether the user has explicitly asked for help.",
    )


# ── State ─────────────────────────────────────────────────────────────────────


class ContextState(State):
    """Internal episode state tracked by the environment.

    Inherits ``episode_id`` (str | None) and ``step_count`` (int) from the
    base :class:`State` class.

    Attributes
    ----------
    current_task : str
        The task difficulty tier for the current episode
        (``"easy"``, ``"medium"``, or ``"hard"``).
    silent_turns_completed : int
        Number of consecutive ``"stay_silent"`` turns completed
        (relevant only for the ``"medium"`` task tier).
    """

    model_config = {"extra": "forbid"}

    current_task: str = Field(
        default="easy",
        description="The task difficulty tier for the current episode.",
    )
    silent_turns_completed: int = Field(
        default=0,
        description=(
            "Number of consecutive 'stay_silent' turns completed "
            "(used for 'medium' task)."
        ),
    )
