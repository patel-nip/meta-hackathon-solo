#!/usr/bin/env python3
"""
Inference script for the ContextAwareEnv.
==========================================

Runs the LLM agent through three task tiers (easy, medium, hard) and
prints structured log lines for the automated evaluation parser.

Architecture
------------
1.  **Server lifecycle** — Starts the FastAPI server as a subprocess,
    waits for the ``/health`` endpoint to respond, and tears it down
    on exit.
2.  **WebSocket episodes** — Each task tier runs as a WebSocket session
    so that ``reset()`` and ``step()`` share the same environment
    instance on the server.
3.  **LLM interaction** — Queries an OpenAI-compatible API (HuggingFace
    Inference Providers) with exponential-backoff retry logic.
4.  **Defensive parsing** — Multiple strategies (direct JSON, regex
    extraction, nested-object unwrapping, keyword detection) ensure
    that any LLM output can be converted to a valid action.

Environment Variables
---------------------
``API_BASE_URL``
    Base URL for the OpenAI-compatible API.
``MODEL_NAME``
    Model identifier to use for chat completions.
``HF_TOKEN``
    HuggingFace token (forwarded as the API key).
``SERVER_WAIT_TIMEOUT``
    Seconds to wait for the server to become healthy (default: 30).

CRITICAL: This script must NEVER crash.  Every LLM parsing failure
is caught and defaults to a safe ``stay_silent`` action.
"""

from __future__ import annotations

import asyncio
import atexit
import json
import os
import re
import signal
import subprocess
import sys
import time
from typing import Optional

# ---------------------------------------------------------------------------
# Import models — try proper package path first, fall back to local.
# ---------------------------------------------------------------------------
try:
    from context_aware_env.models import ContextAction, ContextObservation
except ImportError:
    from models import ContextAction, ContextObservation  # type: ignore[no-redef]


# ═══════════════════════════════════════════════════════════════════════════════
#  CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

# ── API settings  (HuggingFace Inference Providers, OpenAI-compatible) ────────
#
# YOUR TOKEN MUST HAVE THE RIGHT PERMISSION:
#   1. Go to https://huggingface.co/settings/tokens
#   2. Click "Create new token" → "Fine-grained"
#   3. CHECK the box: "Make calls to Inference Providers"
#   4. Paste the new token below
#
# The correct base URL is router.huggingface.co/v1 (NOT api-inference)

API_BASE_URL: str = os.environ.get(
    "API_BASE_URL", "https://router.huggingface.co/v1"
)
MODEL_NAME: str = os.environ.get(
    "MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct"
)
# Read token from environment variable OR from `hf auth login` disk cache
try:
    from huggingface_hub import get_token
    _cached_token = get_token()
except ImportError:
    _cached_token = None

HF_TOKEN: str = os.environ.get("HF_TOKEN", "") or _cached_token or ""

# ── Runtime constants ─────────────────────────────────────────────────────────

TASKS: list[str] = ["easy_1", "medium_1", "hard_1"]
TASK_NAME_MAP: dict[str, str] = {"easy_1": "easy", "medium_1": "medium", "hard_1": "hard"}
MAX_STEPS: int = 8
SERVER_PORT: int = 8000

# ── Score clamping ────────────────────────────────────────────────────────────
# The Meta x Scaler grading pipeline rejects scores exactly 0.0 or 1.0.
# Every score must be strictly in the open interval (0, 1).
SCORE_EPSILON: float = 0.01

def _clamp_score(raw: float) -> float:
    """Clamp *raw* into the open interval (0, 1)."""
    if raw <= 0.0:
        return SCORE_EPSILON
    if raw >= 1.0:
        return 1.0 - SCORE_EPSILON
    return raw

# ENV_SERVER_URL: set this to your deployed HF Space URL to run against the
# remote environment instead of starting a local server.
#   e.g.  ENV_SERVER_URL=https://patel-nip-meta-hackathon.hf.space
_env_server_url: str = os.environ.get("ENV_SERVER_URL", "https://patel-nip-meta-hackathon.hf.space").rstrip("/")
USE_REMOTE_SERVER: bool = bool(_env_server_url)

if USE_REMOTE_SERVER:
    SERVER_URL = _env_server_url
    # HTTPS -> wss://, HTTP -> ws://
    _ws_scheme = "wss" if SERVER_URL.startswith("https") else "ws"
    _ws_host = SERVER_URL.split("://", 1)[1]
    WS_URL = f"{_ws_scheme}://{_ws_host}/ws"
else:
    SERVER_URL = f"http://localhost:{SERVER_PORT}"
    WS_URL = f"ws://localhost:{SERVER_PORT}/ws"

ENV_NAME: str = "ContextAwareEnv"

# ── Retry configuration for LLM calls ────────────────────────────────────────

LLM_MAX_RETRIES: int = 3
"""Number of attempts for each LLM API call (1 initial + 2 retries)."""

LLM_RETRY_BASE_DELAY: float = 1.0
"""Base delay in seconds for exponential backoff between LLM retries."""

# ── HTTP status codes that should NOT be retried ──────────────────────────────
# Auth failures, permission denials, and invalid requests won't fix themselves.
NO_RETRY_STATUS_CODES: frozenset[int] = frozenset({400, 401, 403, 404, 422})

# ── Server startup ────────────────────────────────────────────────────────────

SERVER_WAIT_TIMEOUT: float = float(
    os.environ.get("SERVER_WAIT_TIMEOUT", "30")
)
"""Seconds to wait for the FastAPI server to become healthy."""


# ═══════════════════════════════════════════════════════════════════════════════
#  DEFAULT (SAFE-FALLBACK) ACTION
# ═══════════════════════════════════════════════════════════════════════════════

DEFAULT_ACTION = ContextAction(action_type="stay_silent", payload="")


# ═══════════════════════════════════════════════════════════════════════════════
#  SYSTEM PROMPT FOR THE LLM
# ═══════════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """\
You are a desktop AI assistant.  You observe the user's screen and decide
whether to stay silent, summarize the screen, or proactively offer help.

You MUST respond with ONLY a valid JSON object (no markdown, no extra text).

The JSON must have exactly these fields:
{
  "action_type": "<one of: stay_silent | summarize_screen | proactive_help>",
  "payload": "<string – your message if you speak, or empty>"
}

Guidelines:
- If the user explicitly asks for help, use "summarize_screen".
- If the user is focused (typing fast, coding), use "stay_silent".
- If the user is frustrated (errors on screen, erratic behaviour) but hasn't
  asked for help, use "proactive_help" and reference the specific error in
  your payload.
"""


# ═══════════════════════════════════════════════════════════════════════════════
#  SERVER LIFECYCLE
# ═══════════════════════════════════════════════════════════════════════════════

_server_process: Optional[subprocess.Popen] = None


def start_server() -> subprocess.Popen:
    """Start the FastAPI server as a subprocess.

    The server is launched with ``uvicorn`` pointing at ``server.app:app``.
    An ``atexit`` handler is registered so the server is always cleaned up,
    even if the script crashes.

    Returns
    -------
    subprocess.Popen
        The server process handle.
    """
    global _server_process

    env = os.environ.copy()
    _server_process = subprocess.Popen(
        [
            sys.executable, "-m", "uvicorn",
            "server.app:app",
            "--host", "0.0.0.0",
            "--port", str(SERVER_PORT),
        ],
        env=env,
        cwd=os.path.dirname(os.path.abspath(__file__)),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    atexit.register(stop_server)

    # Wait for the server to be ready before returning
    _wait_for_server(max_wait=SERVER_WAIT_TIMEOUT)
    return _server_process


def _wait_for_server(max_wait: float = 30.0) -> None:
    """Poll the ``/health`` endpoint until the server responds with 200.

    Parameters
    ----------
    max_wait : float
        Maximum seconds to wait before giving up.
    """
    import urllib.request
    import urllib.error

    deadline = time.time() + max_wait
    attempt = 0

    while time.time() < deadline:
        attempt += 1
        try:
            req = urllib.request.Request(f"{SERVER_URL}/health")
            with urllib.request.urlopen(req, timeout=2) as resp:
                if resp.status == 200:
                    print(
                        f"[INFO] Server healthy after {attempt} "
                        f"poll{'s' if attempt != 1 else ''}"
                    )
                    return
        except Exception:
            pass
        time.sleep(0.5)

    print(
        f"[WARN] Server may not be fully ready after {max_wait}s",
        file=sys.stderr,
    )


def stop_server() -> None:
    """Terminate the server subprocess gracefully.

    Sends SIGTERM first and waits up to 5 seconds.  If the process is
    still alive after that, it is forcefully killed.
    """
    global _server_process

    if _server_process is None:
        return

    print("[INFO] Stopping server …")
    _server_process.terminate()
    try:
        _server_process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        _server_process.kill()
        _server_process.wait(timeout=3)
    _server_process = None

    # Flush any buffered output (important on Windows)
    sys.stdout.flush()
    sys.stderr.flush()


# ═══════════════════════════════════════════════════════════════════════════════
#  WEBSOCKET-BASED ENVIRONMENT INTERACTION
# ═══════════════════════════════════════════════════════════════════════════════


def _parse_ws_observation(resp_data: dict) -> ContextObservation:
    """Parse the OpenEnv WebSocket response into a ContextObservation.

    OpenEnv serialises observations as::

        {"observation": {<custom fields>}, "reward": float, "done": bool}

    ``reward`` and ``done`` are at the **top** of *resp_data*, not inside
    the ``observation`` sub-dict.

    Parameters
    ----------
    resp_data : dict
        The ``data`` payload from the WebSocket message.

    Returns
    -------
    ContextObservation
    """
    obs_fields: dict = dict(resp_data.get("observation", {}))
    obs_fields["reward"] = _clamp_score(float(resp_data.get("reward", 0.0)))
    obs_fields["done"] = resp_data.get("done", False)
    return ContextObservation(**obs_fields)


async def ws_run_episode(task_name: str, task_label: str) -> tuple[list[float], int, str, list[str]]:
    """Run a single episode over WebSocket.

    Parameters
    ----------
    task_name : str
        The env-level task name ("easy", "medium", "hard").
    task_label : str
        The display label for logging ("easy_1", "medium_1", "hard_1").

    Returns
    -------
    tuple[list[float], int, str, list[str]]
        ``(rewards_list, step_count, episode_id, action_descriptions)``
    """
    import websockets  # lazy import — only needed at runtime

    rewards: list[float] = []
    action_descs: list[str] = []
    step_num: int = 0
    episode_id: str = "unknown"

    async with websockets.connect(WS_URL, open_timeout=10) as ws:
        # ── RESET ────────────────────────────────────────────────────────
        reset_msg = {"type": "reset", "data": {"task_name": task_name}}
        await ws.send(json.dumps(reset_msg))
        reset_resp = json.loads(
            await asyncio.wait_for(ws.recv(), timeout=30)
        )

        if reset_resp.get("type") == "error":
            raise RuntimeError(f"Reset error: {reset_resp}")

        obs = _parse_ws_observation(reset_resp.get("data", {}))

        # Try to capture the episode_id from the response
        state_data = reset_resp.get("state", {})
        if isinstance(state_data, dict):
            episode_id = state_data.get("episode_id", "unknown")

        messages: list[dict[str, str]] = [
            {"role": "system", "content": SYSTEM_PROMPT}
        ]
        done = obs.done

        # ── STEP LOOP ────────────────────────────────────────────────────
        while not done and step_num < MAX_STEPS:
            step_num += 1

            # Build the user prompt from the current observation
            user_msg = build_user_message(obs)
            messages.append({"role": "user", "content": user_msg})

            # Query the LLM (with retry)
            raw_response = query_llm(messages)

            # Parse the LLM output defensively
            try:
                action = parse_action(raw_response)
            except Exception:
                action = DEFAULT_ACTION

            messages.append(
                {"role": "assistant", "content": raw_response or "{}"}
            )

            # Build action description for logging (clean, no payload dump)
            action_desc = f"{action.action_type}(task={task_name})"

            # ── STEP ─────────────────────────────────────────────────────
            step_msg = {"type": "step", "data": action.model_dump()}
            await ws.send(json.dumps(step_msg))
            step_resp = json.loads(
                await asyncio.wait_for(ws.recv(), timeout=30)
            )

            if step_resp.get("type") == "error":
                log_step(step_num, action_desc, 0.01, True, episode_id)
                rewards.append(0.01)
                action_descs.append(action_desc)
                break

            obs = _parse_ws_observation(step_resp.get("data", {}))

            reward = _clamp_score(float(obs.reward) if obs.reward is not None else 0.0)
            done = obs.done
            rewards.append(reward)
            action_descs.append(action_desc)

            log_step(step_num, action_desc, reward, done, episode_id)

    return rewards, step_num, episode_id, action_descs


# ═══════════════════════════════════════════════════════════════════════════════
#  LLM INTERACTION
# ═══════════════════════════════════════════════════════════════════════════════


def build_user_message(obs: ContextObservation) -> str:
    """Build the user-turn prompt from the current observation.

    The prompt is a structured, line-separated description of the
    user's current desktop state.

    Parameters
    ----------
    obs : ContextObservation
        The environment's latest observation.

    Returns
    -------
    str
        A plain-text prompt for the LLM.
    """
    return (
        f"Active app: {obs.active_app}\n"
        f"Visible text: {obs.visible_text}\n"
        f"User telemetry: {obs.user_telemetry}\n"
        f"Explicit help request: {obs.explicit_help_request}\n"
        "\nRespond with ONLY a JSON object."
    )


def parse_action(raw: str) -> ContextAction:
    """Defensively parse the LLM output into a ContextAction.

    Tries multiple strategies in order of reliability:

    1.  **Direct JSON parse** — the ideal case.
    2.  **Strip markdown fences** — LLMs often wrap JSON in ````json …` ```.
    3.  **Extract first JSON object via regex** — handles preamble text.
    4.  **Nested-object unwrapping** — handles ``{"response": {...}}``.
    5.  **Keyword detection fallback** — last resort heuristic.
    6.  **DEFAULT_ACTION** — guaranteed safe fallback.

    Parameters
    ----------
    raw : str
        The raw string output from the LLM.

    Returns
    -------
    ContextAction
        A valid action, even if the LLM output was garbage.
    """
    if not raw or not raw.strip():
        return DEFAULT_ACTION

    cleaned = raw.strip()

    # ── Strip markdown code fences ────────────────────────────────────────
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
        cleaned = cleaned.strip()

    # ── Attempt 1: direct JSON parse ──────────────────────────────────────
    try:
        data = json.loads(cleaned)
        if isinstance(data, dict):
            # Handle nested response objects: {"response": {"action_type": ...}}
            if "action_type" not in data:
                for key in ("response", "action", "result", "output"):
                    if key in data and isinstance(data[key], dict):
                        data = data[key]
                        break
            return ContextAction(**data)
    except Exception:
        pass

    # ── Attempt 2: regex for first JSON object (handles nested braces) ────
    try:
        # This regex handles one level of nesting
        match = re.search(
            r"\{(?:[^{}]|\{[^{}]*\})*\}", raw, re.DOTALL
        )
        if match:
            data = json.loads(match.group())
            if isinstance(data, dict):
                # Handle nested response objects
                if "action_type" not in data:
                    for key in ("response", "action", "result", "output"):
                        if key in data and isinstance(data[key], dict):
                            data = data[key]
                            break
                return ContextAction(**data)
    except Exception:
        pass

    # ── Attempt 3: keyword detection fallback ─────────────────────────────
    lower = raw.lower()
    if "proactive_help" in lower:
        return ContextAction(
            action_type="proactive_help",
            payload=raw[:200],
        )
    if "summarize_screen" in lower:
        return ContextAction(
            action_type="summarize_screen",
            payload="",
        )

    return DEFAULT_ACTION


# ── Lazily-initialised OpenAI client (created once, reused) ───────────────────
_llm_client: Optional["OpenAI"] = None  # type: ignore[name-defined]


def _get_llm_client() -> "OpenAI":  # type: ignore[name-defined]
    """Return (and lazily create) a singleton OpenAI client."""
    global _llm_client
    if _llm_client is None:
        from openai import OpenAI
        _llm_client = OpenAI(
            base_url=API_BASE_URL,
            api_key=HF_TOKEN or "no-key",
        )
    return _llm_client


def _is_non_retryable(exc: Exception) -> bool:
    """Return True if the error is a non-retryable HTTP status (e.g. 401, 403).

    Auth errors and client mistakes will never succeed on retry, so we
    skip the backoff delay and fail immediately.
    """
    exc_str = str(exc)
    for code in NO_RETRY_STATUS_CODES:
        if f"Error code: {code}" in exc_str or f"status_code={code}" in exc_str:
            return True
    return False


def query_llm(messages: list[dict]) -> str:
    """Call the OpenAI-compatible API with exponential-backoff retry.

    Retries up to ``LLM_MAX_RETRIES`` times with exponential backoff
    (1s, 2s, 4s …) for transient errors (429 rate-limit, 503 overloaded).

    Non-retryable errors (401 auth, 403 permission, 404 not found) are
    detected immediately and skip the retry loop to avoid wasting time.

    Parameters
    ----------
    messages : list[dict]
        The conversation history in OpenAI chat format.

    Returns
    -------
    str
        The assistant's response content, or ``""`` on failure.
    """
    client = _get_llm_client()
    last_error: Optional[Exception] = None

    for attempt in range(1, LLM_MAX_RETRIES + 1):
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                temperature=0.2,
                max_tokens=256,
            )
            content = response.choices[0].message.content or ""
            return content

        except Exception as exc:
            last_error = exc

            # ── Non-retryable errors: fail fast ───────────────────────
            if _is_non_retryable(exc):
                print(
                    f"[WARN] LLM call failed (non-retryable): {exc}"
                )
                sys.stdout.flush()
                return ""

            # ── Retryable errors: backoff and retry ───────────────────
            if attempt < LLM_MAX_RETRIES:
                delay = LLM_RETRY_BASE_DELAY * (2 ** (attempt - 1))
                print(
                    f"[WARN] LLM call attempt {attempt}/{LLM_MAX_RETRIES} "
                    f"failed: {exc}. Retrying in {delay:.1f}s …"
                )
                sys.stdout.flush()
                time.sleep(delay)
            else:
                print(
                    f"[WARN] LLM call failed after {LLM_MAX_RETRIES} "
                    f"attempts. Last error: {last_error}"
                )
                sys.stdout.flush()

    return ""


# ═══════════════════════════════════════════════════════════════════════════════
#  LOGGING HELPERS
# ═══════════════════════════════════════════════════════════════════════════════


def log_start(task: str) -> None:
    """Log the beginning of a task-tier evaluation."""
    print(f"[START] task={task} env=local model={MODEL_NAME}")
    sys.stdout.flush()


def log_step(
    step: int,
    action_desc: str,
    reward: float,
    done: bool,
    episode_id: str = "",
) -> None:
    """Log a single step result."""
    reward = _clamp_score(reward)
    done_str = "true" if done else "false"
    print(
        f"[STEP] step={step} action={action_desc} "
        f"reward={reward:.2f} done={done_str} error=null"
    )
    sys.stdout.flush()


def log_end(
    success: bool,
    steps: int,
    score: float,
    rewards: list[float],
) -> None:
    """Log the end of a task-tier evaluation with score and rewards."""
    success_str = "true" if success else "false"
    clamped = [_clamp_score(r) for r in rewards]
    score = _clamp_score(score)
    rewards_str = ",".join(f"{r:.2f}" for r in clamped)
    print(
        f"[END] success={success_str} steps={steps} "
        f"score={score:.3f} rewards={rewards_str}"
    )
    sys.stdout.flush()


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN LOOP
# ═══════════════════════════════════════════════════════════════════════════════


def _compute_score(rewards: list[float]) -> float:
    """Compute the aggregate score from per-step rewards.

    Uses the mean of clamped rewards, then clamps the result
    to stay strictly within (0, 1).
    """
    if not rewards:
        return SCORE_EPSILON
    clamped = [_clamp_score(r) for r in rewards]
    avg = sum(clamped) / len(clamped)
    return _clamp_score(avg)


async def _run_all_tasks() -> dict[str, dict]:
    """Run all task tiers asynchronously and return a summary.

    This runs inside a single ``asyncio.run()`` call so the event loop
    is created only once (avoids deprecation warnings on Python 3.12+).

    Returns
    -------
    dict
        Mapping from task label to ``{"success": bool, "score": float,
        "steps": int, "rewards": list[float]}``.
    """
    summary: dict[str, dict] = {}

    for task_label in TASKS:
        # Map label -> env task name (e.g. "easy_1" -> "easy")
        env_task = TASK_NAME_MAP.get(task_label, task_label)

        log_start(task_label)

        try:
            rewards, step_num, episode_id, action_descs = await ws_run_episode(
                env_task, task_label
            )
        except Exception as exc:
            # Emit valid log lines so the evaluation parser is happy
            print(
                f"[WARN] episode failed for task={task_label}: {exc}",
                file=sys.stderr,
            )
            log_step(1, "stay_silent()", 0.01, True)
            log_end(False, 1, 0.01, [0.01])
            summary[task_label] = {
                "success": False, "score": 0.01, "steps": 1, "rewards": [0.01]
            }
            continue

        # Edge case: loop ended without any steps
        if step_num == 0:
            step_num = 1
            rewards = [0.01]
            log_step(1, "stay_silent()", 0.01, True)

        score = _compute_score(rewards)
        success = score > SCORE_EPSILON
        log_end(success, step_num, score, rewards)

        summary[task_label] = {
            "success": success,
            "score": score,
            "steps": step_num,
            "rewards": rewards,
        }

    return summary


def run() -> None:
    """Execute all three task tiers and log structured output.

    This is the main entry point.  It:

    1. Starts the FastAPI server as a subprocess.
    2. Runs all task tiers (easy → medium → hard).
    3. Prints a summary report with per-tier and aggregate results.
    4. Shuts down the server.
    """
    # ── Graceful Ctrl+C handling ──────────────────────────────────────────
    def _sigint_handler(sig, frame):
        print("\n[INFO] Ctrl+C received — shutting down …")
        stop_server()
        sys.exit(1)

    signal.signal(signal.SIGINT, _sigint_handler)

    # ── Validate HF_TOKEN before wasting time ─────────────────────────────
    if not HF_TOKEN:
        print("\n" + "!" * 60)
        print("  WARNING: HF_TOKEN is not set!")
        print("  LLM calls will fail with 401 Unauthorized.")
        print("  Only the 'medium' task can pass without an LLM")
        print("  (because its correct action is 'stay_silent').")
        print("")
        print("  Set your token:")
        print('    $env:HF_TOKEN = "hf_your_token_here"   # PowerShell')
        print('    export HF_TOKEN="hf_your_token_here"   # Bash')
        print("!" * 60 + "\n")
        sys.stdout.flush()

    # ── Start the server (skip if using a remote HF Space) ────────────────
    if USE_REMOTE_SERVER:
        print(f"[INFO] Using remote environment: {SERVER_URL}")
        sys.stdout.flush()
    else:
        start_server()

    # ── Run all tasks in a single event loop ──────────────────────────────
    try:
        summary = asyncio.run(_run_all_tasks())
    except Exception as exc:
        print(f"[ERROR] Fatal error during inference: {exc}", file=sys.stderr)
        summary = {}
    finally:
        if not USE_REMOTE_SERVER:
            stop_server()

    # ── Print summary report ──────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  INFERENCE SUMMARY")
    print("=" * 60)

    total_score = 0.0
    total_tasks = len(TASKS)
    passed_tasks = 0

    for task_label in TASKS:
        info = summary.get(task_label, {"success": False, "score": SCORE_EPSILON, "steps": 0, "rewards": []})
        status = "PASS" if info["success"] else "FAIL"
        task_score = _clamp_score(info["score"])
        rewards_str = ",".join(f"{_clamp_score(r):.2f}" for r in info.get("rewards", []))
        print(
            f"  {task_label:10s}  {status}  "
            f"score={task_score:.3f}  steps={info['steps']}  "
            f"rewards={rewards_str}"
        )
        total_score += task_score
        if info["success"]:
            passed_tasks += 1

    aggregate_score = _clamp_score(total_score / max(total_tasks, 1))
    print("-" * 60)
    print(
        f"  TOTAL    {passed_tasks}/{total_tasks} passed  "
        f"aggregate_score={aggregate_score:.3f}"
    )
    print("=" * 60)


if __name__ == "__main__":
    run()
