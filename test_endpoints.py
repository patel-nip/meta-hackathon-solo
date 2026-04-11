"""
Smoke + Negative tests: WebSocket-based episode runs for all three tasks.
=========================================================================

These tests verify:
  ✓ Correct behaviour for each task tier (easy, medium, hard)
  ✓ Wrong actions yield zero reward
  ✓ Edge cases (stepping after done, partial medium completion)
  ✓ Reset produces clean state
  ✓ HTTP endpoints respond correctly

Usage
-----
1. Start the server:

   .. code-block:: bash

       python -m uvicorn server.app:app --host 0.0.0.0 --port 8000

2. Run the tests:

   .. code-block:: bash

       python test_endpoints.py
"""

from __future__ import annotations

import asyncio
import json
import sys
import traceback

# ---------------------------------------------------------------------------
# Import models — try proper package path first, fall back to local.
# ---------------------------------------------------------------------------
try:
    from context_aware_env.models import ContextAction, ContextObservation
except ImportError:
    sys.path.insert(0, ".")
    from models import ContextAction, ContextObservation  # type: ignore[no-redef]


# ── Configuration ─────────────────────────────────────────────────────────────

WS_URL = "ws://localhost:8000/ws"
WS_TIMEOUT = 10  # seconds


# ── Helpers ───────────────────────────────────────────────────────────────────

def parse_obs(resp_data: dict) -> ContextObservation:
    """Parse the OpenEnv WebSocket response format.

    OpenEnv serialises observations as::

        {"observation": {...fields...}, "reward": float, "done": bool}
    """
    obs_fields = dict(resp_data.get("observation", {}))
    obs_fields["reward"] = resp_data.get("reward", 0.0)
    obs_fields["done"] = resp_data.get("done", False)
    return ContextObservation(**obs_fields)


# ── Result tracker ────────────────────────────────────────────────────────────

_results: list[tuple[str, bool, str]] = []
"""List of (test_name, passed, detail) for the summary report."""


def _record(name: str, passed: bool, detail: str = "") -> None:
    _results.append((name, passed, detail))
    status = "  PASS" if passed else "* FAIL"
    msg = f"{status}  {name}"
    if detail:
        msg += f"  ({detail})"
    print(msg)


# ═══════════════════════════════════════════════════════════════════════════════
#  POSITIVE TESTS
# ═══════════════════════════════════════════════════════════════════════════════


async def test_easy_correct_action():
    """Easy task: summarize_screen → reward in [0.40, 0.99], done=True."""
    import websockets

    test_name = "easy_correct_action"
    async with websockets.connect(WS_URL, open_timeout=WS_TIMEOUT) as ws:
        # RESET
        await ws.send(json.dumps({"type": "reset", "data": {"task_name": "easy"}}))
        resp = json.loads(await asyncio.wait_for(ws.recv(), timeout=WS_TIMEOUT))
        assert resp["type"] == "observation", f"Expected observation, got {resp['type']}"
        obs = parse_obs(resp["data"])
        assert obs.explicit_help_request is True, "Easy task should have help request"
        # Verify new observation fields
        assert obs.mouse_activity == "stationary", f"Expected stationary, got {obs.mouse_activity}"
        assert obs.recent_keystrokes_per_minute == 0
        assert obs.error_count == 0

        # STEP — correct action with relevant keywords
        action = ContextAction(
            action_type="summarize_screen",
            payload="The user is watching a React tutorial video on YouTube",
        )
        await ws.send(json.dumps({"type": "step", "data": action.model_dump()}))
        resp = json.loads(await asyncio.wait_for(ws.recv(), timeout=WS_TIMEOUT))
        obs = parse_obs(resp["data"])

        # Continuous scoring: base(0.40) + similarity*0.30 + keywords(up to 0.15) + length(0.15)
        assert 0.40 <= obs.reward <= 0.99, f"Expected reward in [0.40, 0.99], got {obs.reward}"
        assert obs.done is True
        _record(test_name, True, f"reward={obs.reward:.3f}")


async def test_medium_correct_actions():
    """Medium task: 5x stay_silent → variable per-turn rewards, done=True."""
    import websockets

    test_name = "medium_correct_actions"
    async with websockets.connect(WS_URL, open_timeout=WS_TIMEOUT) as ws:
        await ws.send(json.dumps({"type": "reset", "data": {"task_name": "medium"}}))
        resp = json.loads(await asyncio.wait_for(ws.recv(), timeout=WS_TIMEOUT))
        assert resp["type"] == "observation"
        obs = parse_obs(resp["data"])
        assert obs.explicit_help_request is False
        # Verify new observation fields for medium
        assert obs.mouse_activity == "minimal"
        assert obs.recent_keystrokes_per_minute == 45
        assert obs.error_count == 0

        total_reward = 0.0
        # Per-turn rewards are dynamically computed from context analysis
        for i in range(5):
            action = ContextAction(action_type="stay_silent", payload="")
            await ws.send(json.dumps({"type": "step", "data": action.model_dump()}))
            resp = json.loads(await asyncio.wait_for(ws.recv(), timeout=WS_TIMEOUT))
            obs = parse_obs(resp["data"])
            assert 0.10 <= obs.reward <= 0.40, f"Step {i+1}: Expected reward in [0.10, 0.40], got {obs.reward}"
            total_reward += obs.reward

        assert obs.done is True
        assert 0.85 <= total_reward <= 1.25, f"Total reward {total_reward} not in expected range [0.85, 1.25]"
        _record(test_name, True, f"total_reward={total_reward:.2f}")


async def test_hard_correct_action():
    """Hard task: proactive_help with specific error details → high reward."""
    import websockets

    test_name = "hard_correct_action"
    async with websockets.connect(WS_URL, open_timeout=WS_TIMEOUT) as ws:
        await ws.send(json.dumps({"type": "reset", "data": {"task_name": "hard"}}))
        resp = json.loads(await asyncio.wait_for(ws.recv(), timeout=WS_TIMEOUT))
        assert resp["type"] == "observation"
        obs = parse_obs(resp["data"])
        assert obs.user_telemetry == "erratic_mouse"
        # Verify new observation fields for hard
        assert obs.mouse_activity == "erratic_clicking"
        assert obs.recent_keystrokes_per_minute == 5
        assert obs.error_count == 3

        # Action that references the specific error details
        action = ContextAction(
            action_type="proactive_help",
            payload=(
                "Your npm build failed with an ELIFECYCLE error. "
                "The module './components/Dashboard' cannot be resolved "
                "— check that the component file exists and the import path is correct."
            ),
        )
        await ws.send(json.dumps({"type": "step", "data": action.model_dump()}))
        resp = json.loads(await asyncio.wait_for(ws.recv(), timeout=WS_TIMEOUT))
        obs = parse_obs(resp["data"])

        # Multi-signal scoring: base(0.35) + sim*0.25 + critical_kwd(up to 0.20) + specificity(up to 0.15)
        assert 0.65 <= obs.reward <= 0.99, f"Expected reward in [0.65, 0.99], got {obs.reward}"
        assert obs.done is True
        _record(test_name, True, f"reward={obs.reward:.3f}")


# ═══════════════════════════════════════════════════════════════════════════════
#  NEGATIVE / EDGE-CASE TESTS
# ═══════════════════════════════════════════════════════════════════════════════


async def test_easy_wrong_action():
    """Easy task: stay_silent (wrong) → reward=0.01 (clamped), done=True."""
    import websockets

    test_name = "easy_wrong_action"
    async with websockets.connect(WS_URL, open_timeout=WS_TIMEOUT) as ws:
        await ws.send(json.dumps({"type": "reset", "data": {"task_name": "easy"}}))
        resp = json.loads(await asyncio.wait_for(ws.recv(), timeout=WS_TIMEOUT))
        obs = parse_obs(resp["data"])

        action = ContextAction(action_type="stay_silent", payload="")
        await ws.send(json.dumps({"type": "step", "data": action.model_dump()}))
        resp = json.loads(await asyncio.wait_for(ws.recv(), timeout=WS_TIMEOUT))
        obs = parse_obs(resp["data"])

        # Wrong action: 0.0 clamped to 0.01 (SCORE_EPSILON)
        assert obs.reward == 0.01, f"Expected reward 0.01 (clamped), got {obs.reward}"
        assert obs.done is True
        _record(test_name, True, f"reward={obs.reward}")


async def test_medium_interrupted():
    """Medium task: 2x stay_silent then proactive_help → reward=0.01 (clamped) on fail."""
    import websockets

    test_name = "medium_interrupted"
    async with websockets.connect(WS_URL, open_timeout=WS_TIMEOUT) as ws:
        await ws.send(json.dumps({"type": "reset", "data": {"task_name": "medium"}}))
        resp = json.loads(await asyncio.wait_for(ws.recv(), timeout=WS_TIMEOUT))

        # Two successful silent turns (dynamically context-scored rewards)
        for i in range(2):
            action = ContextAction(action_type="stay_silent", payload="")
            await ws.send(json.dumps({"type": "step", "data": action.model_dump()}))
            resp = json.loads(await asyncio.wait_for(ws.recv(), timeout=WS_TIMEOUT))
            obs = parse_obs(resp["data"])
            assert 0.10 <= obs.reward <= 0.40, f"Step {i+1}: reward {obs.reward} not in range"

        # Interrupt with proactive_help → should fail
        action = ContextAction(action_type="proactive_help", payload="need help?")
        await ws.send(json.dumps({"type": "step", "data": action.model_dump()}))
        resp = json.loads(await asyncio.wait_for(ws.recv(), timeout=WS_TIMEOUT))
        obs = parse_obs(resp["data"])

        # Wrong action: 0.0 clamped to 0.01
        assert obs.reward == 0.01, f"Expected 0.01 on interrupt, got {obs.reward}"
        assert obs.done is True
        _record(test_name, True, "interruption correctly penalised")


async def test_hard_generic_payload():
    """Hard task: proactive_help but generic payload → partial reward."""
    import websockets

    test_name = "hard_generic_payload"
    async with websockets.connect(WS_URL, open_timeout=WS_TIMEOUT) as ws:
        await ws.send(json.dumps({"type": "reset", "data": {"task_name": "hard"}}))
        resp = json.loads(await asyncio.wait_for(ws.recv(), timeout=WS_TIMEOUT))

        action = ContextAction(
            action_type="proactive_help",
            payload="It looks like you might need some help.",
        )
        await ws.send(json.dumps({"type": "step", "data": action.model_dump()}))
        resp = json.loads(await asyncio.wait_for(ws.recv(), timeout=WS_TIMEOUT))
        obs = parse_obs(resp["data"])

        # Generic payload: base(0.35) + minimal similarity + no keyword bonuses → ~0.35-0.45
        assert 0.30 <= obs.reward <= 0.50, f"Expected reward in [0.30, 0.50], got {obs.reward}"
        assert obs.done is True
        _record(test_name, True, f"reward={obs.reward:.3f}")


async def test_hard_wrong_action():
    """Hard task: stay_silent (wrong) → reward=0.01 (clamped)."""
    import websockets

    test_name = "hard_wrong_action"
    async with websockets.connect(WS_URL, open_timeout=WS_TIMEOUT) as ws:
        await ws.send(json.dumps({"type": "reset", "data": {"task_name": "hard"}}))
        resp = json.loads(await asyncio.wait_for(ws.recv(), timeout=WS_TIMEOUT))

        action = ContextAction(action_type="stay_silent", payload="")
        await ws.send(json.dumps({"type": "step", "data": action.model_dump()}))
        resp = json.loads(await asyncio.wait_for(ws.recv(), timeout=WS_TIMEOUT))
        obs = parse_obs(resp["data"])

        # Wrong action: 0.0 clamped to 0.01
        assert obs.reward == 0.01, f"Expected 0.01 (clamped), got {obs.reward}"
        assert obs.done is True
        _record(test_name, True, f"reward={obs.reward}")


# ═══════════════════════════════════════════════════════════════════════════════
#  STATE MANAGEMENT TESTS
# ═══════════════════════════════════════════════════════════════════════════════


async def test_reset_clean_state():
    """Verify that reset() fully clears state from a previous episode."""
    import websockets

    test_name = "reset_clean_state"
    async with websockets.connect(WS_URL, open_timeout=WS_TIMEOUT) as ws:
        # Run a complete easy episode
        await ws.send(json.dumps({"type": "reset", "data": {"task_name": "easy"}}))
        resp = json.loads(await asyncio.wait_for(ws.recv(), timeout=WS_TIMEOUT))
        obs = parse_obs(resp["data"])
        assert obs.explicit_help_request is True  # easy task

        action = ContextAction(action_type="summarize_screen", payload="summary")
        await ws.send(json.dumps({"type": "step", "data": action.model_dump()}))
        resp = json.loads(await asyncio.wait_for(ws.recv(), timeout=WS_TIMEOUT))
        obs = parse_obs(resp["data"])
        assert obs.done is True  # episode finished

        # Now reset to medium — state should be completely fresh
        await ws.send(json.dumps({"type": "reset", "data": {"task_name": "medium"}}))
        resp = json.loads(await asyncio.wait_for(ws.recv(), timeout=WS_TIMEOUT))
        obs = parse_obs(resp["data"])

        assert obs.done is False, "Fresh episode should not be done"
        assert obs.reward == 0.0, "Fresh episode should have zero reward"
        assert obs.explicit_help_request is False, "Medium task: no help request"
        assert obs.error_count == 0, "Medium task: no errors"

        # First silent turn should work normally
        action = ContextAction(action_type="stay_silent", payload="")
        await ws.send(json.dumps({"type": "step", "data": action.model_dump()}))
        resp = json.loads(await asyncio.wait_for(ws.recv(), timeout=WS_TIMEOUT))
        obs = parse_obs(resp["data"])
        assert 0.10 <= obs.reward <= 0.40, f"First turn reward {obs.reward} not in range"
        assert obs.done is False, "Medium: should not be done after 1 turn"

        _record(test_name, True, "state fully cleared between episodes")


async def test_step_after_done():
    """Verify that stepping after an episode ends returns zero reward."""
    import websockets

    test_name = "step_after_done"
    async with websockets.connect(WS_URL, open_timeout=WS_TIMEOUT) as ws:
        # Complete an easy episode
        await ws.send(json.dumps({"type": "reset", "data": {"task_name": "easy"}}))
        resp = json.loads(await asyncio.wait_for(ws.recv(), timeout=WS_TIMEOUT))

        action = ContextAction(action_type="summarize_screen", payload="summary here")
        await ws.send(json.dumps({"type": "step", "data": action.model_dump()}))
        resp = json.loads(await asyncio.wait_for(ws.recv(), timeout=WS_TIMEOUT))
        obs = parse_obs(resp["data"])
        assert obs.done is True

        # Try stepping again — should get zero reward, done=True
        await ws.send(json.dumps({"type": "step", "data": action.model_dump()}))
        resp = json.loads(await asyncio.wait_for(ws.recv(), timeout=WS_TIMEOUT))
        obs = parse_obs(resp["data"])
        assert obs.done is True, "Should still be done"
        assert obs.reward == 0.0, f"Expected 0.0 reward after done, got {obs.reward}"

        _record(test_name, True, "zero reward returned after episode end")


# ═══════════════════════════════════════════════════════════════════════════════
#  ENDPOINT TESTS
# ═══════════════════════════════════════════════════════════════════════════════


async def test_health_endpoint():
    """GET /health should return {"status": "healthy"}."""
    import urllib.request

    test_name = "health_endpoint"
    req = urllib.request.Request("http://localhost:8000/health")
    with urllib.request.urlopen(req, timeout=5) as resp:
        assert resp.status == 200
        data = json.loads(resp.read())
        assert data["status"] in ("ok", "healthy"), f"Got status: {data['status']}"
    _record(test_name, True)


async def test_root_endpoint():
    """GET / should return service metadata."""
    import urllib.request

    test_name = "root_endpoint"
    req = urllib.request.Request("http://localhost:8000/")
    with urllib.request.urlopen(req, timeout=5) as resp:
        assert resp.status == 200
        data = json.loads(resp.read())
        assert data["service"] == "ContextAwareEnv"
        assert "tasks" in data
    _record(test_name, True)


# ═══════════════════════════════════════════════════════════════════════════════
#  TEST RUNNER
# ═══════════════════════════════════════════════════════════════════════════════


ALL_TESTS = [
    # Positive tests
    test_easy_correct_action,
    test_medium_correct_actions,
    test_hard_correct_action,
    # Negative / edge-case tests
    test_easy_wrong_action,
    test_medium_interrupted,
    test_hard_generic_payload,
    test_hard_wrong_action,
    # State management tests
    test_reset_clean_state,
    test_step_after_done,
    # Endpoint tests
    test_health_endpoint,
    test_root_endpoint,
]


def _check_server() -> bool:
    """Pre-flight check: verify the server is reachable before running tests."""
    import urllib.request
    import urllib.error

    try:
        req = urllib.request.Request("http://localhost:8000/health")
        with urllib.request.urlopen(req, timeout=3) as resp:
            if resp.status == 200:
                return True
    except (urllib.error.URLError, OSError):
        pass
    return False


async def main():
    """Run all tests, catching failures individually so one bad test
    doesn't block the rest."""
    print("\n" + "=" * 60)
    print("  CONTEXT-AWARE-ENV  TEST SUITE")
    print("=" * 60 + "\n")

    # ── Pre-flight: ensure server is running ──────────────────────────
    if not _check_server():
        print("  ERROR: Server is not running at http://localhost:8000")
        print("")
        print("  Start the server first:")
        print("    python -m uvicorn server.app:app --host 0.0.0.0 --port 8000")
        print("")
        print("-" * 60)
        print("  0/0 tests run — SERVER UNREACHABLE")
        print("-" * 60 + "\n")
        sys.exit(1)

    for test_fn in ALL_TESTS:
        try:
            await test_fn()
        except Exception as exc:
            _record(
                test_fn.__name__.replace("test_", ""),
                False,
                f"{type(exc).__name__}: {exc}",
            )
            traceback.print_exc(file=sys.stderr)

    # ── Summary ───────────────────────────────────────────────────────────
    print("\n" + "-" * 60)
    passed = sum(1 for _, ok, _ in _results if ok)
    total = len(_results)
    status = "ALL PASSED" if passed == total else f"{total - passed} FAILED"
    print(f"  {passed}/{total} tests passed — {status}")
    print("-" * 60 + "\n")

    if passed < total:
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
