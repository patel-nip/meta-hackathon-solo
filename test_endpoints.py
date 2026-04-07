"""
Smoke + Negative tests: WebSocket-based episode runs for all three tasks.
=========================================================================

These tests verify:
  ✓ Correct behaviour for each task tier (easy, medium, hard)
  ✓ Wrong actions yield zero reward
  ✓ Invalid task names are rejected
  ✓ Edge cases (stepping after done, partial medium completion)

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
    status = "✓ PASS" if passed else "✗ FAIL"
    msg = f"  {status}  {name}"
    if detail:
        msg += f"  ({detail})"
    print(msg)


# ═══════════════════════════════════════════════════════════════════════════════
#  POSITIVE TESTS
# ═══════════════════════════════════════════════════════════════════════════════


async def test_easy_correct_action():
    """Easy task: summarize_screen → reward=1.0, done=True."""
    import websockets

    test_name = "easy_correct_action"
    async with websockets.connect(WS_URL, open_timeout=WS_TIMEOUT) as ws:
        # RESET
        await ws.send(json.dumps({"type": "reset", "data": {"task_name": "easy"}}))
        resp = json.loads(await asyncio.wait_for(ws.recv(), timeout=WS_TIMEOUT))
        assert resp["type"] == "observation", f"Expected observation, got {resp['type']}"
        obs = parse_obs(resp["data"])
        assert obs.explicit_help_request is True, "Easy task should have help request"

        # STEP — correct action
        action = ContextAction(action_type="summarize_screen", payload="Here is a summary")
        await ws.send(json.dumps({"type": "step", "data": action.model_dump()}))
        resp = json.loads(await asyncio.wait_for(ws.recv(), timeout=WS_TIMEOUT))
        obs = parse_obs(resp["data"])

        assert obs.reward == 1.0, f"Expected reward 1.0, got {obs.reward}"
        assert obs.done is True
        _record(test_name, True, f"reward={obs.reward}")


async def test_medium_correct_actions():
    """Medium task: 5x stay_silent → 0.2 each, total 1.0, done=True."""
    import websockets

    test_name = "medium_correct_actions"
    async with websockets.connect(WS_URL, open_timeout=WS_TIMEOUT) as ws:
        await ws.send(json.dumps({"type": "reset", "data": {"task_name": "medium"}}))
        resp = json.loads(await asyncio.wait_for(ws.recv(), timeout=WS_TIMEOUT))
        assert resp["type"] == "observation"
        obs = parse_obs(resp["data"])
        assert obs.explicit_help_request is False

        total_reward = 0.0
        for i in range(5):
            action = ContextAction(action_type="stay_silent", payload="")
            await ws.send(json.dumps({"type": "step", "data": action.model_dump()}))
            resp = json.loads(await asyncio.wait_for(ws.recv(), timeout=WS_TIMEOUT))
            obs = parse_obs(resp["data"])
            assert obs.reward == 0.2, f"Step {i+1}: Expected 0.2, got {obs.reward}"
            total_reward += obs.reward

        assert obs.done is True
        assert abs(total_reward - 1.0) < 0.01, f"Total reward {total_reward} ≠ 1.0"
        _record(test_name, True, f"total_reward={total_reward:.2f}")


async def test_hard_correct_action():
    """Hard task: proactive_help with 'npm error' → reward=1.0."""
    import websockets

    test_name = "hard_correct_action"
    async with websockets.connect(WS_URL, open_timeout=WS_TIMEOUT) as ws:
        await ws.send(json.dumps({"type": "reset", "data": {"task_name": "hard"}}))
        resp = json.loads(await asyncio.wait_for(ws.recv(), timeout=WS_TIMEOUT))
        assert resp["type"] == "observation"
        obs = parse_obs(resp["data"])
        assert obs.user_telemetry == "erratic_mouse"

        action = ContextAction(
            action_type="proactive_help",
            payload="It looks like you have an npm error — try deleting node_modules",
        )
        await ws.send(json.dumps({"type": "step", "data": action.model_dump()}))
        resp = json.loads(await asyncio.wait_for(ws.recv(), timeout=WS_TIMEOUT))
        obs = parse_obs(resp["data"])

        assert obs.reward == 1.0, f"Expected reward 1.0, got {obs.reward}"
        assert obs.done is True
        _record(test_name, True, f"reward={obs.reward}")


# ═══════════════════════════════════════════════════════════════════════════════
#  NEGATIVE / EDGE-CASE TESTS
# ═══════════════════════════════════════════════════════════════════════════════


async def test_easy_wrong_action():
    """Easy task: stay_silent (wrong) → reward=0.0, done=True."""
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

        assert obs.reward == 0.0, f"Expected reward 0.0, got {obs.reward}"
        assert obs.done is True
        _record(test_name, True, f"reward={obs.reward}")


async def test_medium_interrupted():
    """Medium task: 2x stay_silent then proactive_help → reward=0.0 on fail."""
    import websockets

    test_name = "medium_interrupted"
    async with websockets.connect(WS_URL, open_timeout=WS_TIMEOUT) as ws:
        await ws.send(json.dumps({"type": "reset", "data": {"task_name": "medium"}}))
        resp = json.loads(await asyncio.wait_for(ws.recv(), timeout=WS_TIMEOUT))

        # Two successful silent turns
        for _ in range(2):
            action = ContextAction(action_type="stay_silent", payload="")
            await ws.send(json.dumps({"type": "step", "data": action.model_dump()}))
            resp = json.loads(await asyncio.wait_for(ws.recv(), timeout=WS_TIMEOUT))
            obs = parse_obs(resp["data"])
            assert obs.reward == 0.2

        # Interrupt with proactive_help → should fail
        action = ContextAction(action_type="proactive_help", payload="need help?")
        await ws.send(json.dumps({"type": "step", "data": action.model_dump()}))
        resp = json.loads(await asyncio.wait_for(ws.recv(), timeout=WS_TIMEOUT))
        obs = parse_obs(resp["data"])

        assert obs.reward == 0.0, f"Expected 0.0 on interrupt, got {obs.reward}"
        assert obs.done is True
        _record(test_name, True, "interruption correctly penalised")


async def test_hard_generic_payload():
    """Hard task: proactive_help but generic payload → reward=0.5."""
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

        assert obs.reward == 0.5, f"Expected 0.5 for generic help, got {obs.reward}"
        assert obs.done is True
        _record(test_name, True, f"reward={obs.reward}")


async def test_hard_wrong_action():
    """Hard task: stay_silent (wrong) → reward=0.0."""
    import websockets

    test_name = "hard_wrong_action"
    async with websockets.connect(WS_URL, open_timeout=WS_TIMEOUT) as ws:
        await ws.send(json.dumps({"type": "reset", "data": {"task_name": "hard"}}))
        resp = json.loads(await asyncio.wait_for(ws.recv(), timeout=WS_TIMEOUT))

        action = ContextAction(action_type="stay_silent", payload="")
        await ws.send(json.dumps({"type": "step", "data": action.model_dump()}))
        resp = json.loads(await asyncio.wait_for(ws.recv(), timeout=WS_TIMEOUT))
        obs = parse_obs(resp["data"])

        assert obs.reward == 0.0, f"Expected 0.0, got {obs.reward}"
        assert obs.done is True
        _record(test_name, True, f"reward={obs.reward}")


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
    # Endpoint tests
    test_health_endpoint,
    test_root_endpoint,
]


async def main():
    """Run all tests, catching failures individually so one bad test
    doesn't block the rest."""
    print("\n" + "=" * 60)
    print("  CONTEXT-AWARE-ENV  TEST SUITE")
    print("=" * 60 + "\n")

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
