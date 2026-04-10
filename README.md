---
title: Context Aware Env
emoji: 🧠
colorFrom: gray
colorTo: indigo
sdk: docker
app_port: 8000
pinned: false
---

# ContextAwareEnv — Social Awareness RL Environment

> **Meta × Scaler OpenEnv Hackathon — Phase 3 Submission**

A fully OpenEnv-compliant reinforcement learning environment that evaluates whether an LLM-powered desktop assistant possesses **social awareness** — specifically, knowing *when to stay silent* and *when to proactively interrupt* to help.

---

## 🧠 Core Concept

Modern AI assistants must be more than just capable — they must be **socially intelligent**. An assistant that interrupts a developer in deep focus is just as problematic as one that ignores a user silently struggling with an error.

**ContextAwareEnv** simulates a user's desktop operating system. The agent observes the screen state — active application, visible text, user telemetry, and explicit help requests — and must choose the socially-appropriate action at each timestep.

### Three Difficulty Tiers

| Task       | Scenario                                                                           | Optimal Action                                   |
|------------|------------------------------------------------------------------------------------|--------------------------------------------------|
| **Easy**   | User watches a YouTube tutorial and *explicitly asks for help*                     | `summarize_screen` — respond to the direct request |
| **Medium** | User is coding in VS Code with focused typing (*deep work mode*)                  | `stay_silent` × 5 consecutive turns               |
| **Hard**   | Terminal shows `npm ERR!` while user has erratic mouse movement (*silent frustration*) | `proactive_help` — mention the specific error      |

### Scoring Rubric

All rewards are **continuous values in the open interval (0, 1)** — never exactly 0.0 or 1.0. This ensures meaningful gradient signal for RL training and satisfies the evaluation platform's strict score constraints.

| Action                              | Easy           | Medium (per turn) | Hard               |
|-------------------------------------|----------------|--------------------|--------------------| 
| `summarize_screen`                  | 0.4–0.99       | 0.0 (instant fail) | 0.0                |
| `stay_silent`                       | 0.0            | 0.14–0.24          | 0.0                |
| `proactive_help` + specific error   | 0.0            | 0.0 (instant fail) | 0.6–0.99           |
| `proactive_help` + generic payload  | 0.0            | 0.0 (instant fail) | 0.4–0.6 (partial)  |

**How scoring works per tier:**

- **Easy** — Single-step. Reward is `base (0.4) + fuzzy similarity to screen context (up to 0.4) + length quality bonus (0.2)`. Summarising with relevant content scores high; wrong action scores 0.
- **Medium** — Multi-step (5 turns). Each silent turn earns a small reward that increases with each turn (~0.14, 0.17, 0.19, 0.21, 0.24). The **task score is the sum** of all per-turn rewards, designed to reach ~0.95 for 5 correct turns. Any interruption immediately ends the episode with zero reward.
- **Hard** — Single-step. Reward is `base (0.4) + fuzzy match to error description (up to 0.4) + keyword bonuses (up to 0.2 for mentioning "npm" and "error")`. Proactive help that references the specific error scores highest.

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    inference.py                          │
│  ┌──────────┐    ┌────────────┐    ┌──────────────────┐ │
│  │ LLM API  │◄──►│  Agent     │    │  Structured      │ │
│  │ (HF/OAI) │    │  Loop      │───►│  Log Output      │ │
│  └──────────┘    └─────┬──────┘    └──────────────────┘ │
│                        │ WebSocket                       │
└────────────────────────┼────────────────────────────────┘
                         │
┌────────────────────────┼────────────────────────────────┐
│                  FastAPI Server (app.py)                  │
│                        │                                 │
│              ┌─────────▼─────────┐                       │
│              │  ContextAwareEnv  │                       │
│              │  (environment.py) │                       │
│              │                   │                       │
│              │  • Task presets   │                       │
│              │  • Reward logic   │                       │
│              │  • Fuzzy scoring  │                       │
│              └───────────────────┘                       │
│                                                          │
│  Endpoints: GET /health, GET /, WS /ws                   │
└──────────────────────────────────────────────────────────┘
```

The inference script manages the full lifecycle:

1. **Server startup** — Launches FastAPI as a subprocess (or connects to a remote HF Space).
2. **WebSocket episodes** — Each task runs as a WebSocket session, ensuring `reset()` and `step()` share the same environment instance.
3. **LLM interaction** — Queries an OpenAI-compatible API with exponential-backoff retry logic.
4. **Defensive parsing** — Six strategies (direct JSON → markdown stripping → regex extraction → nested-object unwrapping → keyword detection → safe fallback) guarantee a valid action from any LLM output.
5. **Structured logging** — Every step emits parseable `[START]`, `[STEP]`, `[END]` lines for the automated evaluation pipeline.

---

## 📦 Installation

### Prerequisites

- Python 3.10+
- pip or uv package manager

### Install dependencies

```bash
pip install -r requirements.txt
```

### Install as editable package (recommended for development)

```bash
pip install -e .
```

---

## 🚀 Quick Start

### 1. Start the Server

```bash
# From the context_aware_env/ directory:
python -m uvicorn server.app:app --host 0.0.0.0 --port 8000
```

The server exposes:
- **`GET /health`** — Liveness probe → `{"status": "healthy"}`
- **`GET /`** — Service metadata (version, available tasks, endpoints)
- **`WS /ws`** — WebSocket endpoint for environment interaction

### 2. Use the Client (Python)

```python
import asyncio
from client import ContextEnvClient
from models import ContextAction

async def main():
    async with ContextEnvClient(base_url="http://localhost:8000") as env:
        # Reset to the "easy" task
        result = await env.reset(task_name="easy")
        print(result.observation)

        # Take the correct action
        action = ContextAction(action_type="summarize_screen")
        result = await env.step(action)
        print(f"Reward: {result.reward}, Done: {result.done}")

asyncio.run(main())
```

### 3. Run Inference (Automated Evaluation)

The inference script starts its own server, runs all three task tiers against an LLM, and prints structured logs:

```bash
# Set environment variables for the LLM API
export API_BASE_URL="https://router.huggingface.co/v1"
export MODEL_NAME="Qwen/Qwen2.5-72B-Instruct"
export HF_TOKEN="your-huggingface-token"

# Run evaluation
python inference.py
```

#### `--no-llm` mode

For testing without an LLM API key, the script supports hardcoded optimal actions:

```bash
python inference.py --no-llm
```

#### Inference Capabilities

- **Exponential-backoff retry** — 3 attempts with 1s → 2s → 4s delays on transient errors (429, 503)
- **Non-retryable error detection** — 401/403 auth errors skip retry and fail fast
- **Defensive JSON parsing** — handles markdown fences, nested response objects, keyword fallback
- **Single asyncio event loop** for all tasks (no deprecation warnings on Python 3.12+)
- **Graceful Ctrl+C handling** with proper server cleanup
- **Summary report** at the end with per-tier pass/fail and aggregate score

---

## 🧪 Running Tests

Start the server, then run the comprehensive test suite:

```bash
# Terminal 1: Start server
python -m uvicorn server.app:app --host 0.0.0.0 --port 8000

# Terminal 2: Run tests (9 tests: 3 positive + 4 negative + 2 endpoint)
python test_endpoints.py
```

Expected output:

```
============================================================
  CONTEXT-AWARE-ENV  TEST SUITE
============================================================

  ✓ PASS  easy_correct_action        (reward=0.89)
  ✓ PASS  medium_correct_actions     (total_reward=0.95)
  ✓ PASS  hard_correct_action        (reward=0.72)
  ✓ PASS  easy_wrong_action          (reward=0.01)
  ✓ PASS  medium_interrupted         (interruption correctly penalised)
  ✓ PASS  hard_generic_payload       (reward=0.50)
  ✓ PASS  hard_wrong_action          (reward=0.01)
  ✓ PASS  health_endpoint
  ✓ PASS  root_endpoint

------------------------------------------------------------
  9/9 tests passed — ALL PASSED
------------------------------------------------------------
```

---

## 📝 Log Format

The `inference.py` script outputs structured lines for the automated evaluation parser:

```
[START] task=easy_1 env=local model=Qwen/Qwen2.5-72B-Instruct
[STEP] step=1 action=summarize_screen(task=easy) reward=0.89 done=true error=null
[END] success=true steps=1 score=0.891 rewards=0.89
```

After all tasks complete, a summary report is printed:

```
============================================================
  INFERENCE SUMMARY
============================================================
  easy_1      PASS  score=0.891  steps=1  rewards=0.89
  medium_1    PASS  score=0.950  steps=5  rewards=0.14,0.17,0.19,0.21,0.24
  hard_1      PASS  score=0.715  steps=1  rewards=0.72
------------------------------------------------------------
  TOTAL    3/3 passed  aggregate_score=0.852
============================================================
```

> **Note:** Medium score is the **sum** of per-turn rewards (not the average), since each turn earns a fraction designed to total ~0.95 over 5 correct turns. Easy and hard scores equal their single-step reward directly.

---

## 🐳 Docker

### Build and Run

```bash
docker build -t context-aware-env .
docker run -p 8000:8000 context-aware-env
```

### Docker Features

- **Multi-stage build** for smaller image size
- **Non-root user** (`appuser`) for security
- **Built-in `HEALTHCHECK`** for auto-restart on failure
- **`.dockerignore`** excludes `.git`, `__pycache__`, `uv.lock`, tests, and docs

---

## 📋 OpenEnv Compliance

This environment is fully compliant with the OpenEnv specification:

```bash
openenv validate
```

**Compliance checklist:**

- [x] `openenv.yaml` manifest with correct env name and entry points
- [x] `reset(task_name)` / `step(action)` API via WebSocket
- [x] All scores strictly in the open interval (0, 1) — never 0.0 or 1.0
- [x] Structured `[START]`, `[STEP]`, `[END]` log lines in `inference.py`
- [x] Health endpoint at `GET /health`
- [x] Docker deployment with `HEALTHCHECK`
- [x] No hardcoded secrets in source code

---

## 📁 Project Structure

```
context_aware_env/
├── __init__.py            # Package re-exports & metadata
├── models.py              # Pydantic data contracts (Action, Observation, State)
├── client.py              # Typed async WebSocket client
├── inference.py           # LLM evaluation script (retry, defensive parsing, logging)
├── test_endpoints.py      # Comprehensive test suite (9 tests)
├── openenv.yaml           # OpenEnv manifest
├── pyproject.toml         # Python project configuration
├── requirements.txt       # Pinned dependencies
├── Dockerfile             # Multi-stage container (non-root, HEALTHCHECK)
├── .gitignore             # Git exclusions
├── .dockerignore          # Docker build exclusions
├── README.md              # This file
├── project_documentation.md  # Detailed technical documentation
└── server/
    ├── __init__.py        # Server sub-package
    ├── environment.py     # Core RL environment (task presets, reward logic)
    └── app.py             # FastAPI server (health, CORS, WebSocket)
```

---

## 🔧 Environment Variables

| Variable              | Default                                  | Description                                  |
|-----------------------|------------------------------------------|----------------------------------------------|
| `API_BASE_URL`        | `https://router.huggingface.co/v1`       | OpenAI-compatible API endpoint               |
| `MODEL_NAME`          | `Qwen/Qwen2.5-72B-Instruct`             | Model identifier for chat completions        |
| `HF_TOKEN`            | *(empty)*                                | HuggingFace API token                        |
| `ENV_SERVER_URL`      | *(remote HF Space)*                      | Use a deployed environment instead of local  |
| `SERVER_WAIT_TIMEOUT` | `30`                                     | Seconds to wait for server health check      |

> ⚠️ **Security:** Never hardcode `HF_TOKEN` in source code. Always set it as an environment variable. If you accidentally committed a token, **revoke it immediately** at https://huggingface.co/settings/tokens.

---

## 🛡️ Robustness & Safety

| Feature                        | Description                                                              |
|--------------------------------|--------------------------------------------------------------------------|
| **Strict Pydantic models**     | `extra="forbid"` catches typos in field names immediately                |
| **Auto-sanitised payloads**    | `model_validator` clears payload when `action_type="stay_silent"`        |
| **Step ceiling**               | `MAX_STEPS_PER_EPISODE = 20` prevents runaway agents                    |
| **Episode-done guard**         | Returns zero-reward if `step()` is called after episode ends             |
| **Task validation**            | `ValueError` on invalid task names (no silent fall-through)              |
| **Non-retryable detection**    | 401/403 errors skip retry loop (fail fast, save time)                    |
| **6-strategy defensive parse** | Guarantees a valid action from any LLM output, including garbage         |
| **Token warning**              | Clear setup instructions printed when `HF_TOKEN` is missing              |
| **Score clamping**             | All rewards clamped to (0, 1) to satisfy evaluation platform constraints |

---

## 🚢 Deployment

### HuggingFace Spaces (recommended)

```bash
pip install -U "huggingface_hub[cli]"
hf auth login
hf upload YOUR_HF_USERNAME/context-aware-env . . --repo-type space
```

### Docker Hub

```bash
docker build -t context-aware-env .
docker tag context-aware-env YOUR_DOCKERHUB/context-aware-env:latest
docker push YOUR_DOCKERHUB/context-aware-env:latest
```

### GitHub

```bash
git add .
git commit -m "feat: ContextAwareEnv v1.0.0"
git remote add origin https://github.com/YOUR_USERNAME/context-aware-env.git
git push -u origin main
```

📖 See [`project_documentation.md`](project_documentation.md) for detailed deployment guides and code walkthroughs.

---

## License

MIT
