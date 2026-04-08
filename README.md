---
title: Context Aware Env
emoji: 🧠
colorFrom: gray
colorTo: indigo
sdk: docker
app_port: 8000
pinned: false
---

# ContextAwareEnv – Social Awareness RL Environment

> **Meta × Scaler OpenEnv Hackathon**

An OpenEnv-compliant reinforcement learning environment that evaluates whether
an LLM agent possesses **social awareness** — specifically, knowing when to
**stay silent** and when to **proactively interrupt** to help.

---

## 🧠 Concept

The environment simulates a user's desktop operating system. The agent observes
the screen state (active app, visible text, user telemetry) and must decide on
the socially-appropriate action.

### Three Difficulty Tiers

| Task       | Scenario                                                                    | Optimal Action                                   | Max Reward |
|------------|-----------------------------------------------------------------------------|--------------------------------------------------|------------|
| **Easy**   | User watches a YouTube tutorial and *explicitly asks for help*              | `summarize_screen`                               | 1.0        |
| **Medium** | User is coding in VS Code with focused typing (deep work)                  | `stay_silent` × 5 turns                          | 1.0        |
| **Hard**   | Terminal shows `npm ERR!` while user has erratic mouse (silent frustration) | `proactive_help` (mentioning "npm" or "error")   | 1.0        |

### Scoring Details

Scores are **continuous** — they depend on the quality of the agent's response,
not just a binary correct/incorrect check:

| Action                              | Easy           | Medium (per turn) | Hard               |
|-------------------------------------|----------------|--------------------|--------------------|
| `summarize_screen`                  | 0.4–0.99       | 0.0 (fail)         | 0.0                |
| `stay_silent`                       | 0.0            | 0.14–0.24          | 0.0                |
| `proactive_help` + specific error   | 0.0            | 0.0 (fail)         | 0.6–0.99           |
| `proactive_help` + generic payload  | 0.0            | 0.0 (fail)         | 0.4–0.6 (partial)  |

- **Easy** rewards are based on payload similarity to the screen context + response length quality.
- **Medium** rewards increase per turn (earlier turns earn less, later turns earn more).
- **Hard** rewards combine keyword detection (npm, error) with fuzzy text matching.

---

## 📦 Installation

### Prerequisites

- Python 3.10 or higher
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

The inference script starts its own server, runs all three task tiers against
an LLM, and prints structured logs:

```bash
# Set environment variables for the LLM API
export API_BASE_URL="https://router.huggingface.co/v1"
export MODEL_NAME="Qwen/Qwen2.5-72B-Instruct"
export HF_TOKEN="your-huggingface-token"

# Run evaluation
python inference.py
```

#### Inference Features

- **Exponential-backoff retry** (3 attempts) on LLM API calls
- **Defensive JSON parsing** — handles markdown fences, nested objects, keyword fallback
- **Single asyncio event loop** for all tasks (no deprecation warnings)
- **Graceful Ctrl+C handling** with proper server cleanup
- **Summary report** at the end with per-tier pass/fail status

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

  ✓ PASS  easy_correct_action        (reward=1.0)
  ✓ PASS  medium_correct_actions     (total_reward=1.00)
  ✓ PASS  hard_correct_action        (reward=1.0)
  ✓ PASS  easy_wrong_action          (reward=0.0)
  ✓ PASS  medium_interrupted         (interruption correctly penalised)
  ✓ PASS  hard_generic_payload       (reward=0.5)
  ✓ PASS  hard_wrong_action          (reward=0.0)
  ✓ PASS  health_endpoint
  ✓ PASS  root_endpoint

------------------------------------------------------------
  9/9 tests passed — ALL PASSED
------------------------------------------------------------
```

---

## 🐳 Docker

### Build and Run

```bash
docker build -t context-aware-env -f server/Dockerfile .
docker run -p 8000:8000 context-aware-env
```

### Docker Features

- **Multi-stage build** for smaller image size
- **Non-root user** (`appuser`) for security
- **Built-in `HEALTHCHECK`** for auto-restart on failure
- **`.dockerignore`** excludes `.git`, `__pycache__`, `uv.lock`, tests, and docs

---

## 📋 OpenEnv Validation

```bash
openenv validate
```

---

## 📝 Log Format

The `inference.py` script outputs structured lines for the automated evaluation parser:

```
[START] task=easy_1 env=local model=Qwen/Qwen2.5-72B-Instruct
[STEP] step=1 action=summarize_screen(task=easy) reward=0.85 done=true error=null
[END] success=true steps=1 score=0.847 rewards=0.85
```

After all tasks, a summary report is printed:

```
============================================================
  INFERENCE SUMMARY
============================================================
  easy_1      PASS  score=0.847  steps=1  rewards=0.85
  medium_1    PASS  score=0.190  steps=5  rewards=0.14,0.17,0.19,0.21,0.24
  hard_1      PASS  score=0.706  steps=1  rewards=0.71
------------------------------------------------------------
  TOTAL    3/3 passed  aggregate_score=0.581
============================================================
```

---

## 📁 Project Structure

```
context_aware_env/
├── .gitignore             # Git exclusions (pycache, env, lock)
├── .dockerignore          # Docker build exclusions
├── __init__.py            # Package re-exports & metadata
├── models.py              # Pydantic data contracts (Action, Observation, State)
├── client.py              # Typed async WebSocket client
├── inference.py           # LLM evaluation script (with retry & defensive parsing)
├── test_endpoints.py      # Comprehensive test suite (9 tests)
├── openenv.yaml           # OpenEnv manifest
├── pyproject.toml         # Python project configuration
├── requirements.txt       # Pinned dependencies
├── README.md              # This file
├── project_documentation.md  # Detailed technical documentation
└── server/
    ├── __init__.py        # Server sub-package
    ├── environment.py     # Core RL environment logic (task presets, rewards)
    ├── app.py             # FastAPI server (health, CORS, WebSocket)
    └── Dockerfile         # Multi-stage container (non-root, HEALTHCHECK)
```

---

## 🔧 Environment Variables

| Variable              | Default                                  | Description                                  |
|-----------------------|------------------------------------------|----------------------------------------------|
| `API_BASE_URL`        | `https://router.huggingface.co/v1`       | OpenAI-compatible API endpoint               |
| `MODEL_NAME`          | `Qwen/Qwen2.5-72B-Instruct`             | Model identifier for chat completions        |
| `HF_TOKEN`            | *(empty)*                                | HuggingFace API token (**never hardcode!**)  |
| `SERVER_WAIT_TIMEOUT` | `30`                                     | Seconds to wait for server health check      |

> ⚠️ **Security Warning:** NEVER put your `HF_TOKEN` directly in source code.
> Always set it as an environment variable. If you accidentally committed a token,
> **revoke it immediately** at https://huggingface.co/settings/tokens.

---

## 🛡️ Robustness Features

- **`extra="forbid"`** on all Pydantic models — catches typos in field names immediately
- **`model_validator`** on `ContextAction` — auto-clears payload when `action_type="stay_silent"`
- **`MAX_STEPS_PER_EPISODE = 20`** — prevents runaway agents from looping forever
- **Episode-done guard** — returns zero-reward observation if `step()` is called after done
- **Task name validation** — raises `ValueError` on invalid task names instead of silently defaulting
- **Non-retryable error detection** — 401/403 auth errors skip retry (fail fast)
- **6-strategy defensive parsing** — guarantees valid action from any LLM output
- **Startup token warning** — clear instructions when `HF_TOKEN` is missing

---

## 🚢 Deployment

### GitHub

```bash
git add .
git commit -m "feat: ContextAwareEnv v1.0.0"
git remote add origin https://github.com/YOUR_USERNAME/context-aware-env.git
git push -u origin main
```

### Docker Hub

```bash
docker build -t context-aware-env -f server/Dockerfile .
docker tag context-aware-env YOUR_DOCKERHUB/context-aware-env:latest
docker push YOUR_DOCKERHUB/context-aware-env:latest
```

### HuggingFace Spaces

```bash
pip install -U "huggingface_hub[cli]"
hf auth login
hf upload YOUR_HF_USERNAME/context-aware-env . . --repo-type space
```

📖 See [`project_documentation.md`](project_documentation.md) for detailed deployment guides.

---

## License

MIT

