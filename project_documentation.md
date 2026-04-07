# ContextAwareEnv: Comprehensive Project Documentation

---

## 1. Project Overview

**ContextAwareEnv** is a custom, OpenEnv-compliant reinforcement learning (RL) environment designed to evaluate an LLM-based intelligent agent's capacity for "social awareness." In human-computer interaction, a truly smart assistant must distinguish between scenarios where it should proactively offer help and scenarios where it is better to stay completely silent and avoid interrupting the user's workflow.

**The Problem It Solves:** Current autonomous agents frequently struggle with context awareness — they tend to be overly intrusive, triggering unprompted responses regardless of what the user is currently doing. ContextAwareEnv provides a standardized, isolated testing bed to evaluate an agent's ability to remain unobtrusive.

**Target Users and Use-Cases:** Primarily designed for AI researchers, agent developers, and participants in hackathons (like the Meta × Scaler OpenEnv Hackathon). The use-case is evaluating Large Language Models (LLMs) sequentially to determine if their decision-making logic aligns with human expectations of politeness and helpfulness.

**Key Objectives and Goals:**
- To securely evaluate an agent across distinct stateful scenarios (easy, medium, hard).
- To provide a robust set of evaluation APIs to read system states and dispatch actions.
- To validate that AI assistants can successfully parse observations and output structured, safe actions without unhandled exceptions.

---

## 2. Conceptual Foundation

**Core Ideas and Principles:** At its core, the project builds upon the Reinforcement Learning paradigm: an agent exists in an "Environment", receives "Observations", takes "Actions", and receives a "Reward" based on the optimality of the choice.

**Relevant Technologies:**
- **OpenEnv Standards:** An abstraction layer conceptually similar to OpenAI Gym/Gymnasium, but tailored natively for asynchronous, language-based capabilities.
- **Stateful Mocking:** The environment simulates a user's desktop operating system, providing the agent with textual telemetry (such as the active application, visible text, and the speed of mouse movements).

**Why this approach is used:** The MDP (Markov Decision Process) approach of RL naturally maps onto interaction models for LLMs. Assigning scalar rewards (e.g., 1.0 for success) makes programmatic bulk evaluation straightforward, reducing reliance on expensive and variable human-in-the-loop testing.

---

## 3. System Architecture

**High-Level Architecture Components:**
1. **The Server (Environment Backend):** A FastAPI web server hosting the desktop simulation logic. Maintains internal state representations and executes the logic required to calculate rewards. Exposes HTTP endpoints (`/health`, `/`) and a WebSocket endpoint (`/ws`) with CORS support.
2. **The Client Interface:** A typed async WebSocket client (`ContextEnvClient`) that connects to the environment server, serialising actions and deserialising observations using Pydantic models.
3. **The Inference Engine (Agent Wrapper):** The component driving the evaluation. It manages the server lifecycle, bridges the environment with the hosted Large Language Model (via OpenAI-compatible API), uses exponential-backoff retry logic (with smart non-retryable error detection), and employs a 6-strategy defensive parsing pipeline to handle arbitrary LLM outputs.

**Data Flow Between Components:**

```
┌──────────────┐     WebSocket      ┌──────────────────┐
│  Inference   │ ←──────────────────→│   FastAPI Server  │
│   Engine     │  reset / step      │  (environment.py) │
│              │  + observations    │                    │
│  ┌────────┐  │                    │  ┌──────────────┐  │
│  │  LLM   │  │                    │  │ Task Presets  │  │
│  │ (API)  │  │                    │  │ easy/med/hard │  │
│  └────────┘  │                    │  └──────────────┘  │
└──────────────┘                    └──────────────────┘
```

**Step-by-Step Data Flow:**
1. **Server Startup:** The Inference Engine starts the FastAPI server as a subprocess and polls `/health` until it responds `{"status": "healthy"}` (up to 30 seconds, configurable).
2. **Reset:** The Inference Engine sends a WebSocket `reset` message with the task name (`"easy"`, `"medium"`, or `"hard"`).
3. **Observation:** The Server validates the task name (raising `ValueError` on invalid input), initialises `ContextState`, and sends back a `ContextObservation` containing the desktop screen state.
4. **Prompting:** The Inference Engine wraps the observation into a structured prompt and queries the remote LLM (with up to 3 retry attempts for transient errors; auth errors fail immediately).
5. **Parsing:** The LLM response is defensively parsed through 6 strategies (see Section 4).
6. **Action Dispatch:** The parsed `ContextAction` is sent as a WebSocket `step` message.
7. **Evaluation:** The Server executes the action, updates its internal rules engine, and returns the new observation. This repeats until `done=True` or `MAX_STEPS_PER_EPISODE` (20) is reached.

---

## 4. Detailed Component Breakdown

---

### 4.1 `openenv.yaml` — OpenEnv Manifest (14 lines)

The OpenEnv discovery file. Tells the framework where to find the server, how to containerise it, and basic metadata.

```yaml
name: context-aware-env        # Project identifier
version: 1.0.0                 # Semantic version
description: >                 # Multi-line description
  A social-awareness RL environment that evaluates whether an LLM agent
  knows when to stay silent and when to proactively interrupt to help.
  Simulates a user's desktop OS with three difficulty tiers:
  easy (explicit help request), medium (deep work / stay silent),
  and hard (unspoken frustration / proactive help).
author: hackathon-team         # Team name
entrypoint: server/app.py      # Where create_fastapi_app is called
docker:
  dockerfile: server/Dockerfile  # Container build recipe
  port: 8000                     # Exposed port
```

---

### 4.2 `models.py` — Pydantic Data Contracts (156 lines)

Defines three strictly-typed Pydantic models that form the data contract between agent, client, and server. Every model uses `model_config = {"extra": "forbid"}` to reject unknown fields (catches typos).

**`ContextAction(Action)`** — The agent's action:
- `action_type`: Literal `"stay_silent"` | `"summarize_screen"` | `"proactive_help"`
- `payload`: Optional text when speaking (auto-cleared for `stay_silent` by `@model_validator`)
- `model_config = {"extra": "forbid"}` — rejects unknown fields like `paylaod`

**`ContextObservation(Observation)`** — The environment's output:
- `active_app`, `visible_text`, `user_telemetry`, `explicit_help_request`
- Inherits `done` and `reward` from base `Observation`

**`ContextState(State)`** — Internal server state:
- `current_task`, `silent_turns_completed`
- Inherits `episode_id` and `step_count`

**`VALID_ACTION_TYPES`** — Tuple constant for reuse: `("stay_silent", "summarize_screen", "proactive_help")`

---

### 4.3 `client.py` — Typed WebSocket Client (142 lines)

Implements `EnvClient[ContextAction, ContextObservation, ContextState]` with three abstract methods:
- `_step_payload(action)` — Serialises via `model_dump()`
- `_parse_result(payload)` — Merges top-level `reward`/`done` into observation dict
- `_parse_state(payload)` — Constructs `ContextState` from raw dict

Uses dual-mode imports (package path → local fallback) and `logging` module.

---

### 4.4 `server/environment.py` — Core RL Logic (346 lines)

The `ContextAwareEnvironment` class implements the grading rubric:

**Constants:**
- `MAX_STEPS_PER_EPISODE = 20` — prevents runaway agents
- `MEDIUM_SILENT_TURNS_REQUIRED = 5` — turns for medium task
- `MEDIUM_PER_TURN_REWARD = 0.2` — reward per silent turn
- `VALID_TASK_NAMES = frozenset({"easy", "medium", "hard"})` — validates input

**`reset(task_name)`:**
- Validates `task_name` against `VALID_TASK_NAMES` (raises `ValueError` on invalid)
- Creates fresh `ContextState` with UUID `episode_id`
- Loads `TASK_PRESETS` for the chosen tier
- Returns initial `ContextObservation(done=False, reward=0.0)`

**`step(action)`:**
- Guard 1: `_episode_done` → returns zero-reward done observation
- Guard 2: `step_count >= MAX_STEPS_PER_EPISODE` → forces done
- **Easy**: `summarize_screen` → 1.0; else → 0.0; always done
- **Medium**: `stay_silent` → 0.2 + increment counter; 5 turns → done; interruption → 0.0 + done
- **Hard**: `proactive_help` + "npm"/"error" → 1.0; generic → 0.5; else → 0.0; always done

**`get_debug_info()`** — Returns full internal state as JSON-serialisable dict.

**Task Presets:**

| Task   | active_app | visible_text                    | user_telemetry | explicit_help_request |
|--------|------------|---------------------------------|----------------|-----------------------|
| Easy   | YouTube    | Video: React Tutorial           | idle           | True                  |
| Medium | VS Code    | def main():...                  | typing_fast    | False                 |
| Hard   | Terminal   | npm ERR! code ELIFECYCLE        | erratic_mouse  | False                 |

---

### 4.5 `server/app.py` — FastAPI Server (152 lines)

- Creates the app via `create_fastapi_app()` from OpenEnv
- Adds CORS middleware (all origins in dev)
- Structured logging: `timestamp | LEVEL | module | message`
- `GET /health` → `{"status": "healthy", "service": "context-aware-env", "version": "1.0.0"}`
- `GET /` → Service metadata with version, description, endpoints, tasks
- `WS /ws` → WebSocket endpoint (managed by OpenEnv framework)
- `main()` → Direct-run entry point via `uvicorn.run()`

---

### 4.6 `inference.py` — Evaluation Orchestrator (738 lines)

The main evaluation script with 5 subsystems:

**1. Server Lifecycle Management:**
- `start_server()` — Launches uvicorn as subprocess, registers `atexit` cleanup
- `_wait_for_server()` — Polls `/health` up to 30s (configurable via `SERVER_WAIT_TIMEOUT`)
- `stop_server()` — SIGTERM → wait 5s → SIGKILL; flushes stdout/stderr
- Ctrl+C handler via `signal.SIGINT`

**2. WebSocket Episode Runner:**
- `ws_run_episode(task_name)` — Opens WebSocket, sends `reset`, runs step loop (up to `MAX_STEPS=8`)
- Returns `(rewards_list, step_count, episode_id)`

**3. LLM Interaction (with Smart Retry):**
- `_get_llm_client()` — Lazy singleton OpenAI client (created once, reused)
- `_is_non_retryable(exc)` — Detects 401/403/400/404/422 errors that should NOT be retried
- `query_llm(messages)` — Exponential-backoff retry (3 attempts: 1s → 2s → 4s) for transient errors (429, 503); auth errors fail immediately

**4. Defensive Parsing (6-Strategy Cascade — `parse_action()`):**

| # | Strategy | Example Input | Handles |
|---|----------|---------------|---------|
| 1 | Strip markdown fences | `` ```json {...} ``` `` | LLMs wrapping JSON in code blocks |
| 2 | Direct JSON parse | `{"action_type": "..."}` | Well-formed output |
| 3 | Nested object unwrap | `{"response": {"action_type": "..."}}` | LLMs nesting responses |
| 4 | Regex extraction | `Here's my answer: {"action_type": "..."}` | Preamble text before JSON |
| 5 | Keyword detection | `"I'll use proactive_help"` | Completely broken JSON |
| 6 | Default fallback | Anything else | `stay_silent` (always safe) |

**5. Startup Token Validation:**
- Checks if `HF_TOKEN` is empty at startup
- Prints clear warning with instructions for PowerShell and Bash
- Explains that only medium can pass without LLM

---

### 4.7 `test_endpoints.py` — Test Suite (333 lines)

9 comprehensive tests with isolated try/except per test:

| # | Test | Type | Verifies |
|---|------|------|----------|
| 1 | `easy_correct_action` | Positive | `summarize_screen` → reward 1.0 |
| 2 | `medium_correct_actions` | Positive | 5× `stay_silent` → total 1.0 |
| 3 | `hard_correct_action` | Positive | `proactive_help` + "npm error" → 1.0 |
| 4 | `easy_wrong_action` | Negative | `stay_silent` on easy → 0.0 |
| 5 | `medium_interrupted` | Negative | 2× silent then interrupt → 0.0 |
| 6 | `hard_generic_payload` | Negative | Generic proactive_help → 0.5 |
| 7 | `hard_wrong_action` | Negative | `stay_silent` on hard → 0.0 |
| 8 | `health_endpoint` | Endpoint | `GET /health` → "healthy" |
| 9 | `root_endpoint` | Endpoint | `GET /` → service metadata |

---

### 4.8 `__init__.py` — Package Entry Point (70 lines)

- Exports `__version__`, `__author__`, `__all__`
- Re-exports all models and client via dual try/except (installed-package → local fallback)
- Silent `ImportError` catch for pip install resolution

---

### 4.9 `server/__init__.py` — Server Sub-package (18 lines)

- Safe re-export of `ContextAwareEnvironment` with try/except
- `__all__ = ["ContextAwareEnvironment"]`

---

### 4.10 `pyproject.toml` — Project Configuration (37 lines)

- Build system: `setuptools` with legacy backend
- Package: `context-aware-env` v1.0.0, Python ≥3.10, MIT license
- 7 runtime dependencies: `openenv-core`, `fastapi`, `uvicorn`, `pydantic`, `openai`, `websockets`, `httpx`
- 2 dev dependencies: `pytest`, `pytest-asyncio`
- Console script: `server` → `server.app:main`
- Package discovery includes both `context_aware_env*` and `server*`

---

### 4.11 `requirements.txt` — Dependencies (27 lines)

```
openenv-core>=0.2.0    # OpenEnv framework
fastapi>=0.104.0       # Web server
uvicorn>=0.24.0        # ASGI server
pydantic>=2.0.0        # Data validation
openai>=1.0.0          # LLM API client
websockets>=12.0       # WebSocket client
httpx>=0.25.0          # HTTP client for health checks
```

---

### 4.12 `server/Dockerfile` — Container (40 lines)

**Stage 1 (builder):** Install dependencies to `/install` prefix (layer caching).
**Stage 2 (production):** Copy deps from builder, copy source, switch to non-root `appuser`, HEALTHCHECK every 30s.

---

### 4.13 `.dockerignore` — Build Exclusions (35 lines)

Excludes `.git`, `__pycache__`, virtual environments, `uv.lock`, IDE files, markdown docs (except README), test files.

---

### 4.14 `.gitignore` — Version Control Exclusions

Excludes `__pycache__`, `*.pyc`, `uv.lock`, `.env`, virtual environments, IDE files, OS files, coverage reports.

---

## 5. Scoring Rubric

| Task | Optimal Action | Full Reward | Partial Reward | Zero Reward |
|------|----------------|-------------|----------------|-------------|
| **Easy** | `summarize_screen` | 1.0 | — | Any other action |
| **Medium** | `stay_silent` × 5 turns | 1.0 (5×0.2) | ≤0.8 (interrupted early) | Non-silent action |
| **Hard** | `proactive_help` + "npm"/"error" | 1.0 | 0.5 (generic payload) | Any other action |

**Maximum aggregate reward: 3.0** (1.0 per task × 3 tasks)

---

## 6. Environment Variables Reference

| Variable | Default | Description |
|----------|---------|-------------|
| `API_BASE_URL` | `https://router.huggingface.co/v1` | OpenAI-compatible API endpoint |
| `MODEL_NAME` | `Qwen/Qwen2.5-72B-Instruct` | Model for chat completions |
| `HF_TOKEN` | *(empty)* | HuggingFace API token (**NEVER hardcode!**) |
| `SERVER_WAIT_TIMEOUT` | `30` | Seconds to wait for server health |

---

## 7. Working Mechanism (Real-World Walkthrough)

### Easy Task — Explicit Help Request
1. **Obs:** YouTube, "Video: React Tutorial", user idle, `explicit_help_request=True`
2. **LLM Reasoning:** User asked for help → summarize the screen
3. **Action:** `{"action_type": "summarize_screen", "payload": "The video is about React..."}`
4. **Reward:** 1.0 ✓

### Medium Task — Deep Work Mode
1. **Obs:** VS Code, "def main():...", typing_fast, `explicit_help_request=False`
2. **LLM Reasoning:** User is focused → don't interrupt
3. **Action:** `{"action_type": "stay_silent", "payload": ""}` × 5 turns
4. **Reward:** 0.2 per turn × 5 = 1.0 ✓

### Hard Task — Unspoken Frustration
1. **Obs:** Terminal, "npm ERR! code ELIFECYCLE", erratic_mouse, `explicit_help_request=False`
2. **LLM Reasoning:** User struggling but didn't ask → proactively help
3. **Action:** `{"action_type": "proactive_help", "payload": "I see an npm error..."}`
4. **Reward:** 1.0 (mentions "npm" or "error") ✓

---

## 8. Technology Stack

| Technology | Version | Purpose |
|------------|---------|---------|
| Python | ≥3.10 | Core language |
| FastAPI | ≥0.104.0 | Async web server for HTTP + WebSocket |
| Uvicorn | ≥0.24.0 | ASGI server |
| Pydantic | ≥2.0.0 | Data validation (`extra="forbid"`, `model_validator`) |
| WebSockets | ≥12.0 | Stateful bi-directional communication |
| OpenAI SDK | ≥1.0.0 | Client for OpenAI-compatible APIs |
| OpenEnv Core | ≥0.2.0 | Framework: `Environment`, `EnvClient`, `create_fastapi_app` |
| Docker | — | Multi-stage container with HEALTHCHECK |
| httpx | ≥0.25.0 | HTTP client library |

---

## 9. Robustness Features

- **`extra="forbid"`** on all Pydantic models — catches field typos immediately
- **`@model_validator`** on `ContextAction` — auto-clears payload for `stay_silent`
- **`MAX_STEPS_PER_EPISODE = 20`** — prevents runaway agent loops
- **Episode-done guard** — returns zero-reward if `step()` called after done
- **Task name validation** — raises `ValueError` on invalid names
- **Non-retryable error detection** — 401/403 auth errors skip retry backoff
- **6-strategy defensive parsing** — guarantees valid action from any LLM output
- **Startup token warning** — clear instructions when `HF_TOKEN` is missing
- **Graceful shutdown** — Ctrl+C handler, atexit cleanup, SIGTERM → SIGKILL

---

## 10. Test Coverage

### Full Test Results (3/3 passed with LLM)

```
============================================================
  INFERENCE SUMMARY
============================================================
  easy      ✓ PASS  reward=1.00  steps=1
  medium    ✓ PASS  reward=1.00  steps=5
  hard      ✓ PASS  reward=1.00  steps=1
------------------------------------------------------------
  TOTAL    3/3 passed  aggregate_reward=3.00
============================================================
```

### Unit Test Results (9/9 passed)

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

## 11. Challenges and Limitations

**Technical Challenges Solved:**
- LLM output parsing — solved with 6-strategy cascade
- Free-tier API rate limits — solved with exponential-backoff retry + non-retryable detection
- Cross-platform signal handling — careful SIGTERM/SIGKILL implementation
- Mixed stdout/stderr output — moved all logging to stdout with flush

**Known Limitations:**
- **Static Observations:** Relies on hardcoded presets, not real desktop screenshots
- **Limited Granularity:** Only 3 task tiers; no long-term memory scenarios
- **Single-Tenant:** Each WebSocket connection creates its own environment

---

## 12. Future Enhancements

- **Visual State Processing:** Base64 screenshots for VLMs
- **Dynamic Task Generation:** Random OS events for adaptability testing
- **Additional Task Tiers:** Distracted browsing, multi-app switching, collaborative work
- **Persistent Metrics Dashboard:** Web UI for aggregating results across models
- **CI/CD Pipeline:** Automated testing on every commit

---

## 13. Deployment Guide: GitHub

### 13.1 Initial Setup

```bash
# Navigate to project
cd context_aware_env

# Initialize git (if not already done)
git init

# Add all files
git add .

# Verify nothing sensitive is tracked
git status
# Make sure .env, uv.lock, __pycache__ are NOT listed

# Create initial commit
git commit -m "feat: ContextAwareEnv v1.0.0 — social awareness RL environment"
```

### 13.2 Create GitHub Repository

1. Go to https://github.com/new
2. Repository name: `context-aware-env` (or `ContextAwareEnv`)
3. Description: "Social-awareness RL environment for the Meta × Scaler OpenEnv Hackathon"
4. Visibility: **Public** (required for hackathon)
5. Do NOT initialise with README (you already have one)
6. Click **Create repository**

### 13.3 Push to GitHub

```bash
# Add your GitHub remote (replace with YOUR username)
git remote add origin https://github.com/YOUR_USERNAME/context-aware-env.git

# Push
git branch -M main
git push -u origin main
```

### 13.4 Protect Your Token

> ⚠️ **NEVER** commit your `HF_TOKEN` in source code! Always use:
>
> PowerShell: `$env:HF_TOKEN = "hf_your_token_here"`
>
> Bash: `export HF_TOKEN="hf_your_token_here"`
>
> If you accidentally committed a token, revoke it immediately at
> https://huggingface.co/settings/tokens and create a new one.

---

## 14. Deployment Guide: Docker Hub

### 14.1 Build the Image

```bash
# From the context_aware_env/ directory
docker build -t context-aware-env:1.0.0 -f server/Dockerfile .

# Also tag as latest
docker tag context-aware-env:1.0.0 context-aware-env:latest
```

### 14.2 Test Locally

```bash
docker run -p 8000:8000 context-aware-env:latest

# In another terminal, verify:
curl http://localhost:8000/health
# → {"status":"healthy","service":"context-aware-env","version":"1.0.0"}
```

### 14.3 Push to Docker Hub

```bash
# Login to Docker Hub
docker login

# Tag with your Docker Hub username
docker tag context-aware-env:1.0.0 YOUR_DOCKERHUB_USERNAME/context-aware-env:1.0.0
docker tag context-aware-env:latest YOUR_DOCKERHUB_USERNAME/context-aware-env:latest

# Push both tags
docker push YOUR_DOCKERHUB_USERNAME/context-aware-env:1.0.0
docker push YOUR_DOCKERHUB_USERNAME/context-aware-env:latest
```

### 14.4 Verify

```bash
# Pull and run from Docker Hub
docker pull YOUR_DOCKERHUB_USERNAME/context-aware-env:latest
docker run -p 8000:8000 YOUR_DOCKERHUB_USERNAME/context-aware-env:latest
```

---

## 15. Deployment Guide: HuggingFace

### 15.1 Create a HuggingFace Space (Web UI Method)

1. Go to https://huggingface.co/spaces
2. Click **"Create new Space"**
3. Fill in:
   - **Owner:** Your HF username
   - **Space name:** `context-aware-env`
   - **License:** MIT
   - **SDK:** Docker
   - **Visibility:** Public
4. Click **Create Space**

### 15.2 Push via Git

```bash
# Clone the HuggingFace Space
git clone https://huggingface.co/spaces/YOUR_HF_USERNAME/context-aware-env hf-space
cd hf-space

# Copy your project files (from the context_aware_env directory)
cp -r /path/to/context_aware_env/* .

# HuggingFace Spaces expect Dockerfile at the root, so copy it:
cp server/Dockerfile ./Dockerfile

# If using Docker SDK, you may need to adjust the Dockerfile's COPY paths
# since the Dockerfile is now at the root instead of server/

# Add, commit, and push
git add .
git commit -m "feat: deploy ContextAwareEnv v1.0.0"
git push
```

### 15.3 Alternative: Push via HuggingFace CLI

```bash
# Install the HuggingFace CLI
pip install -U "huggingface_hub[cli]"

# Login
hf auth login

# Upload (from the context_aware_env directory)
hf upload YOUR_HF_USERNAME/context-aware-env . . --repo-type space
```

### 15.4 Configure HF Token as a Space Secret

1. Go to your Space settings: `https://huggingface.co/spaces/YOUR_HF_USERNAME/context-aware-env/settings`
2. Scroll to **"Repository secrets"**
3. Add: Name = `HF_TOKEN`, Value = your token
4. This makes the token available as an env var inside the Space

---

## 16. Deployment Guide: OpenEnv Validation

### 16.1 Validate Locally

```bash
# Make sure openenv CLI is installed
pip install openenv-core

# Run validation from the project directory
cd context_aware_env
openenv validate

# This checks:
# ✓ openenv.yaml exists and is valid
# ✓ Entrypoint file (server/app.py) exists
# ✓ Dockerfile builds successfully
# ✓ Server starts and responds to /health
```

### 16.2 Run Full Evaluation

```bash
# Set your token
$env:HF_TOKEN = "hf_your_token_here"   # PowerShell
# export HF_TOKEN="hf_your_token_here"  # Bash

# Run inference (starts server automatically)
python inference.py

# Expected output:
# TOTAL    3/3 passed  aggregate_reward=3.00
```

---

## 17. Quick Reference: Commands

| Action | Command |
|--------|---------|
| Install deps | `pip install -r requirements.txt` |
| Start server | `python -m uvicorn server.app:app --host 0.0.0.0 --port 8000` |
| Run tests | `python test_endpoints.py` |
| Run inference | `python inference.py` |
| OpenEnv validate | `openenv validate` |
| Docker build | `docker build -t context-aware-env -f server/Dockerfile .` |
| Docker run | `docker run -p 8000:8000 context-aware-env` |
| Set token (PS) | `$env:HF_TOKEN = "hf_..."` |
| Set token (Bash) | `export HF_TOKEN="hf_..."` |

---

## 18. Project Structure

```
context_aware_env/
├── .gitignore                 # Git exclusions (pycache, env, lock files)
├── .dockerignore              # Docker build exclusions
├── openenv.yaml               # OpenEnv manifest (name, entry, docker)
├── pyproject.toml             # Python project config (deps, scripts)
├── requirements.txt           # Pinned dependencies
├── README.md                  # Quick-start guide
├── project_documentation.md   # This file — full documentation
├── code_analysis_line_by_line.md  # Line-by-line code analysis
│
├── __init__.py                # Package re-exports & metadata
├── models.py                  # Pydantic models (Action, Observation, State)
├── client.py                  # Typed async WebSocket client
├── inference.py               # LLM evaluation script (retry, parsing, lifecycle)
├── test_endpoints.py          # 9-test comprehensive suite
│
└── server/
    ├── __init__.py            # Server sub-package
    ├── environment.py         # Core RL environment (task presets, rewards)
    ├── app.py                 # FastAPI server (health, CORS, WebSocket)
    └── Dockerfile             # Multi-stage container (non-root, HEALTHCHECK)
```

---

## 19. Summary

ContextAwareEnv is a production-ready, OpenEnv-compliant reinforcement learning environment designed to evaluate LLM contextual intelligence. With its 6-strategy defensive parsing, smart retry logic, strict Pydantic validation, comprehensive 9-test suite, multi-stage Docker deployment, and detailed deployment guides for GitHub/Docker Hub/HuggingFace, the system provides a robust pipeline for the Meta × Scaler OpenEnv Hackathon.

**Final Inference Results: 3/3 PASSED — Aggregate Reward: 3.00**
