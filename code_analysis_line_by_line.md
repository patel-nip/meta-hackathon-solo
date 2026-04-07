# ContextAwareEnv Codebase Analysis (Line-by-Line)

This document provides a detailed line-by-line explanation of every file in the ContextAwareEnv project.

---

## 1. `openenv.yaml` (14 lines)

- **L1**: `name: context-aware-env` — Defines the name of the project for OpenEnv registry.
- **L2**: `version: 1.0.0` — Semantic version of the environment.
- **L3–L8**: `description: >` — Multi-line YAML string describing the environment's purpose: social-awareness evaluation across three difficulty tiers (easy, medium, hard).
- **L9**: `author: hackathon-team` — Specifies the author team.
- **L10**: `entrypoint: server/app.py` — Tells OpenEnv where the FastAPI app lives.
- **L11–L13**: `docker:` — Container configuration: Dockerfile path (`server/Dockerfile`) and port mapping (`8000`).

---

## 2. `requirements.txt` (27 lines)

- **L1–L7**: Header comments explaining how to install and keep in sync with `pyproject.toml`.
- **L9–L10**: `openenv-core>=0.2.0` — Core framework providing `Environment`, `EnvClient`, `create_fastapi_app`.
- **L12–L14**: `fastapi>=0.104.0`, `uvicorn>=0.24.0` — Web server stack.
- **L16–L17**: `pydantic>=2.0.0` — Data validation with `extra="forbid"` and `model_validator`.
- **L19–L20**: `openai>=1.0.0` — OpenAI-compatible client for HuggingFace Inference Providers.
- **L22–L23**: `websockets>=12.0` — WebSocket client used by inference.py and tests.
- **L25–L26**: `httpx>=0.25.0` — HTTP client for health-check polling.

---

## 3. `pyproject.toml` (37 lines)

- **L1–L3**: `[build-system]` — Uses setuptools with the legacy backend.
- **L5–L11**: `[project]` — Package metadata: name, version, description, readme, Python ≥3.10, MIT license.
- **L13–L21**: `dependencies` — All 7 runtime dependencies including `httpx`.
- **L23–L24**: `[project.scripts]` — Console entry point: `server` → `server.app:main`.
- **L26–L30**: `[project.optional-dependencies]` — Dev dependencies: `pytest`, `pytest-asyncio`.
- **L32–L33**: `packages.find` — Includes both `context_aware_env*` and `server*` packages (fixed from original which only had `server*`).
- **L35–L36**: `package-data` — Bundles `*.yaml`, `*.txt`, `*.md` files in the distribution.

---

## 4. `__init__.py` (70 lines)

- **L1–L24**: Module docstring with quick-start example showing how to use `ContextEnvClient` and `ContextAction`.
- **L26–L30**: Package metadata: `__version__ = "1.0.0"`, `__author__ = "hackathon-team"`.
- **L32–L57**: Dual-mode import system:
  - **L39–L45**: First tries proper package import: `from context_aware_env.models import ...` (works when pip-installed).
  - **L46–L53**: Falls back to local import: `from models import ...` (works when running from project directory).
  - **L54–L57**: Catches all `ImportError` silently (prevents crash during pip install resolution).
- **L59–L69**: `__all__` export list: `__version__`, `__author__`, `ContextAction`, `ContextObservation`, `ContextState`, `ContextEnvClient`.

---

## 5. `models.py` (156 lines)

- **L1–L18**: Module docstring listing the three Pydantic models and their purposes.
- **L20–L24**: Imports: `Literal` (type constraints), `Field` + `model_validator` (Pydantic), base types from OpenEnv.
- **L27–L30**: `VALID_ACTION_TYPES` — Tuple constant `("stay_silent", "summarize_screen", "proactive_help")`.
- **L32–L37**: `__all__` export list.
- **L43–L80**: **`ContextAction(Action)`**:
  - **L61**: `model_config = {"extra": "forbid"}` — Unknown fields raise `ValidationError` (catches typos).
  - **L63–L70**: `action_type` (Literal with 3 options) and `payload` (str, default `""`) fields.
  - **L74–L80**: `@model_validator(mode="after")` — If `action_type` is `"stay_silent"` and `payload` is non-empty, auto-clears the payload. Uses `object.__setattr__` for Pydantic compatibility.
- **L86–L121**: **`ContextObservation(Observation)`**:
  - **L104**: `extra="forbid"` config.
  - **L106–L121**: Four fields: `active_app`, `visible_text`, `user_telemetry`, `explicit_help_request`.
- **L127–L155**: **`ContextState(State)`**:
  - **L143**: `extra="forbid"` config.
  - **L145–L155**: Two fields: `current_task` (default `"easy"`), `silent_turns_completed` (default `0`).

---

## 6. `client.py` (142 lines)

- **L1–L29**: Module docstring with usage example showing async client workflow.
- **L31–L37**: Imports: `__future__` annotations, `logging`, `typing`, OpenEnv `EnvClient` and `StepResult`.
- **L43–L50**: Dual-mode import for models (package path → local fallback).
- **L52**: `__all__ = ["ContextEnvClient"]`.
- **L54**: Logger instance.
- **L60–L142**: **`ContextEnvClient`** class:
  - **L71–L84**: `_step_payload()` — Serialises `ContextAction` via `model_dump()`.
  - **L88–L125**: `_parse_result()` — Extracts `observation` sub-dict, merges top-level `reward`/`done` if missing, constructs `ContextObservation`, returns `StepResult`.
  - **L129–L142**: `_parse_state()` — Constructs `ContextState` from raw dict.

---

## 7. `server/__init__.py` (18 lines)

- **L1–L10**: Module docstring describing the server sub-package contents.
- **L12–L15**: Safe import of `ContextAwareEnvironment` with try/except (prevents crash during pip install).
- **L17**: `__all__ = ["ContextAwareEnvironment"]`.

---

## 8. `server/environment.py` (346 lines)

- **L1–L23**: Module docstring explaining the three task tiers, design decisions (`extra="forbid"`, `MAX_STEPS_PER_EPISODE`, `ValueError` on bad task names).
- **L25–L32**: Imports: `__future__`, `logging`, `typing`, `uuid4`, OpenEnv `Environment` interface.
- **L38–L47**: Dual-mode import for models (package path → local fallback with `sys.path`).
- **L50**: Logger instance.
- **L55–L62**: Named constants:
  - `MAX_STEPS_PER_EPISODE = 20` — Hard ceiling on steps.
  - `MEDIUM_SILENT_TURNS_REQUIRED = 5` — Turns needed for medium task.
  - `MEDIUM_PER_TURN_REWARD = 0.2` — Reward per silent turn.
- **L67–L86**: `TASK_PRESETS` dict — Three preset observation configs for easy, medium, hard.
- **L88–L89**: `VALID_TASK_NAMES` — `frozenset` of valid task names.
- **L95–L346**: **`ContextAwareEnvironment`** class:
  - **L109–L126**: `__init__()` — Default state (easy task), `_episode_done` flag, logger info.
  - **L130–L189**: `reset(task_name)`:
    - **L159–L164**: Validates `task_name` against `VALID_TASK_NAMES`, raises `ValueError` with helpful message on mismatch.
    - **L170–L177**: Creates fresh `ContextState`, loads preset.
    - **L185–L189**: Returns initial observation with `done=False, reward=0.0`.
  - **L193–L322**: `step(action)`:
    - **L213–L224**: Guard 1 — `_episode_done` flag → returns zero-reward done observation.
    - **L226–L238**: Guard 2 — `step_count >= MAX_STEPS_PER_EPISODE` → forces done.
    - **L240–L244**: Increments step count, sets defaults.
    - **L246–L256**: **Easy branch** — `summarize_screen` → 1.0; else → 0.0; always done.
    - **L258–L275**: **Medium branch** — `stay_silent` → 0.2 + increment; 5 turns → done; interruption → 0.0 + done.
    - **L277–L297**: **Hard branch** — `proactive_help` + "npm"/"error" → 1.0; generic → 0.5; else → 0.0; always done.
    - **L299–L303**: Else branch (unrecognised task — should never happen).
    - **L305–L316**: Sets `_episode_done`, debug log.
    - **L318–L322**: Returns final observation.
  - **L326–L329**: `state` property — read-only access to `_state`.
  - **L333–L345**: `get_debug_info()` — Returns JSON-serialisable dict of all internals.

---

## 9. `server/app.py` (151 lines)

- **L1–L19**: Module docstring listing endpoints (`/health`, `/ws`, `/`) and usage.
- **L22–L26**: Imports: `__future__`, `logging`, `sys`, `os`.
- **L28–L29**: FastAPI and CORS imports.
- **L35–L40**: `sys.path` adjustment — the ONE tolerated bootstrap for `uvicorn server.app:app`.
- **L42**: Imports `create_fastapi_app` from OpenEnv.
- **L47–L52**: Dual-mode model imports (package → local fallback).
- **L57–L62**: Logging config: `timestamp | LEVEL | module | message` format.
- **L67–L72**: `app = create_fastapi_app(...)` — Creates the FastAPI instance.
- **L78–L85**: CORS middleware (all origins, all methods — dev mode).
- **L90–L91**: Module-level log message (replaces deprecated `@app.on_event`).
- **L97–L108**: `GET /health` — Returns `{"status": "healthy", "service": "context-aware-env", "version": "1.0.0"}`.
- **L111–L130**: `GET /` — Returns service metadata: name, version, description, endpoints, available tasks.
- **L137–L148**: `main()` — Direct-run entry point using `uvicorn.run()`.
- **L150–L151**: `if __name__ == "__main__"` guard.

---

## 10. `server/Dockerfile` (40 lines)

- **L1–L9**: **Stage 1 (builder)** — `python:3.11-slim`, copies `requirements.txt`, installs dependencies to `/install` prefix.
- **L12–L13**: **Stage 2 (production)** — Fresh `python:3.11-slim` image.
- **L15–L17**: Creates non-root user `appuser` (UID/GID 1000) for security.
- **L19**: Sets working directory `/app`.
- **L21–L22**: Copies installed packages from builder stage.
- **L24–L25**: Copies project source code.
- **L27–L28**: Switches to non-root user.
- **L30–L31**: Exposes port 8000.
- **L33–L36**: `HEALTHCHECK` — Polls `/health` every 30s, 5s timeout, 10s start period, 3 retries.
- **L38–L39**: `CMD` — Launches uvicorn server.

---

## 11. `.dockerignore` (35 lines)

Excludes from Docker build context:
- **L2–L3**: `.git`, `.gitignore`
- **L6–L11**: Python artifacts (`__pycache__`, `*.pyc`, `*.egg-info`, `dist/`, `build/`)
- **L14–L17**: Virtual environments (`.venv`, `venv`, `env`)
- **L20**: `uv.lock` (540KB lock file)
- **L23–L26**: IDE files (`.vscode`, `.idea`, swap files)
- **L29–L30**: Markdown docs (except `README.md`)
- **L33–L34**: Test files (`test_*.py`, `tests/`)

---

## 12. `test_endpoints.py` (332 lines)

- **L1–L23**: Module docstring describing the 9 tests (positive, negative, endpoint) and usage instructions.
- **L26–L40**: Imports and dual-mode model import.
- **L45–L46**: Config: `WS_URL`, `WS_TIMEOUT = 10`.
- **L51–L61**: `parse_obs()` — Extracts `observation` sub-dict, merges `reward`/`done`.
- **L66–L76**: `_results` list and `_record()` — Tracks test results for summary.
- **L84–L105**: **`test_easy_correct_action()`** — Reset easy, send `summarize_screen`, assert reward=1.0, done=True.
- **L108–L131**: **`test_medium_correct_actions()`** — Reset medium, 5× `stay_silent`, assert 0.2 each, total 1.0, done=True.
- **L134–L156**: **`test_hard_correct_action()`** — Reset hard, send `proactive_help` with npm error, assert reward=1.0.
- **L164–L181**: **`test_easy_wrong_action()`** — Send `stay_silent` on easy → reward=0.0, done=True.
- **L184–L209**: **`test_medium_interrupted()`** — 2× silent then `proactive_help` → reward=0.0 on interruption.
- **L212–L231**: **`test_hard_generic_payload()`** — `proactive_help` without keywords → reward=0.5.
- **L234–L250**: **`test_hard_wrong_action()`** — `stay_silent` on hard → reward=0.0.
- **L253–L263**: **`test_health_endpoint()`** — `GET /health` → status in ("ok", "healthy").
- **L266–L277**: **`test_root_endpoint()`** — `GET /` → service="ContextAwareEnv", has "tasks".
- **L285–L297**: `ALL_TESTS` list — All 9 test functions.
- **L301–L328**: `main()` — Runs all tests with try/except isolation, prints summary.
- **L331–L332**: `asyncio.run(main())`.

---

## 13. `inference.py` (682 lines)

- **L1–L36**: Shebang + module docstring: Architecture overview (4 subsystems), env vars, crash-safety guarantee.
- **L38–L49**: Standard library imports: `asyncio`, `atexit`, `json`, `os`, `re`, `signal`, `subprocess`, `sys`, `time`, `Optional`.
- **L54–L57**: Dual-mode import for models.
- **L60–L104**: **Configuration block:**
  - **L74–L80**: API settings: `API_BASE_URL`, `MODEL_NAME`, `HF_TOKEN` from env vars.
  - **L84–L89**: Runtime constants: `TASKS`, `MAX_STEPS=8`, `SERVER_PORT=8000`, URLs, `ENV_NAME`.
  - **L93–L97**: Retry config: `LLM_MAX_RETRIES=3`, `LLM_RETRY_BASE_DELAY=1.0`.
  - **L101–L104**: `SERVER_WAIT_TIMEOUT` from env var (default 30s).
- **L111**: `DEFAULT_ACTION` — Safe fallback `ContextAction(action_type="stay_silent")`.
- **L118–L136**: `SYSTEM_PROMPT` — LLM instructions specifying JSON response format and action guidelines.
- **L143–L237**: **Server lifecycle:**
  - **L146–L177**: `start_server()` — Launches uvicorn subprocess, registers atexit cleanup, waits for health.
  - **L180–L212**: `_wait_for_server()` — Polls `/health` with countdown, logs attempt count.
  - **L215–L237**: `stop_server()` — SIGTERM → wait 5s → SIGKILL; flushes stdout/stderr.
- **L245–L353**: **WebSocket episode runner:**
  - **L245–L267**: `_parse_ws_observation()` — Extracts obs fields, merges reward/done.
  - **L270–L353**: `ws_run_episode(task_name)` — Opens WebSocket, sends reset, enters step loop (up to MAX_STEPS), queries LLM, parses action, sends step, logs each step. Returns `(rewards, step_count, episode_id)`.
- **L361–L522**: **LLM interaction:**
  - **L361–L383**: `build_user_message()` — Formats observation as structured text prompt.
  - **L386–L465**: `parse_action()` — 6-strategy defensive parsing:
    - L414–L417: Strip markdown fences
    - L419–L431: Direct JSON with nested-object unwrapping
    - L433–L450: Regex extraction with one-level nesting support
    - L452–L463: Keyword detection fallback
    - L465: Default `stay_silent` fallback
  - **L468–L522**: `query_llm()` — Exponential-backoff retry (3 attempts, 1s/2s/4s delays).
- **L530–L565**: **Logging helpers:**
  - `log_start()`, `log_step()`, `log_end()` — Structured output with optional `episode_id`.
- **L573–L677**: **Main loop:**
  - **L573–L619**: `_run_all_tasks()` — Async function running all 3 tasks in a single event loop.
  - **L622–L677**: `run()` — Entry point: Ctrl+C handler, start server, `asyncio.run()`, stop server, print summary report.
- **L680–L681**: `if __name__ == "__main__": run()`.

---

**Summary**: This document provides exhaustive line-by-line mapping of every function, class, import, constant, validator, guard, and configuration across all 13 source files in the ContextAwareEnv project. Every architectural layer — data models, server environment, FastAPI app, client, inference engine, tests, and deployment artifacts — is thoroughly documented.
