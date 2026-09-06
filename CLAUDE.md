# local-hawk-ai

## Setup and environment

- `core.py` needs `--gsip <host:port>` passed explicitly — defaults to `127.0.0.1:9000`, which silently polls itself instead of the real GS and looks like "no work available" forever.
- `--aip <host:port>` is the autopilot flag (not `--autopilot-ip`, despite older README wording).
- Prefer `cuair-imaging.local` (mDNS) over raw IP for SSH to the Pi — dev machines with 2 active ethernet interfaces get flaky/nondeterministic routing to a bare IP; mDNS pins the correct interface.
- GroundingDINO (vendored under `GroundingDINO/`) needs transformers ~4.33.x API (`BertModel.get_head_mask`); transformers 5.x removed it and also broke `get_extended_attention_mask` internals. Real fix is a Python 3.11 venv with `transformers==4.33.2` pinned (3.13 has no prebuilt `tokenizers` wheel for that era) — not yet done; only `get_head_mask` is patched around in `bertwarper.py`.
- gs-backend (native, not Docker) needs its own JDK 11 (`brew install openjdk@11`, keep keg-only) — Gradle 7.5.1 doesn't run on Java 21+. Export `JAVA_HOME` per-invocation rather than changing system default.
- Fresh Homebrew Postgres has no `postgres` role by default (only your OS-username role) — gs-backend expects `postgres`/`admin` and a `groundserver` DB; create both manually before first run.
- `git merge-tree <base> A B` (legacy 3-arg form) can report a merge as conflict-free when the real `git merge` finds one — verify with an actual `git merge --no-commit --no-ff` (then `git merge --abort`) before trusting a "clean" dry run.

## Ports and startup

- The dashboard is on **9080** (API, SSE, `/export`, command channel) and **9081** (frontend HTML). Not 8080/8081: `main()`'s signature defaults disagree with argparse, and argparse wins. `--local` rewrites the GS/cloud/autopilot **addresses** but does not touch the ports.
- `--local` sets the cloud server to `127.0.0.1:8000`, which collides with MPS's own default port. Run a local fake MPS on 8100 instead.
- `start_server` catches a bind failure, prints it, and **the process keeps running with no API server**. A second instance looks alive but serves nothing, so a stale dashboard is often really a port collision. Check `lsof -nP -iTCP:9080 -sTCP:LISTEN` first.
- `main()` is called **fully positionally**. Append new parameters to the end of both the signature and the call site, or every argument silently shifts by one.
- Capture controls are inert unless `--mps <host:port>` is passed; without it `/api/capture/status` returns `{"configured": false}` without opening a socket, and the dashboard button reads "not configured".
- `--classify-every-n` (default **2**) gates classification only. Every image is still pulled, exported, SSE-notified and saved for mapping in `request_image()`; the gate wraps `self.run_model()` in `run_task()`, which runs after that. It is a **count, not a duration** - 2 means one classification per 4 s only while MPS captures every 2 s. A duration gate is not viable: the assignment's `image.timestamp` is wrong by decades in real data, and wall-clock spacing goes unpredictable while the loop drains a backlog.

## HTTP handler traps

- `MapCommandHandler.do_GET` **never 404s**. Any unmatched path returns HTTP 200 with the mapping-status body. A new GET route must be inserted *before* that fallthrough or it is silently unreachable.
- `do_GET` unquotes the path and then splits off the query, discarding it. Parse query params from the raw `self.path` with `urlsplit` + `parse_qs`; parsing the already-unquoted half misreads a `%3F` inside a value.
- Every SSE connection **permanently leaks a thread**: registration ends in `while True: time.sleep(60)`, and the disconnect cleanup below it is unreachable because `time.sleep` never raises on peer disconnect. Dead clients are only reaped on the next broadcast. Do not stream anything high-frequency (logs, telemetry) over `/api/stream` - poll instead.
- New handler config follows one pattern: a class attribute on `MapCommandHandler`, assigned in `start_server()` before `ThreadingHTTPServer` is constructed.
- `GET /export/<rel>` builds `EXPORT_DIR / rel` from an already-unquoted path with **no containment check**, on a `0.0.0.0`-bound server. `GET /export/%2e%2e/%2e%2e/etc/passwd` escapes the directory. Still unfixed; the one-line fix is an `is_relative_to(EXPORT_DIR.resolve())` guard.
- `request_image` retries `get_image_assignment()` in a `while True` with **no sleep** when it returns `None`, so a downed gs-backend pegs a core. The dashboard keeps working because it is on other threads.

## Frontend (`frontend/index.html`)

- The main script has a **flat global scope** - no IIFE, no modules, no `DOMContentLoaded`. A duplicate top-level `const` throws `SyntaxError` and blanks the **entire page**. Put new JS in its own `<script>` element wrapped in an IIFE: each element is compiled independently, so a typo can only break its own feature. The capture bar and log panel are separate elements for exactly this reason.
- `refresh()` wipes `#log-tent`, `#log-mannequin`, and `#gs-panel` every 10 s and on every SSE event. Never put interactive UI inside them; full-width panels go after `</main>`.
- `__MAP_API_BASE__` on line 6 is a **literal string replace** in `render_frontend_index()`. Do not alter that line or introduce a second occurrence.
- `.board` is a fixed three-column grid (`minmax(0,1fr) minmax(0,1fr) minmax(0,2fr)` - the GS column is deliberately double width). A fourth `<section>` inside `<main>` wraps onto a second row rather than becoming a column. Put full-width panels *after* `</main>` instead, which is where the log panel lives.
- `.log`'s `max-height: calc(100vh - Npx)` is hardcoded against the height of everything stacked above it (topbar, `#admin-status`, `#cap-bar`). Adding a row up there pushes the column scroll area off-screen without any other symptom.
- `runAdminCommand` writes to the shared `#admin-status` and defaults to `refreshOnSuccess: true`. Do not reuse it for high-frequency controls - a capture error would be wiped by the next Check Cloud click, and every click would trigger a full column re-render.

## Talking to MPS

- **Never** use `WorkClient._do_request_with_retries` for a non-idempotent command. It retries, and a retried capture start whose first response was merely lost issues a *second* start - which restarts the pipeline mid-pass, the exact thing the guard exists to prevent. Capture traffic is zero-retry.
- The anti-restart guard is enforced **server-side** (re-read `/pipeline/status` under `CAPTURE_CMD_LOCK`, refuse unless `mode == "idle"`), not just in the UI, so a stale tab, a second browser, or a raw `curl` cannot bypass it.
- Capture status lives on its own endpoint and its own frontend timer, deliberately kept out of `/api/best`, so an unreachable Pi cannot slow the main dashboard poll.
- The MPS log tab is proxied with a circuit breaker: after 3 consecutive failures it stops dialling the aircraft for 30 s. That link also carries image uploads.

## Log panel

- Sources are an **allowlist keyed by id** (`lhai`, `gsbackend`, `mps`); the client sends a key, never a path. The `mps` source only appears when `--mps` is set.
- gs-backend's file log is at `<checkout>/../gs-backend/logs/server.log` via the docker-compose bind mount, and only exists after gs-backend's first request. It is an HTTP access log, so it shows every frame arriving from the aircraft and every ADLC claim.
- Do **not** add a log endpoint to gs-backend: its `LoggingFilter` logs every request, so polling such an endpoint would append a line per poll and feed itself. Read the bind-mounted file instead.
- Render log lines with `textContent`, never `innerHTML` - gs-backend's access log echoes the request URI, so any LAN client can inject markup by requesting a crafted path.
- The `gen` token must hash the file's **first complete line**. A fixed-size head changes as a short file grows (every poll then looks like a restart), and inode alone cannot detect MPS truncating its log in place.
