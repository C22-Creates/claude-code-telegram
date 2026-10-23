# Telegram Harness v2 — Design

*Status: approved in brainstorming 2026-10-17 · Owner: Carl · Repos: `c22os-telegram` (harness), `c22os` (fast paths)*

## 1. Problem

The Telegram bot is Carl's single front door to c22os (voice notes, topics, scheduled
pushes), and it has become unreliable for interactive work:

- **Silent long runs.** A 23s voice note on 2026-10-14 resumed a 1.1 MB session and ran
  ~42 min wall-clock with only ~7 min of model time and ~30s of tool time; Carl saw nothing
  until the 1800s cap killed it. The 9/22 stream-idle watchdog did not fire.
- **Tool thrashing.** A simple lookup on 2026-10-17 ("Do I have Steve McGovern at
  Crystal Ventures' email?") took 35+ tool calls over 3+ minutes: invalid Missive
  arguments, "paginate at least one mailbox" errors, four `gog --help` discovery calls,
  four 404s from bad calendar syntax, and a 403 (Google People API not enabled). It never
  invoked `/c22:contact-lookup`.
- **Root causes of the thrashing:** the intent map (`.claude/commands/c22/README.md`) is
  load-on-demand, so the agent never sees it; the map lacks Carl's real phrasings; and
  `contact-lookup.md` still points at retired Gmail tooling.
- **Codebase fit.** The bot is a ~22k-line fork of a generic upstream project; c22os uses a
  small fraction. Upstream syncs have broken behavior (SDK 0.1.x → 0.2.157 on 10/06).

Scheduled jobs (11 in `seed_jobs.py`) are healthy. Failures cluster on interactive,
resumed, multi-step conversations.

## 2. Goals and success criteria

| Criterion | Target |
|---|---|
| Person lookup ("do I have X's email", "who is X") | answer < 60s, ≤ 8 tool calls |
| Silent period | never > 5 min without a visible status change |
| Hard timeouts (1800s-style) during 48h soak | zero |
| Run outcome | every run ends with explicit ✅ done / ⚠️ stopped (reason) / ❌ error |
| Carl's workflow | no need to pre-classify "deep work" — one front door for everything |

**Non-goals:** replacing Telegram as the interface (Claude Code Remote Control is held in
reserve, not built); multi-user support; sending files back to Telegram (v1).

## 3. Delivery order

Two tracks. **Track 1 ships first** and works on the current bot — it addresses the
most frequent cause of slow runs. Track 2 starts in parallel on a branch.

---

## 4. Track 1 — Fast paths (repo: `c22os`)

### 4.1 Routing the agent actually sees
- Create `context/intent-map.md`: a compact one-line-per-command table (phrasing →
  command), generated from `.claude/commands/c22/README.md`, under ~60 lines.
- Add Carl's real phrasings, at minimum: "do I have X's email", "what's X's email",
  "find the person from <event/company>", "who did I talk to about …".
- **Injection:** the harness appends `context/intent-map.md` to the system prompt on every
  run (Track 2, §5.3). Until Track 2 ships, the current bot's `sdk_integration.py` system
  prompt builder gets the same one-file append (a ~10-line change on the deployed branch).

### 4.2 Retired-tool audit of every `/c22:*` command
- Grep `.claude/commands/c22/` and `.claude/skills/` for `Gmail`, `mcp__claude_ai_Gmail`,
  `Google Calendar MCP`, `gog gmail`. Each hit is rewired per CLAUDE.md rule 1 (email →
  Missive MCP, calendar → `gog`).
- `contact-lookup.md` first: new order = contacts index (§4.4) → Notion People DB →
  Missive search (correct params per §4.3) → stop. Update its `description` frontmatter
  to drop "Gmail".
- Log the drift class in `context/examples/failure-modes.md` and the decision in
  `context/decisions.md`.

### 4.3 Tool recipes cheat sheet
- Create `context/tool-recipes.md`, preloaded via the same system-prompt injection.
- 10–15 recipes, each a copy-pasteable, **verified-by-running** invocation with one line
  of expected output shape. Required set:
  - Missive: search conversations by text and by domain (with the mailbox/`all` param that
    avoids "paginate at least one mailbox"); search contacts across both contact books;
    get a conversation's messages.
  - `gog`: calendar event search across the four accounts (correct flags; no positional
    calendar id with `--cal`); Drive search; Docs export; Sheets read.
  - Notion: People DB query by name / email / company (collection ID from
    `context/ecosystem-map.md`).
- Rule at the top: "Use a recipe verbatim before reading `--help`. If a recipe fails,
  fix the recipe file in the same session."

### 4.4 Contacts index (data cache)
- Script `scripts/build-contacts-index.py` writes `data/contacts-index.json`:
  `[{name, emails[], company, sources[], notion_url?, last_seen?}]`, deduped by
  lower-cased email.
- Sources: Notion People DB (all rows), Missive contact books, calendar attendees from the
  last 18 months across the four accounts (via `gog`).
- Runs nightly as a scheduled job (02:30 ET) plus on demand. File is gitignored
  (contains PII); lives on the VPS only.
- Lookup is a plain JSON read/grep (target < 1s). `contact-lookup` falls back to live
  APIs only on a miss or when the user asks for "latest".

### 4.5 Google People API 403
- Enable the People API on the Google Cloud project behind the `nevado` `gog`
  credentials (one-time console action — Carl or Claude via console). Verify with
  `gog contacts search` on each account.

### 4.6 Track 1 acceptance
Re-run the McGovern question and three other lookups on the current bot: each answers in
< 60s with ≤ 8 tool calls and invokes `contact-lookup`.

---

## 5. Track 2 — Harness rewrite (repo: `c22os-telegram`, branch `rewrite/harness-v2`)

### 5.1 Scope
Rewrite in place. Replace `src/bot/`, `src/claude/`, and `src/bot/orchestrator.py` with a
new package `src/harness/`. **Keep:** `src/bot/features/voice_handler.py` (whisper.cpp,
moved under `src/harness/`), the scheduler + `seed_jobs.py`, the notifications service,
`config/mcp.json`, the systemd unit, the SQLite file (new tables). Target ~1.5–2k lines.
Upstream syncs stop after the rewrite; the repo becomes c22os-owned.

**v1 inputs:** text, voice notes, photos/screenshots, documents. **v1 interaction:**
inline buttons for agent questions. **Out of v1:** sending files back to the chat.

### 5.2 Components

| Unit | Responsibility | Depends on |
|---|---|---|
| `TelegramAdapter` | Turn updates (text, voice, photo, document, button tap) into `Inbound{topic_key, text, attachments, reply_to}`; never blocks the polling loop | python-telegram-bot, `voice_handler` |
| `TopicRouter` | One `TopicWorker` per `topic_key`; idle → start run, busy → inject | `AgentRun`, `SessionStore` |
| `AgentRun` | Owns one `ClaudeSDKClient` in streaming-input mode; emits `ProgressEvent`s; supports `inject(msg)`, `interrupt()`; logs every SDK event with timestamp | claude-agent-sdk |
| `StatusMessage` | One Telegram message per run, edited ≤ every 5s (current tool, turn, elapsed); final answer posted as a new message with an outcome footer | Telegram API |
| `SessionStore` | SQLite: `topic_key → session_id, created_at, last_active, transcript_bytes`; rotation; persisted run state for restart recovery | SQLite |
| `AskBridge` | Agent questions → inline keyboard; tap resumes the run; 30-min expiry | `AgentRun`, Telegram API |
| `Scheduler` | Existing APScheduler + `seed_jobs.py`; each job runs through `AgentRun` with a fresh session, posting to its topic | `AgentRun` |
| `PromptBuilder` | System prompt = `CLAUDE.md` + `context/intent-map.md` + `context/tool-recipes.md` | c22os checkout |

Each unit is testable with a fake SDK client and a fake Telegram bot.

### 5.3 Data flow
1. **New message, idle topic:** voice → whisper.cpp (~6s) → status message
   "🎙️ heard: '<first 80 chars>' · starting" → `TopicRouter` → `AgentRun` starts with
   `PromptBuilder` output → status edits ("🔧 notion-search · turn 7 · 4m12s") → final
   answer as a new message → status flips to ✅.
2. **Message during an active run:** `TopicRouter` → `AgentRun.inject()` →
   `client.query()` on the live client → 👀 reaction on Carl's message. No new status
   message.
3. **Photos/documents:** saved to `data/inbox/<topic_key>/<ts>-<name>`; path included in
   the prompt; images also passed as image content blocks.
4. **Session rotation:** before starting a run, if `last_active` > 2h ago or
   `transcript_bytes` > 400 KB, generate a ≤ 300-word carry-over summary of the old
   session with a Haiku-class model and start a new session seeded with it. `/new` forces
   rotation without a summary.

### 5.4 Error handling and limits

| Condition | Carl sees | Harness does |
|---|---|---|
| No SDK event for 5 min | "⚠️ stalled on `<tool>` for 5m, stopped. Reply to retry." | `interrupt()`, keep session |
| Run reaches 20 / 40 min | heartbeat edit "still working · 20m" | — |
| Run reaches 60 min | "⚠️ hit 60m cap" + partial output | `interrupt()` |
| SDK/CLI crash | "❌ agent crashed: <one line>" | log full trace; reset that topic's client |
| MCP server unavailable at start | "⚠️ <server> unavailable" in status | continue without it |
| Unanswered question | buttons expire at 30 min → "no answer, stopped" | `interrupt()` |
| Bot restarts mid-run | on boot: "♻️ restarted during your last request — reply to resume" | read persisted run state |
| Telegram API error | — | retry with backoff; status edits best-effort |
| `/cancel` | "⚠️ cancelled" | `interrupt()` |

**Concurrency:** runs in different topics are parallel, global cap 3; overflow shows
"⏳ queued". Scheduled jobs count toward the cap and never pre-empt interactive runs.

Timeouts, cap, rotation thresholds, and concurrency cap live in `settings.py` with the
values above as defaults.

### 5.5 Testing
- Unit tests per component with a fake `ClaudeSDKClient` (scripted event streams,
  including stalls and crashes) and a fake Telegram bot: injection, rotation, watchdog,
  60-min cap, restart recovery, button flow, concurrency cap.
- Replay tests: the 2026-10-14 stalled conversation and the 2026-10-17 McGovern lookup,
  run against a real SDK in a test topic; both must show live status and end with an
  explicit outcome.
- 80% coverage on `src/harness/`.

### 5.6 Cutover and rollback
1. Tag the current deployed commit `pre-harness-v2`.
2. Merge `rewrite/harness-v2` via PR; restart `c22os-telegram.service` (system-level).
3. 48h soak against §2 criteria, watching `journalctl -u c22os-telegram`.
4. Rollback = `git checkout pre-harness-v2 && sudo systemctl restart c22os-telegram`.
5. After a clean soak, update `workflows/telegram-bot.md` and
   `docs/runbooks/vps-runtime.md` in c22os.

## 6. Open risks
- **Unexplained 35-min gap (10/14).** The cause is not confirmed (candidates: MCP call
  not reported as a tool event, API retries, a stall outside tool calls). §5.2's per-event
  logging and the 5-min watchdog make it visible and bounded; root cause gets recorded in
  `failure-modes.md` once observed.
- **Injection semantics.** Streaming-input injection mid-turn must be validated against
  claude-agent-sdk 0.2.157 in the first implementation task; fallback is queue-behind with
  👀 → ⏳ reaction.
- **Contacts index staleness.** Up to 24h old by design; mitigated by live-API fallback.
