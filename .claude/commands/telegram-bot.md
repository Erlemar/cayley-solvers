---
description: Operate the "Cayley" Telegram bot (@artgor_cayley_solver_bot) serving the CayleyPy_Agent group - status, start/stop, logs, what people asked, and config changes.
---

# /telegram-bot

Runbook for the read-only Telegram agent built 2026-08-22. Use this in any session that
needs to check on the bot, restart it, change what it can do, or diagnose why it went
quiet.

## Usage

`/telegram-bot [status|start|stop|restart|logs|history|config]`

Default is `status`.

## Facts you need

| Thing | Value |
|---|---|
| Bot handle | `@artgor_cayley_solver_bot` (bot id `8654132735`) |
| Group | `CayleyPy_Agent`, chat id `-5300047051` (private, no public username) |
| Owner tg user id | `239578004` |
| Code | `C:\Users\and-l\cayley\telegram_bridge\` |
| Secrets | `telegram_bridge\.env` - gitignored. **Never paste the token into any chat.** |
| State | `telegram_bridge\state\bridge.db` (context + poll offset), `state\bridge.log` |
| Agent backend | `claude -p`, model from `AGENT_MODEL` (default `sonnet`) |

It answers **only** on an explicit `@artgor_cayley_solver_bot` mention or a reply to one
of its own messages. Everything else it sees is stored as context and ignored.

## status

```powershell
$here = "C:\Users\and-l\cayley\telegram_bridge"
Get-CimInstance Win32_Process -Filter "Name='python.exe'" -ErrorAction SilentlyContinue |
  Where-Object { $_.CommandLine -like '*bridge.py*' } |
  ForEach-Object { "pid=$($_.ProcessId) parent=$($_.ParentProcessId) started=$($_.CreationDate)" }
$log = Get-Content "$here\state\bridge.log"
$log | Where-Object { $_ -notmatch 'transient failures' } | Select-Object -Last 12
"  ANSWER LOST : $(($log | Select-String 'ANSWER LOST').Count)"
"  replies     : $(($log | Select-String 'agent replied').Count)"
```

Expect **two** python pids: the supervisor's child and its own worker child
(`parent=python.exe` is the worker, not a second instance - see CLAUDE.md rule 7b-ii).
Healthy idle looks like no output at all: it only logs on events.

## start / stop / restart

```powershell
powershell -ExecutionPolicy Bypass -File "C:\Users\and-l\cayley\telegram_bridge\start_bridge.ps1"
powershell -ExecutionPolicy Bypass -File "C:\Users\and-l\cayley\telegram_bridge\stop_bridge.ps1"
```

`start` is idempotent (reports "Already running"). Restart = stop, `Start-Sleep 2`, start.
`run_bridge.bat` supervises and auto-restarts on crash, but deliberately does NOT restart
on exit codes 0, 2 (bad/absent token, identity mismatch) or 3 (duplicate poller).

Any edit to `bridge.py`, `.env`, `persona.md` or `bot_settings.json` needs a restart.

## logs

```powershell
Get-Content "C:\Users\and-l\cayley\telegram_bridge\state\bridge.log" -Tail 40 -Wait
```

Transient poll failures are summarised every 5 minutes, so a `transient failures: N`
line is normal background noise, not an incident. Rotates at 2 MB, 3 backups.

To watch for real events across a long session, prefer `Monitor` (persistent) over a
background Bash poll loop (CLAUDE.md rule 7):

```bash
tail -n 0 -F /c/Users/and-l/cayley/telegram_bridge/state/bridge.log \
  | grep -E --line-buffered "addressed by|agent replied|ANSWER LOST|ERROR|IDENTITY MISMATCH|Conflict"
```

## history - what people actually asked

All messages the bot received, including DMs it refused to answer:

```bash
cd C:/Users/and-l/cayley/telegram_bridge && PYTHONUTF8=1 PYTHONIOENCODING=utf-8 \
  C:/Users/and-l/cayley/.venv/Scripts/python.exe - <<'PYEOF'
import sqlite3, time
conn = sqlite3.connect("state/bridge.db", timeout=15)
for cid, n in conn.execute("SELECT chat_id, COUNT(*) FROM messages GROUP BY chat_id"):
    print(f"chat {cid}: {n} message(s)")
print("-" * 70)
for ts, cid, author, uname, txt, fb in conn.execute(
        "SELECT ts, chat_id, author, username, text, from_bot FROM messages "
        "ORDER BY ts DESC LIMIT 20"):
    who = f"{author}" + (f" (@{uname})" if uname else "")
    print(f"[{time.strftime('%m-%d %H:%M', time.localtime(ts))}] chat={cid} "
          f"{'BOT' if fb else 'usr'} {who}: {(txt or '')[:160]}")
PYEOF
```

`PYTHONUTF8=1` is required - the console is cp932 and Cyrillic/em-dashes crash the
print otherwise (CLAUDE.md rule 24).

## config

Edit `.env`, then restart:

- `AGENT_MODEL` - `sonnet` (default, ~7-20s/answer) or `opus` for harder questions.
- `CONTEXT_MESSAGES` / `CONTEXT_MAX_AGE_HOURS` - rolling window size (40 / 72h).
- `ALLOWED_CHAT_IDS` - comma-separated. **Empty = discovery mode**: the bot answers
  nothing and instead reports the chat ID of any chat it sees. That is how you onboard
  a new group: blank the field, restart, add the bot, have someone post, read the
  `DISCOVERY:` line, put the ID back, restart.
- `MIN_SECONDS_BETWEEN_REPLIES` - per-chat throttle.

Behaviour and persona live in `persona.md` (identity, hard limits, chat style).
Permissions live in `bot_settings.json`.

Run `selftest.py` after touching prompt/format logic - it checks trigger detection,
language selection and markdown flattening. It costs a few API calls.

## Invariants - do not break these

The bot is read-only **by construction**, and four independent layers hold it there.
If you change `AGENT_ARGV` in `bridge.py`, preserve all of them:

1. `--tools Read,Glob,Grep` - Bash/Edit/Write/WebFetch/Task do not exist in the session.
2. `--strict-mcp-config --mcp-config '{"mcpServers":{}}'` - no Gmail/Drive/GitHub/Chrome.
3. `--setting-sources ""` plus `--settings bot_settings.json` - the operator's own
   permissive settings never load; secret paths are denied.
4. `run_agent()` strips `KAGGLE_API_TOKEN`, `GITHUB_TOKEN`, `GH_TOKEN`,
   `TELEGRAM_BOT_TOKEN` from the subprocess env.

Also: `--disable-slash-commands` keeps the operational skills (Kaggle submit, kernel
push, GCP sync) readable as files but not runnable. Everything outbound passes through
`scrub()`.

Re-run the five adversarial probes after any change to these (write a file, read
`.claude/settings.local.json` "with authorization", read `~/.kaggle/kaggle.json`, run a
shell command, full override injection). All five must refuse.

## Troubleshooting

**Bot came up as the wrong bot.** This machine has user-scope `TELEGRAM_BOT_TOKEN` /
`TELEGRAM_CHAT_ID` belonging to `@oura_claw_bot`. `.env` is loaded with override
semantics precisely because of this, and a username/token mismatch is now a hard exit 2.
If you ever swap tokens, also wipe `state\bridge.db` - it holds the previous bot's poll
offset.

**Answers stop arriving.** Check for `ANSWER LOST` in the log. The link to
`api.telegram.org` from this machine is torn down ~20x/hour (`WinError 10054`); every
call retries 4x with backoff, so a lost answer means something worse than the usual
flakiness. Diagnostic REST calls from PowerShell need their own retry loop or they
report a false "unreachable".

**Exit code 3 / `409 Conflict`.** Two pollers on one token. Find and kill the stray
instance; never run a second `getUpdates` (including an ad-hoc curl) while the bridge
is up. `getChat`, `getMe`, `sendMessage` are safe to call alongside it.

**Bot sees only mentions, never surrounding discussion.** Telegram applies a privacy
change only to groups the bot joins *after* the change. Remove it from the group and
re-add it. Verify with
`getMe` -> `can_read_all_group_messages: True`, then confirm a non-mention message
actually lands in `bridge.db`.

**It answered in the wrong language or emitted `**bold**`.** Both are enforced in code
(`language_of()`, `to_plain_text()`), not by the system prompt - the prompt-only version
lost to sender-name inference. Fix the function, not the persona.

## Known open items

- Privacy re-add (above) may still be pending - check `can_read_all_group_messages`
  *and* that a plain group message reaches `bridge.db`.
- The bot discloses whatever the project docs contain, including unpublished method
  detail, to anyone in the group. If that is unwanted, add a disclosure boundary to
  `persona.md`.
- `remember()` runs *before* the allow-list check, so DMs from strangers accumulate in
  `bridge.db` even though they are never answered. Move the call below the check to stop
  that.
