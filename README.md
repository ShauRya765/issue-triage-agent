# Issue Triage Agent

A LangGraph agent that takes a real open issue from [vercel/next.js](https://github.com/vercel/next.js),
extracts facts about it with an LLM, decides what to do with it using plain
Python rules, pauses for a human to approve or edit the proposed actions, and
then executes them (or, in `SHADOW_MODE`, just says what it would have done).

## The central design rule

**The model extracts, the code decides.**

| | Model (`app/extract.py`) | Code (`app/policy.py`) |
|---|---|---|
| Can decide | What the issue text says: has a version, has a repro, has logs, has expected behaviour, claimed area, confidence in that claim, kind of issue | Priority (P0-P3), which label to apply, what "enough information" means, which actions to take, **and whether a human must approve the run** |
| Cannot decide | Priority, labels, actions, sufficiency of information, whether to act alone | Anything about the issue's content -- it only ever sees facts already extracted, plus a fetched context record |

A third input sits alongside the model: `app/context.py` looks up *who filed
the issue* from the GitHub API. That's a fetch, not a judgment -- no model is
involved -- and `policy` reads it the way a human triager glances at who
opened a ticket before deciding how fast to move.

Concretely: `app/policy.py` makes zero LLM calls and imports nothing from
`app/graph.py`. Its three functions (`priority`, `component`, `actions`) are
pure functions over a `Facts` dict and a list of GitHub labels. `tests/test_policy.py`
passes with `ANTHROPIC_API_KEY` unset and no network access -- that's the
proof the decisions don't depend on the model. If you find yourself wanting
an LLM call inside `policy.py`, the fix is a new field on `Facts` extracted
by the model, not a new import.

Why this split, concretely: an LLM can misjudge severity, or drift over time
as prompts get tweaked, and there's no way to unit test "the model's
judgment." A regex for "data loss" or a check for `area_confidence == "high"`
is auditable, testable offline, and won't silently change behavior when you
swap models.

## Why `vercel/next.js`, and the label vocabulary check

Before writing any graph code, `scripts/check_labels.py` was run against the
real repo:

```bash
python scripts/check_labels.py
```

Findings (176 non-PR closed issues sampled):

```
129  invalid link      <- bot/moderation label, not a component
 24  locked
 18  Turbopack
 14  Runtime
 11  Performance
 11  Dynamic Routes
  6  Output
  5  Linking and Navigating
  4  Middleware
  4  Internationalization (i18n)
  3  TypeScript / Cache Components / Webpack
  2  Error Overlay / Headers / Parallel & Intercepting Routes / Metadata /
     Redirects / Module Resolution / Pages Router / Not Found
  1  Form (next/form) / Cookies / create-next-app / Script (next/script) /
     SWC / React / Route Handlers
```

First pass at "how many of 100 closed issues carry exactly one component
label" came back at **21/100** -- small, as flagged as the stop-and-check
threshold. The reason: `vercel/next.js` bot-closes a large fraction of
issues via `invalid link`/`locked` before a human ever triages them; those
issues were never candidates for a component label, so counting them as
eval-eligible understates the real signal.

Excluding bot-closed issues (only `invalid link`/`locked` labels) from the
pool and recomputing on the remaining genuinely-triaged issues:

```
64 genuinely-triaged closed issues
31 / 64 carry exactly one component label (~48%)
```

`COMPONENTS` (in `.env.example` / `app/policy.py`) is exactly the real
non-noise label vocabulary found by that script -- not a guessed list.
`vercel/next.js` applies component labels as bare names (`Turbopack`,
`Runtime`, ...), not with a colon prefix, so `LABEL_PREFIX` defaults to
empty; it's kept as a knob for repos that do use one.

For the problems hit while building this and why each was resolved the way it
was, see [ENGINEERING_NOTES.md](ENGINEERING_NOTES.md).

## Files

```
app/state.py     TypedDict state threaded through the graph
app/github.py    httpx client: fetch_open, fetch_closed, fetch_issue,
                 add_label, comment -- writes respect SHADOW_MODE
app/extract.py   the only LLM calls (claude-sonnet-5 via langchain_anthropic);
                 JSON-only prompt, parse returns {} on any failure
app/context.py   reporter lookup (tier + prior issue count); a fetch, no LLM,
                 fails soft to unknown
app/policy.py    every decision; pure, offline, no LLM, no graph import --
                 priority, component, actions, review_reason, validate_actions
app/graph.py     extract -> fetch_context -> decide -> propose -> (human_review
                 interrupt | auto_approve) -> execute, checkpointed to Supabase
app/main.py      FastAPI: GET /issues, POST /runs, POST /runs/{number},
                 POST /runs/{number}/approve, GET /runs/{number}
eval/replay.py   fetch closed issues, run the same extract+policy path,
                 score predicted vs actual component label
tests/test_policy.py      priority, component, actions, escalation, the
                          autonomy gate, action validation
tests/test_context.py     tier mapping and the fail-soft paths
tests/test_graph.py       routing, the pause, edited approvals, idempotency
ENGINEERING_NOTES.md      decisions, tradeoffs and what's still unproven
scripts/check_labels.py   the label-vocabulary check described above
scripts/check_db.py       connects to DATABASE_URL, runs the checkpointer's
                          setup(), reports the checkpoint tables it found
```

## Running it

Dependencies:

```bash
python3 -m venv venv
./venv/bin/pip install -r requirements.txt
cp .env.example .env   # fill in ANTHROPIC_API_KEY, DATABASE_URL
```

Unit tests (no API key, no network):

```bash
./venv/bin/python -m pytest tests/ -q
```

Eval (needs `ANTHROPIC_API_KEY`; `REPO` defaults to `vercel/next.js`):

```bash
./venv/bin/python -m eval.replay --n 20 --version v1
```

Prints accuracy and the most common want-to-got confusions, and writes
per-issue rows to `eval/results/v1.json`.

API (needs `DATABASE_URL` pointing at a real Postgres -- the durable pause
depends on it). This project uses **Supabase** for that Postgres:

1. Create a project at [supabase.com](https://supabase.com) (free tier is
   enough -- the checkpointer writes a few KB per run).
2. Project -> **Connect** -> copy a Postgres URI. Prefer the **session pooler**
   (`aws-0-<region>.pooler.supabase.com:5432`); the direct `db.<ref>.supabase.co`
   host is IPv6-only and unreachable on many home/office networks.
3. Paste it into `.env` as `DATABASE_URL`, substituting the database password
   for `[YOUR-PASSWORD]`, and keep `?sslmode=require`.
4. Confirm it works before starting the API:

```bash
./venv/bin/python scripts/check_db.py
```

That creates the `checkpoints*` tables in the `public` schema and prints them.
Then:

```bash
./venv/bin/python -m uvicorn app.main:app --reload
```

- `GET /issues?n=20` -- list open issues
- `POST /runs?n=10` -- triage a batch unattended; the entry point a cron job
  or worker calls. Returns per-issue status
- `POST /runs/{number}` -- start one run. Returns `awaiting_approval` with the
  proposed facts/priority/component/actions, or `executed_autonomously` if the
  routing gate cleared it
- `POST /runs/{number}/approve` -- body `{"actions": [...]}` to use an edited
  list, or `{}`/omitted to approve as proposed; executes and returns the log
- `GET /runs/{number}` -- inspect the checkpointed state at any point,
  including after a restart

To confirm the durability claim yourself: start a run, kill the `uvicorn`
process before approving, start it again, then `POST /runs/{number}/approve`
-- it resumes from the Postgres checkpoint, not from memory.

Supabase's transaction pooler (port 6543) also works: `app/graph.py` opens the
connection with `prepare_threshold=None`, since pgbouncer in transaction mode
rejects the prepared statements psycopg would otherwise use. The session pooler
is still the safer default because it can hold session state.

The checkpoint tables live in Supabase's `public` schema, which is exposed via
PostgREST. They contain issue text and proposed actions, not secrets, but
nothing in this app needs anon-key access to them -- leave RLS on and reach
them only through `DATABASE_URL`.

**Durability: verified.** The restart test above has been run end to end
against a real Supabase project (Postgres 17.6, session pooler). One process
started a run and exited at the `human_review` pause; a second, unrelated
process -- knowing only the issue number -- recovered the full state from
Postgres, applied an *edited* action list, and executed it. Resuming that
completed thread a second time executed nothing further, confirming the
idempotency-key check across process boundaries.

`tests/test_graph.py` covers the same control flow against `MemorySaver` so it
stays testable offline; the durability claim specifically is what needed the
real database.

## Human in the loop, and running alone

The graph has exactly one branch, after `propose`:

```
extract -> fetch_context -> decide -> propose --+--> human_review (interrupt) --+--> execute
                                                |                               |
                                                +--> auto_approve --------------+
```

Both paths converge on the same `execute` node, so an autonomous run and an
approved run execute through identical code, including the idempotency check.

**The human path.** `human_review` calls `interrupt()`, which checkpoints and
stops. The reviewer can approve as proposed, or POST a different list to
replace it wholesale -- that edited list is what executes. Edits are validated
against the repo's real label vocabulary before anything resumes
(`policy.validate_actions`), because once `execute_node` starts calling GitHub
a bad action partway down the list can't be rolled back.

**The autonomous path.** `AUTONOMOUS=false` (the default) sends every run to a
human regardless. With it on, `policy.review_reason` decides, and it fails
towards the human -- it returns the *reason* a run stopped, which gets recorded
on the state, so a paused run can be explained later without re-deriving it. A
run executes alone only when all of these hold:

| Condition | Why |
|---|---|
| Not a P0 | The case where being wrong is most expensive |
| No comment action | A label is silently removable; a comment emails every subscriber and can't be unsent |
| Component identified | Failing to route is exactly when a human adds value |
| `area_confidence == "high"` | Low confidence is the model telling you to check |
| Extraction succeeded | `extract_facts` returns `{}` on failure; don't act on an extraction that didn't happen |
| Reporter context available | Don't apply a rule whose inputs are known to be incomplete |

`POST /runs` triages a batch with no issue number in the request -- that's what
lets a scheduler start work unprompted. Every issue still goes through the same
gate.

## Customer context

`app/context.py` fetches, per issue: the reporter's login, a tier normalised
from GitHub's `author_association` (`internal` / `contributor` / `external`),
and how many issues they've previously filed on the repo.

`policy.priority` uses it to escalate **at most one step, and never to P0** --
P0 stays reserved for the hard signals (a security label, a data-loss or
build-breaking pattern). Who filed something is never on its own grounds to
call it a P0, or the label stops meaning "drop everything" and starts meaning
"someone important is watching." Escalation applies to maintainers, and to
contributors with at least `ESTABLISHED_REPORTER_ISSUES` prior filings.

A failed lookup returns `unknown=True`, which suppresses escalation entirely
and forces human review. It deliberately does *not* fall back to `external`:
that would silently deprioritise a maintainer's report the moment the search
API got rate limited.

On a commercial support desk these fields would come from a CRM -- plan tier,
contract value, open ticket count. The shape is the same either way: a fetched
record about the requester, kept separate from the request text, feeding a rule
rather than a prompt. Swapping in a CRM means rewriting `app/context.py` alone;
`Facts`, `policy`'s signature and the graph don't move.

`SHADOW_MODE` defaults to `true`: every `add_label`/`comment` call returns a
descriptive string and writes nothing to GitHub. This project doesn't own
`vercel/next.js`, so shadow mode should stay on unless you're pointing it at
a repo you control.

## Results

Run against `vercel/next.js` closed issues. Only issues carrying exactly one
`COMPONENTS` label are scored, so 15 qualified out of 80 fetched. The ~48%
single-label figure from the vocabulary check is the meaningful baseline.

| Version | n scored | Accuracy | Change |
|---|---|---|---|
| v1 | 15 | 40.0% | First real run. 3/15 extractions failed outright (`response.content` arrives as a list of content blocks on newer models, not a string -- the regex in `_parse_response` raised `TypeError`). |
| v2 | 15 | 53.3% | Content-block handling fixed; zero extraction failures. Every remaining miss abstained to `needs-triage` -- not one wrong label. |
| v3 | 15 | **66.7%** | Injected the real label vocabulary into the extraction prompt. Diagnosis: the model was confident (6/7 misses at `high`) but naming areas that don't exist -- `next-devtools`, `Use Cache`, `Telemetry`. It was guessing a taxonomy nobody had shown it. |

**v3's remaining 5 misses split 3 abstentions / 2 mislabels.** That split is the
interesting part: constraining the vocabulary traded abstentions for a small
number of genuine errors (`Performance` -> `Cache Components`, twice). Before
v3 the system never applied a wrong label, only declined to route. Which of
those failure modes is preferable is a product decision, not a technical one --
and the autonomy gate already treats them differently, since a `needs-triage`
result can never execute without a human.

**On reading too much into this:** n=15 means one issue moves accuracy by 6.7
points, so the gap between v2 and v3 is real but the precise figure isn't
stable. A larger run is the obvious next step.

Earlier runs are kept rather than overwritten: `eval/results/v1.json` is the
record of what a silently-broken extractor scores, which is the comparison that
makes the later numbers meaningful.
