# Engineering notes

A log of the problems that actually came up building this, what was decided,
and why. Written so the reasoning survives without re-reading the code.

Each entry is: what happened → what was done → the point worth making about it.

---

## 1. The label vocabulary was guessed, then measured, then measured again

**What happened.** The first instinct was to invent a component list
(`routing`, `build`, `styling`...). None of those are labels `vercel/next.js`
actually uses. An agent proposing labels that don't exist would fail on every
single issue, and the eval would have been scoring against fiction.

**What was done.** Wrote `scripts/check_labels.py` *before* any graph code and
ran it against the real repo. It sampled 176 closed non-PR issues and counted
label frequency. `COMPONENTS` is that output verbatim.

**The twist.** The first sanity check — "how many closed issues carry exactly
one component label?" — came back **21/100**. That was low enough to stop and
question the whole premise. The cause: `vercel/next.js` bot-closes a large
share of issues (`invalid link`, `locked`) before a human ever triages them.
Those issues were never candidates for a component label, so counting them
understated the real signal. Excluding them: **31/64, ~48%**.

**Point worth making.** The number that made me stop wasn't wrong, it was
measuring the wrong population. Worth checking what's *in* your denominator
before concluding the task is impossible. The second measurement also set the
eval's ceiling honestly — ~48% of issues are single-labelled, so that's the
realistic target, not 100%.

---

## 2. The central constraint: the model extracts, the code decides

**The problem.** The easy build is one prompt: "here's an issue, output a
priority and a label." It demos well and is untestable. You can't unit test
"the model's judgment," it drifts when you tweak the prompt, and it changes
silently when you swap model versions.

**What was done.** A hard split.
- `app/extract.py` is the only file that calls an LLM. It asks only what the
  text *says* — has a version, has a repro, has logs, claimed area, confidence
  — plus a verbatim quote as evidence for each.
- `app/policy.py` makes every decision and imports no model, no graph, nothing
  network-touching. `priority`, `component`, `actions`, `review_reason`,
  `validate_actions` are pure functions.

**The proof.** `tests/test_policy.py` passes with `ANTHROPIC_API_KEY` unset and
no network. That's not a convenience — it's the evidence the decisions don't
depend on the model.

**The rule that keeps it honest.** If you want an LLM call inside `policy.py`,
the answer is a new *field on `Facts`*, extracted by the model, not a new
import. That rule is what stops the boundary eroding over time.

**Point worth making.** "P0 requires a matched signal, never the model's
say-so." A regex for data loss is auditable and testable; a model's opinion on
severity is neither. The model is used where it's genuinely better than code
(reading messy prose) and nowhere else.

---

## 3. LangGraph's interrupt doesn't come back from `invoke()`

**What happened.** The natural assumption is that `graph.invoke()` returns the
interrupt payload when the graph pauses. It doesn't — it returns the state
accumulated so far. The interrupt lives on the *paused task*, not the state.

**What was done.** `_pending_interrupt()` in `app/main.py` reads it off
`graph.get_state(config).tasks[*].interrupts[0].value`. That same call is how
the code distinguishes "paused for a human" from "ran to completion
autonomously" — no pending interrupt means the run finished on its own.

**Point worth making.** Small API detail, but it's the difference between a
HITL endpoint that works and one that returns an empty object and looks broken.

---

## 4. Conditional edges re-run on every resume

**What happened.** The obvious place to decide "human or autonomous?" is inside
the router function on the conditional edge. That's a bug waiting to happen:
LangGraph evaluates the router again on resume, so if anything it reads had
changed in between, a run could pause for a human and then take the autonomous
branch on the way back.

**What was done.** `propose_node` computes the decision *once*, writes
`review_required` / `review_reason` onto the state, and `route_after_propose`
only reads the stored flag. Decisions get made in nodes and persisted; routers
only read.

**Point worth making.** A general principle for checkpointed graphs — anything
evaluated more than once must be a pure read of committed state, or the run
isn't deterministic across a resume.

---

## 5. The idempotency ledger was being wiped on re-run

**What happened.** The thread id is `repo#number` — permanent per issue, which
is what makes "resume this issue's run" work. But `start_run` seeded the state
with `"executed_keys": []` on every call. Starting a second run on the same
issue therefore erased the record of what the first run had already executed.

**Severity.** Labels survive it (GitHub dedupes server-side), but **comments
would double-post** — and the missing-info comment is exactly the action most
likely to repeat. A user gets pinged twice asking for the same reproduction.

**How it was found.** Not by reading the code — by writing a test that ran the
same issue through twice and asserted on the GitHub calls:

```
after run 1: [('label','P1'), ('label','Turbopack')]
after run 2: [('label','P1'), ('label','Turbopack'), ('label','P1'), ('label','Turbopack')]
```

**The fix.** `_initial_state()` reads the existing checkpoint and carries
`executed_keys` forward. A fresh thread gets `[]`; an existing one keeps its
history. Locked in by `test_rerunning_the_same_issue_does_not_re_execute`.

**Point worth making.** The idempotency *mechanism* (hash-based keys, checked
before each call) was correct. The bug was in the state seeding around it — the
guarantee was real within a run and silently absent across runs. That's the
kind of gap that only shows up when you test the boundary rather than the unit.

---

## 6. Where the human sits, and when the agent may act alone

**The tension.** A human-in-the-loop agent that always stops isn't an agent,
it's a form. One that never stops is unsafe on a repo you don't own.

**What was done.** One branch in the graph, after `propose`, with both paths
converging on the same `execute` node — so autonomous and approved runs go
through identical code, including the idempotency check.

The gate (`policy.review_reason`) returns the *reason* a run needs a human, or
`None`. Returning a reason rather than a bool means a paused run is explainable
after the fact without re-deriving the decision — it's recorded on the state.

A run may act alone only if: not a P0, no comment action, component identified,
`area_confidence == "high"`, extraction succeeded, and reporter context was
available.

**The criterion that organises that list: reversibility.** A label is silently
removable and notifies nobody. A comment emails every subscriber to the issue
and cannot be unsent — a wrong one is a public mistake on a stranger's issue.
So label-only runs can be delegated and anything that writes to a person can't.

**Fails towards the human.** Every branch not positively known to be safe
returns a reason. `AUTONOMOUS` defaults to `false`, so the shipped behaviour is
that everything stops.

**Point worth making.** "When may it act alone?" is a policy question, and it
lives in the same pure, offline-testable module as every other decision — not
in a prompt, not as an emergent property of a confidence score.

---

## 7. Customer context: fetched, never inferred

**The requirement.** A human triager checks who filed something before deciding
urgency. The first version of this project had nothing — `Issue` didn't even
record the reporter's name, so a first-time anonymous filer and a core
maintainer scored identically.

**What was done.** `app/context.py` fetches the reporter's tier (normalised
from GitHub's `author_association`) and their prior issue count on the repo.
No model involved — it's a lookup.

**Three decisions worth defending:**

*Escalate at most one step, never to P0.* Who filed something is never on its
own grounds to call it a P0, or the label stops meaning "drop everything" and
starts meaning "someone important is watching." Once that happens the priority
scale is dead.

*A failed lookup is `unknown`, not `external`.* This is the one that matters.
Defaulting a rate-limited lookup to "external" would silently deprioritise a
maintainer's report the moment the API got slow — a failure that looks exactly
like normal operation. `unknown=True` suppresses escalation *and* forces human
review.

*Unrecognised association values fall through to `external`.* GitHub has added
values to that field before (`FIRST_TIME_CONTRIBUTOR`). Unknown input must land
on the conservative side, which here means granting no escalation.

**On the OSS-vs-CRM question** (likely to be asked): on a commercial desk these
fields come from a CRM — plan tier, contract value, open tickets. This repo is
public, so it uses the closest equivalents the platform actually exposes. The
*shape* is what transfers: a fetched record about the requester, kept separate
from the request text, feeding a rule rather than a prompt. Swapping in a CRM
rewrites `app/context.py` alone — `Facts`, `policy`'s signature and the graph
don't move.

---

## 8. The approve endpoint accepted arbitrary JSON

**What happened.** Letting a reviewer replace the proposed action list is the
whole point of the human step. But the edited list went straight from the HTTP
body into `Command(resume=...)` and then into `execute_node`, which does
`action["type"]` and `action["label"]`. A typo'd key raised `KeyError`
*partway through the loop* — after earlier actions had already hit the GitHub
API and couldn't be taken back.

**What was done.** `policy.validate_actions` checks the entire list up front,
so execution is all-or-nothing. It also checks labels against the repo's real
vocabulary, so an approved-but-misspelled label can't quietly create a brand
new label on someone else's repo. Invalid input returns 400 before the graph
resumes at all.

**Point worth making.** Trusting the human is not the same as trusting the
payload. The reviewer is authorised; their JSON still isn't validated.

---

## 9. Infrastructure problems that ate real time

**No local Postgres or Docker in the dev environment.** The durable pause needs
a real checkpoint store. Interrupt/resume was verified against `MemorySaver`
first — which proves control flow (pauses correctly, resumes with the right
actions, a second resume executes nothing) but explicitly *not* durability
across a process restart. Being clear about what a test does and doesn't prove
matters more than claiming it passed.

**Supabase's transaction pooler breaks prepared statements.** Port 6543 is
pgbouncer in transaction mode; psycopg3 uses prepared statements by default and
`PostgresSaver` depends on them. Symptom: `prepared statement "_pg3_0" already
exists`. Fix: build the connection explicitly with `prepare_threshold=None`
instead of using `PostgresSaver.from_conn_string`. Costs a little per-query
planning, works on either pooler.

**Supabase's direct connection is IPv6-only.** `db.<ref>.supabase.co` is
unreachable from most home and office networks and fails as a timeout, which
reads like a wrong password. The session pooler (port 5432) is IPv4 and is the
right default to document.

**`python-dotenv` was a dependency that nothing ever called.** The README said
"fill in `.env`", but no module loaded it — `uvicorn` would never have seen
`DATABASE_URL`. The fix had to go in `app/__init__.py`, not `main.py`, because
`app/policy.py` resolves `COMPONENTS` and `LABEL_PREFIX` at *import* time, so
the load has to happen before any submodule is imported.

**A LangGraph node can't share a name with a state key.** `add_node("context",
...)` raises `'context' is already being used as a state key`. The node is
`fetch_context`; the state key stays `context`.

**GitHub returns `null` for deleted accounts' `user` field**, and the search API
is rate limited at 10 req/min unauthenticated (vs 5000/hr for REST). Both are
normalised in `app/github.py` rather than left for callers to discover.

---

## 10. A deprecated parameter silently zeroed every extraction

**What happened.** The first real run against live Supabase and a live API key
triaged a textbook bug report -- version, stack trace, repro link, stated
expectation -- as `P3` / `needs-triage`. Every extracted fact came back
`False`, `kind` came back `question`, confidence `low`.

That is exactly the shape of `extract_facts` returning `{}`, its failure path.
The real cause, once the swallowed exception was surfaced:

```
400 invalid_request_error: `temperature` is deprecated for this model
```

`ChatAnthropic(model=..., temperature=0, ...)` -- a parameter that had been
harmless for years -- is rejected outright by newer Claude models. Every single
call 400'd, and `except Exception: return {}` turned a hard configuration error
into "this issue apparently contains no information."

**What was done.** Dropped the parameter, and added a `logger.warning` on both
failure paths. The fail-closed contract is unchanged -- policy still treats
`{}` conservatively and the graph keeps running -- but the failure is now
audible. After the fix the same issue triages as `P1` / `Turbopack` with
`area_confidence: high`.

**Point worth making.** Fail-closed and fail-silent are different things, and
I had conflated them. Degrading to "we know nothing" was the right *behaviour*
— nothing crashed, and the conservative path meant a human saw it. But with no
log line, a total model outage was indistinguishable from a genuinely empty
issue, and the system would have gone on confidently mislabelling everything as
`needs-triage` forever. The tell was noticing that a *specific* output looked
wrong for its input, not any error surfacing on its own.

---

## 11. The eval, and the two bugs it surfaced

**40% → 53.3% → 66.7%**, across three runs on the same 15 issues.

**v1 (40%).** The `logger.warning` added in note 10 paid for itself on the first
run: 3 of 15 extractions failed with `TypeError: expected string or bytes-like
object, got 'list'`. `langchain_anthropic` returns a plain string for simple
replies, but newer models return a *list of content blocks*. Passing that to
`re.search` raises, and the bare `except` turned it into `{}`. Fixed by
normalising both shapes in `_content_to_text`.

**v2 (53.3%).** Zero extraction failures — and every remaining miss was
`needs-triage`. Not one wrong label. So the question wasn't "why is it wrong",
it was "why won't it commit".

**The diagnosis.** Re-ran extraction on just the 7 misses and printed what the
model actually claimed:

```
 actual             model claimed_area        conf
 Error Overlay      next-devtools             high
 create-next-app    next upgrade CLI          high
 Performance        Use Cache                 high
 Cache Components   Navigation                high
 Runtime            Telemetry                 high
```

Six of seven at **high** confidence. It wasn't uncertain — it was naming areas
that don't exist. The prompt asked for "which product area this belongs to"
with two examples and never showed the closed list, so the model invented a
plausible taxonomy and `policy.component` correctly dropped every one.

**v3 (66.7%).** Injected `policy.COMPONENTS` into the system prompt with an
instruction to copy one value exactly or return null.

**Point worth making.** The confidence score was honest and useless. The model
was genuinely confident about an answer drawn from the wrong vocabulary, so no
confidence threshold could have caught it — tightening the gate would have
abstained more, loosening it would have applied labels that don't exist. The
fix had to be giving the model the vocabulary, not tuning how much to trust it.
Worth noting this stays on the right side of the architecture: the vocabulary
is code-owned, derived by `scripts/check_labels.py`, and `policy.component`
still decides whether the claim is usable.

**The honest caveat.** v3's 5 misses split 3 abstentions / 2 mislabels —
constraining the vocabulary traded abstentions for a couple of genuine errors.
Before v3 the system never applied a wrong label, only declined to route. Which
failure mode you prefer is a product decision. And at n=15 one issue is worth
6.7 points, so the direction is real but the number isn't stable.

---

## 12. What is still unproven

Worth saying out loud rather than being caught on:

- **The eval is small.** n=15 scored, so one issue moves accuracy by 6.7
  points. v3's 66.7% clears the ~48% single-label baseline, but a larger run is
  needed before treating the figure as settled.
- **The 2 mislabels in v3 are unexamined.** `Performance` -> `Cache Components`
  happened twice, which suggests a systematic confusion between two genuinely
  adjacent areas rather than noise.
- **One long-lived connection, no pool.** `get_graph()` holds a module-global
  connection forever. Supabase closes idle connections, so a run after a quiet
  period may hit a dead socket where local Postgres would have tolerated it.
  `psycopg_pool` is already a dependency; switching is the known next step.
- **`review_reason`'s thresholds are judgment, not measurement.**
  `ESTABLISHED_REPORTER_ISSUES = 3` was chosen, not derived. With the eval run
  and some labelled outcomes it could be tuned — but it's a constant in a pure
  function, which is exactly where a tunable number should live.

---

## Questions this design invites, and the short answers

**"Why not just let the model decide priority?"** Because you can't unit test
it, it drifts when prompts change, and it changes silently on a model upgrade.
The model reads prose; the code makes decisions. `tests/test_policy.py` passing
with no API key is the proof.

**"Isn't a regex for 'data loss' brittle?"** Yes, and deliberately narrow. A
false negative means a P0 gets scored P1, which the human reviewer catches. A
false positive means a bogus P0, which erodes trust in the label. The failure
modes aren't symmetric, so the rule is tuned to fail in the cheaper direction.

**"What stops it double-commenting on a retry?"** Hash-based idempotency keys
recorded on the checkpointed state and checked before each call — plus the fix
in note 5, because the mechanism was right and the state seeding around it
wasn't.

**"How does the human edit an action?"** `POST /runs/{n}/approve` with a
replacement list. It's validated against the real label vocabulary before the
graph resumes, so execution is all-or-nothing.

**"When does it act without a human?"** Only when the action is reversible
(labels, never comments), the component was confidently identified, it isn't a
P0, and every input was actually available. Default is off.

**"What would you do next?"** Run the eval to get a real accuracy number, then
tune `component`'s confidence gate against the confusions it reports. Then
swap the single connection for a pool. In that order — the eval tells you
whether the routing is good enough to matter.
