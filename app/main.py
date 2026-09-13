"""FastAPI surface over the graph.

POST /runs/{number} starts a run and blocks until it hits the human_review
interrupt, returning the proposed actions for a human to look at.
POST /runs/{number}/approve resumes that same thread from its checkpoint --
including after a process restart, since the checkpoint lives in Postgres --
and executes the (possibly edited) action list.
"""

import os

from fastapi import FastAPI, HTTPException
from langgraph.types import Command
from pydantic import BaseModel

from app import github, policy
from app.graph import get_graph, thread_config

app = FastAPI(title="Issue Triage Agent")


def _repo() -> str:
    return os.environ.get("REPO", "vercel/next.js")


class ApproveRequest(BaseModel):
    # None means "approve as proposed". A list means "use this instead" --
    # the human's edited action list, in the same shape as proposed_actions.
    actions: list[dict] | None = None


@app.get("/issues")
def list_issues(n: int = 20):
    return github.fetch_open(_repo(), n=n)


def _initial_state(graph, config: dict, issue: dict) -> dict:
    """Build the starting state, preserving what this thread has already done.

    executed_keys is the idempotency ledger, and the thread id is permanent per
    issue -- so seeding it with [] on every start would wipe the record of what
    a previous run already executed and let a re-run comment twice on the same
    issue. Carry it forward instead: a fresh thread gets [], an existing one
    keeps its history.
    """
    existing = graph.get_state(config).values
    return {
        "issue": issue,
        "executed_keys": existing.get("executed_keys", []),
        "execution_log": existing.get("execution_log", []),
    }


@app.post("/runs/{number}")
def start_run(number: int):
    """Start a run. Pauses for approval, unless AUTONOMOUS cleared it to proceed."""
    repo = _repo()
    graph = get_graph()
    config = thread_config(repo, number)

    issue = github.fetch_issue(repo, number)
    result = graph.invoke(_initial_state(graph, config, issue), config)

    interrupt_payload = _pending_interrupt(graph, config)
    if interrupt_payload is not None:
        return {"status": "awaiting_approval", **interrupt_payload}

    # No interrupt pending means the routing step cleared this run to act alone.
    return {
        "status": "executed_autonomously",
        "priority": result.get("priority"),
        "component": result.get("component"),
        "context": result.get("context"),
        "executed_keys": result.get("executed_keys", []),
        "execution_log": result.get("execution_log", []),
    }


@app.post("/runs/{number}/approve")
def approve_run(number: int, request: ApproveRequest):
    repo = _repo()
    graph = get_graph()
    config = thread_config(repo, number)

    state = graph.get_state(config)
    if not state.next:
        raise HTTPException(404, "no run waiting on approval for this issue")

    proposed = state.values.get("proposed_actions", [])
    submitted = request.actions if request.actions is not None else proposed

    # Validate before resuming. Once execute_node starts calling GitHub, a bad
    # action partway down the list can't be rolled back -- the earlier ones
    # have already happened.
    try:
        approved = policy.validate_actions(submitted)
    except policy.InvalidAction as exc:
        raise HTTPException(400, str(exc)) from exc

    result = graph.invoke(Command(resume=approved), config)
    return {
        "priority": result.get("priority"),
        "component": result.get("component"),
        "executed_keys": result.get("executed_keys", []),
        "execution_log": result.get("execution_log", []),
    }


@app.post("/runs")
def triage_batch(n: int = 10):
    """Triage the n most recent open issues in one pass -- the unattended entry point.

    This is what makes the agent able to start work on its own: nothing here
    names a single issue, so a cron job or worker can call it on a schedule.
    Each issue still goes through the same graph and the same routing gate, so
    with AUTONOMOUS off every one of them simply parks awaiting approval, and
    with it on only the cleared ones execute.

    Issues are processed sequentially rather than concurrently: the checkpointer
    holds one connection, and GitHub's search rate limit is the real ceiling
    anyway, so parallelism would buy little and complicate failure handling.
    """
    repo = _repo()
    graph = get_graph()
    summary = []

    for issue in github.fetch_open(repo, n=n):
        number = issue["number"]
        config = thread_config(repo, number)
        try:
            result = graph.invoke(_initial_state(graph, config, issue), config)
        except Exception as exc:
            # One bad issue must not abort the batch. Record and continue.
            summary.append({"number": number, "status": "error", "detail": str(exc)})
            continue

        if _pending_interrupt(graph, config) is not None:
            state = graph.get_state(config).values
            summary.append(
                {
                    "number": number,
                    "status": "awaiting_approval",
                    "priority": state.get("priority"),
                    "component": state.get("component"),
                    "review_reason": state.get("review_reason", ""),
                }
            )
        else:
            summary.append(
                {
                    "number": number,
                    "status": "executed_autonomously",
                    "priority": result.get("priority"),
                    "component": result.get("component"),
                    "execution_log": result.get("execution_log", []),
                }
            )

    return {"repo": repo, "count": len(summary), "results": summary}


@app.get("/runs/{number}")
def get_run(number: int):
    repo = _repo()
    graph = get_graph()
    config = thread_config(repo, number)

    state = graph.get_state(config)
    if state.values == {}:
        raise HTTPException(404, "no run found for this issue")
    return {
        "values": state.values,
        "next": state.next,
    }


def _pending_interrupt(graph, config: dict):
    """Read the pending interrupt's payload off the checkpointed state.

    invoke() only returns the state produced so far, not the interrupt
    itself -- the interrupt lives on the paused task in get_state().
    """
    state = graph.get_state(config)
    for task in state.tasks:
        if task.interrupts:
            return task.interrupts[0].value
    return None
