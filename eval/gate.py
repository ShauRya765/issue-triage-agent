"""Re-score a finished eval run through the autonomy gate, with no model calls.

eval/replay.py answers "how often is the label right?". That is not the number
a person deploying this needs. `AUTONOMOUS=true` only lets through the runs
`policy.review_reason` positively clears, so the question that matters is the
mislabel rate *among the runs that would have executed alone* -- and, just as
importantly, how many runs that is. A gate that catches every mistake by
clearing almost nothing is not a deployment story.

This module answers that from a stored results file. It makes no LLM calls: it
replays each row's recorded facts plus the issue's real body and labels (from
eval/cache/) back through app.policy, exactly the path app/graph.py takes, and
reports what the gate would have done. That means the gate can be measured
without re-running a $1-and-ten-minutes extraction, and re-measured for free
whenever the gate changes.

Two honest limits on the numbers it prints:

- It re-scores the gate, not the model. Facts are replayed as recorded, so this
  says what the gate would have done with *that* extraction, not what a rerun
  would produce.
- It needs the `facts` and `reporter_context` keys replay now writes. Results
  files from before those existed (v1-v5) cannot be scored and are rejected by
  name rather than scored with defaults standing in for data.

Run: python -m eval.gate --version v6
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from app import policy
from app.state import ReporterContext, facts_from_raw
from eval import replay

ABSTAIN = replay.ABSTAIN


class UnscorableResults(SystemExit):
    """The results file predates the fields the gate needs."""


def _issues_by_number(repo: str, count: int) -> dict[int, dict]:
    """The cached issues, keyed by number.

    Body and labels are read back from the cache rather than copied onto every
    row: policy.priority matches its data-loss and build-breaking patterns
    against the full issue body, and 150 Next.js issue bodies would be an order
    of magnitude more file than the results they'd be stored in.
    """
    path = replay._cache_path(repo)
    if not path.exists():
        raise UnscorableResults(
            f"no cached issues at {path}; the gate needs each issue's body and "
            f"labels. Run: python -m eval.replay --version <tag>"
        )
    cached = json.loads(path.read_text())
    return {issue["number"]: issue for issue in cached["issues"][:count]}


def _require_gate_fields(summary: dict) -> None:
    rows = summary.get("rows", [])
    if not rows:
        raise UnscorableResults("results file has no rows")
    if "facts" not in rows[0]:
        raise UnscorableResults(
            f"{summary.get('version', 'this file')} has no per-row `facts`, so the "
            f"gate cannot be replayed from it -- the recorded run predates that "
            f"key (v1-v5 all do). Re-run: python -m eval.replay --version <tag>"
        )


def score_row(row: dict, issue: dict) -> dict:
    """Replay one recorded run through decide -> propose -> the gate.

    The call sequence mirrors app/graph.py's decide_node and propose_node
    exactly, on purpose: anything reimplemented here instead of imported would
    be measuring a gate that never shipped.
    """
    facts = facts_from_raw(row["facts"], issue["body"])
    stored_context = row.get("reporter_context")
    context: ReporterContext | None = (
        ReporterContext(**stored_context) if stored_context else None
    )

    priority_value = policy.priority(facts, issue["labels"], context)
    component_value = policy.component(facts)
    proposed = policy.actions(facts, priority_value, component_value)
    reasons = policy.review_reasons(
        facts, priority_value, component_value, proposed, context
    )

    # An abstention is "needs-triage" either way; every other prediction must
    # reproduce, or policy.COMPONENTS has changed since the run and the numbers
    # below would be a mix of two gates.
    predicted = replay._strip_prefix(component_value)

    return {
        "number": row["number"],
        "url": row["url"],
        "actual": row["actual"],
        "predicted": predicted,
        "recorded_predicted": row["predicted"],
        "correct": predicted == row["actual"],
        "outcome": (
            "correct"
            if predicted == row["actual"]
            else ("abstained" if predicted == ABSTAIN else "mislabelled")
        ),
        "priority": priority_value,
        "component": component_value,
        "cleared": not reasons,
        "reasons": reasons,
        "context_known": bool(stored_context) and not stored_context["unknown"],
        "context_recorded": stored_context is not None,
    }


def summarise_gate(scored: list[dict]) -> dict:
    """The four numbers, plus the per-condition breakdown behind them."""
    total = len(scored)
    cleared = [s for s in scored if s["cleared"]]
    stopped = [s for s in scored if not s["cleared"]]

    outcomes = Counter(s["outcome"] for s in scored)
    mislabelled = [s for s in scored if s["outcome"] == "mislabelled"]
    abstained = [s for s in scored if s["outcome"] == "abstained"]

    mislabelled_cleared = [s for s in mislabelled if s["cleared"]]
    # An abstention always trips "component could not be determined". Whether a
    # *second* condition also fired is the interesting part: if it never does,
    # that one condition is the whole gate for this bucket, and anything that
    # weakened it would let every abstention straight through.
    abstained_multi = [s for s in abstained if len(s["reasons"]) >= 2]

    by_condition = []
    for condition in policy.GATE_CONDITIONS:
        tripped = [s for s in scored if condition in s["reasons"]]
        # Sole reason: had this condition not fired, the run would have cleared.
        # A condition that is never the sole reason never changed an outcome on
        # this sample -- it is decoration here, whatever it would do elsewhere.
        sole = [s for s in tripped if len(s["reasons"]) == 1]
        by_condition.append(
            {
                "condition": condition,
                "tripped": len(tripped),
                "sole_reason": len(sole),
                "mislabels_caught": sum(
                    1 for s in tripped if s["outcome"] == "mislabelled"
                ),
                "mislabels_caught_solely": sum(
                    1 for s in sole if s["outcome"] == "mislabelled"
                ),
            }
        )

    def rate(count: int, denominator: int) -> float:
        return count / denominator if denominator else 0.0

    return {
        "n": total,
        "cleared_n": len(cleared),
        "clearance_rate": rate(len(cleared), total),
        "stopped_n": len(stopped),
        "outcomes": dict(outcomes),
        "mislabels_n": len(mislabelled),
        "mislabels_cleared_n": len(mislabelled_cleared),
        "mislabels_cleared": [
            {"number": s["number"], "actual": s["actual"], "predicted": s["predicted"]}
            for s in mislabelled_cleared
        ],
        "abstentions_n": len(abstained),
        "abstentions_stopped_twice_n": len(abstained_multi),
        # The number that matters to someone deploying this: of the runs that
        # would have gone out unsupervised, how many put a wrong label on
        # someone else's issue.
        "cleared_mislabel_rate": rate(len(mislabelled_cleared), len(cleared)),
        "cleared_accuracy": rate(
            sum(1 for s in cleared if s["outcome"] == "correct"), len(cleared)
        ),
        "cleared_wilson_ci95": list(
            replay.wilson_interval(
                sum(1 for s in cleared if s["outcome"] == "correct"), len(cleared)
            )
        ),
        "reasons_per_stopped_run": (
            round(sum(len(s["reasons"]) for s in stopped) / len(stopped), 2)
            if stopped
            else 0.0
        ),
        "by_condition": by_condition,
        "context_recorded_n": sum(1 for s in scored if s["context_recorded"]),
        "context_known_n": sum(1 for s in scored if s["context_known"]),
        "prediction_drift_n": sum(
            1 for s in scored if s["predicted"] != s["recorded_predicted"]
        ),
        "rows": scored,
    }


def print_gate_report(gate: dict, version: str) -> None:
    n = gate["n"]
    cleared = gate["cleared_n"]
    print()
    print(f"autonomy gate replayed over {version} ({n} issues)")
    print("(policy.review_reason as it stands; no extraction re-run)")
    print()

    if gate["prediction_drift_n"]:
        print(
            f"WARNING: {gate['prediction_drift_n']} rows now predict a different "
            f"component than the recorded run. policy.component or COMPONENTS "
            f"changed since; the gate numbers below are not comparable to that "
            f"run's accuracy.",
            file=sys.stderr,
        )
        print()

    if gate["context_recorded_n"] < n:
        print(
            f"note: {n - gate['context_recorded_n']} of {n} rows carry no reporter "
            f"context, so the {policy.NO_CONTEXT_REASON!r} condition is unmeasured "
            f"for them and priority escalation was skipped",
            file=sys.stderr,
        )
        print()

    print("what the gate would do:")
    print(
        f"  cleared to execute alone   {cleared:4d}/{n}  {gate['clearance_rate']:6.1%}"
    )
    print(
        f"  stopped for a human        {gate['stopped_n']:4d}/{n}  "
        f"{1 - gate['clearance_rate']:6.1%}"
    )
    print()

    print("does it stop the mistakes:")
    print(
        f"  mislabels in the run       {gate['mislabels_n']:4d}"
        f"   ({gate['mislabels_n'] - gate['mislabels_cleared_n']} stopped, "
        f"{gate['mislabels_cleared_n']} would have executed alone)"
    )
    for miss in gate["mislabels_cleared"]:
        print(
            f"    #{miss['number']}  wanted {miss['actual']!r}, "
            f"applied {miss['predicted']!r}"
        )
    print(
        f"  abstentions                {gate['abstentions_n']:4d}"
        f"   ({gate['abstentions_stopped_twice_n']} of them tripped a second "
        f"condition too)"
    )
    lo, hi = gate["cleared_wilson_ci95"]
    print()
    print("the number that matters if you turn it on:")
    print(
        f"  mislabel rate among cleared runs  {gate['cleared_mislabel_rate']:6.1%}"
        f"  ({gate['mislabels_cleared_n']}/{cleared})"
    )
    print(
        f"  accuracy among cleared runs       {gate['cleared_accuracy']:6.1%}"
        f"   95% CI [{lo:.1%}, {hi:.1%}]"
    )
    print()

    print("which condition did the stopping:")
    print(f"  {'condition':<46} {'trip':>5} {'sole':>5} {'mis':>5} {'mis!':>5}")
    for row in gate["by_condition"]:
        print(
            f"  {row['condition']:<46} {row['tripped']:>5} {row['sole_reason']:>5} "
            f"{row['mislabels_caught']:>5} {row['mislabels_caught_solely']:>5}"
        )
    print(
        "  trip = runs tripping it; sole = runs it alone stopped; "
        "mis = mislabels it caught; mis! = mislabels it alone caught"
    )
    print(
        f"  average conditions tripped per stopped run: "
        f"{gate['reasons_per_stopped_run']}"
    )


def run_gate(version: str) -> dict:
    path = replay.RESULTS_DIR / f"{version}.json"
    if not path.exists():
        raise UnscorableResults(f"no results file at {path}")
    summary = json.loads(path.read_text())
    _require_gate_fields(summary)

    issues = _issues_by_number(
        summary["repo"], summary.get("funnel", {}).get("fetched", 0) or 10**9
    )
    scored = []
    for row in summary["rows"]:
        issue = issues.get(row["number"])
        if issue is None:
            raise UnscorableResults(
                f"issue #{row['number']} is in {path.name} but not in the issue "
                f"cache; the cache was refreshed after that run, so its body and "
                f"labels can no longer be recovered"
            )
        scored.append(score_row(row, issue))
    return summarise_gate(scored)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--version",
        required=True,
        help="results file to re-score, e.g. v6 for eval/results/v6.json",
    )
    parser.add_argument(
        "--out",
        help="also write the full gate breakdown as JSON to this path",
    )
    args = parser.parse_args()

    gate = run_gate(args.version)
    print_gate_report(gate, args.version)

    if args.out:
        Path(args.out).write_text(json.dumps(gate, indent=2))
        print()
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
