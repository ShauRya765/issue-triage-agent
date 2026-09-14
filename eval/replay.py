"""Replay closed issues through the same extract + policy path used live, and
score the predicted component label against what a maintainer actually
applied.

Two filters decide what is scorable, and both matter for reading the number:

1. Bot-closed issues are dropped -- an issue whose only labels are
   `invalid link` / `locked` was closed by automation and never triaged, so it
   has no maintainer judgment to score against. This is the same exclusion
   scripts/check_labels.py applies, imported from here so the two can't drift.
2. Of what remains, only issues carrying exactly one label from
   policy.COMPONENTS are scored. Zero labels means nobody routed it; two or
   more means "the right answer" isn't a single value.

That leaves a minority of closed issues, so scoring ~150 means fetching several
hundred. Fetched issues are cached under eval/cache/ so re-running to compare
prompt versions doesn't re-hit the GitHub API -- and so every version is scored
against the same sample rather than a fresh one that has drifted.

Run: python -m eval.replay --n 150 --version v4
"""

import argparse
import json
import math
import os
import sys
import statistics
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from app import github, policy, usage
from app.extract import extract_facts

RESULTS_DIR = Path(__file__).parent / "results"
CACHE_DIR = Path(__file__).parent / "cache"

ABSTAIN = "needs-triage"

# Labels applied by bots/moderation rather than by a maintainer doing component
# triage. An issue whose only labels are these was never a triage candidate, so
# counting it as a scoring miss would measure the bot, not the extractor.
# scripts/check_labels.py imports these two names -- keep them in step.
AUTOMATION_ONLY_LABELS = {"invalid link", "locked"}


def is_bot_closed(label_names: list[str]) -> bool:
    """True if every label on the issue is automation-applied (and there is one).

    An issue with no labels at all is not bot-closed -- it's untriaged, which is
    a different thing, and it gets dropped later for lacking a component label
    anyway.
    """
    names = set(label_names)
    return bool(names) and names.issubset(AUTOMATION_ONLY_LABELS)


def _strip_prefix(label: str) -> str:
    """Compare against the repo's own label names when LABEL_PREFIX is set.

    policy.component applies LABEL_PREFIX to what it returns, so with a prefix
    configured every prediction would differ from the maintainer's label by that
    prefix and accuracy would read as 0%.
    """
    prefix = policy.LABEL_PREFIX
    if prefix and label.startswith(prefix):
        return label[len(prefix) :]
    return label


def _single_component_label(labels: list[str]) -> str | None:
    matches = [name for name in labels if name in policy.COMPONENTS]
    return matches[0] if len(matches) == 1 else None


# --- fetching and caching ---------------------------------------------


def _cache_path(repo: str) -> Path:
    return CACHE_DIR / f"{repo.replace('/', '__')}-closed.json"


def load_or_fetch(
    repo: str, fetch_n: int, refresh: bool = False
) -> tuple[list[dict], bool]:
    """Return (issues, from_cache). Re-fetches only when the cache can't serve.

    A cache holding at least as many issues as requested is used as-is, sliced
    to fetch_n. A smaller cache is discarded rather than topped up: the list is
    ordered by update time, so a later page fetched now wouldn't splice cleanly
    onto pages fetched days ago.
    """
    path = _cache_path(repo)
    if not refresh and path.exists():
        try:
            cached = json.loads(path.read_text())
            issues = cached["issues"]
        except (json.JSONDecodeError, KeyError, OSError) as exc:
            print(f"cache at {path} unreadable ({exc}); re-fetching", file=sys.stderr)
        else:
            if len(issues) >= fetch_n:
                print(
                    f"using {fetch_n} of {len(issues)} cached issues "
                    f"(fetched {cached.get('fetched_at', 'unknown')}); "
                    f"--refresh to re-fetch"
                )
                return issues[:fetch_n], True
            print(
                f"cache has {len(issues)} issues, need {fetch_n}; re-fetching",
                file=sys.stderr,
            )

    print(f"fetching up to {fetch_n} closed issues from {repo}...")
    issues = github.fetch_closed(repo, n=fetch_n)
    CACHE_DIR.mkdir(exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "repo": repo,
                "fetched_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "count": len(issues),
                "issues": issues,
            },
            indent=2,
        )
    )
    print(f"fetched {len(issues)} issues, cached to {path}")
    return issues, False


def select_scorable(
    issues: list[dict], limit: int
) -> tuple[list[tuple[dict, str]], dict]:
    """Pick issues that can be scored, and report how the rest were filtered."""
    funnel = {"fetched": len(issues), "bot_closed": 0, "no_single_component": 0}
    selected: list[tuple[dict, str]] = []

    for issue in issues:
        if is_bot_closed(issue["labels"]):
            funnel["bot_closed"] += 1
            continue
        actual = _single_component_label(issue["labels"])
        if actual is None:
            funnel["no_single_component"] += 1
            continue
        if len(selected) < limit:
            selected.append((issue, actual))

    funnel["eligible"] = (
        funnel["fetched"] - funnel["bot_closed"] - funnel["no_single_component"]
    )
    funnel["scored"] = len(selected)
    return selected, funnel


# --- scoring ----------------------------------------------------------


def score_issues(selected: list[tuple[dict, str]], concurrency: int) -> list[dict]:
    """Extract facts for each issue in parallel, then decide with policy.

    Only extraction is parallel -- policy.component is pure and instant. Rows
    come back in the input order regardless of completion order, so a rerun with
    a different --concurrency produces a byte-identical row list given the same
    model output.
    """
    total = len(selected)
    done = 0
    lock = threading.Lock()

    def work(item: tuple[dict, str]) -> dict:
        nonlocal done
        issue, actual = item

        # The collector is opened here, inside the worker, because a ContextVar
        # binding does not cross into a new thread -- one opened around the pool
        # would capture nothing. See app.usage.
        started = time.perf_counter()
        with usage.collector() as calls, usage.node("extract"):
            facts = extract_facts(issue["title"], issue["body"])
        wall_ms = (time.perf_counter() - started) * 1000

        if not facts:
            # Extraction failed closed; record it as a miss rather than
            # silently dropping it, so failure rate stays visible in output.
            predicted = ABSTAIN
        else:
            predicted = _strip_prefix(policy.component(facts))

        with lock:
            done += 1
            print(f"\r  scored {done}/{total}", end="", file=sys.stderr, flush=True)

        input_tokens = sum(c["input_tokens"] for c in calls)
        output_tokens = sum(c["output_tokens"] for c in calls)
        return {
            "number": issue["number"],
            "url": issue["url"],
            "actual": actual,
            "predicted": predicted,
            "correct": predicted == actual,
            "extraction_failed": not bool(facts),
            # Telemetry. wall_ms covers extraction only -- the eval calls
            # extract_facts directly and never fetches reporter context, so this
            # is not the full agent latency. Retries inflate it, which is the
            # point: a retried 429 is latency the caller really waited.
            "wall_ms": round(wall_ms, 1),
            "llm_calls": len(calls),
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cost_usd": round(usage.cost_usd(input_tokens, output_tokens), 6),
        }

    print(f"extracting {total} issues with concurrency={concurrency}...")
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        rows = list(pool.map(work, selected))
    print(file=sys.stderr)
    return rows


def per_component_metrics(rows: list[dict]) -> list[dict]:
    """Precision and recall per component label.

    Precision answers "when it applied this label, how often was that right" --
    the number that matters for trusting an autonomous run. Recall answers "of
    the issues that belonged here, how many did it find" -- the number that
    matters for how much triage work is actually absorbed. Abstentions cost
    recall but never precision, which is exactly the asymmetry the autonomy gate
    is built around.
    """
    labels = {r["actual"] for r in rows} | {
        r["predicted"] for r in rows if r["predicted"] != ABSTAIN
    }
    metrics = []
    for label in sorted(labels):
        tp = sum(1 for r in rows if r["predicted"] == label and r["actual"] == label)
        fp = sum(1 for r in rows if r["predicted"] == label and r["actual"] != label)
        fn = sum(1 for r in rows if r["actual"] == label and r["predicted"] != label)
        metrics.append(
            {
                "component": label,
                "support": tp + fn,
                "predicted_n": tp + fp,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "precision": (tp / (tp + fp)) if (tp + fp) else None,
                "recall": (tp / (tp + fn)) if (tp + fn) else None,
            }
        )
    return sorted(metrics, key=lambda m: (-m["support"], m["component"]))


def baselines(rows: list[dict]) -> dict:
    """What the same scored set yields without reading the issue at all.

    Two reference points, because 74% means nothing on its own -- it depends
    entirely on how many components there are and how evenly the issues spread
    across them:

    - majority class: always guess the most common component. This is the bar a
      classifier has to clear to be doing anything, and on a repo with one
      dominant area it can be deceptively high.
    - random, weighted by the observed distribution: guess component L with
      probability equal to L's frequency, giving expected accuracy sum(p^2).
      Weighted rather than uniform (1/22 here) because uniform is a strawman --
      it assumes a guesser ignorant of the obvious fact that Turbopack issues
      are common.

    Both are computed over the same rows as the headline number, so they move
    with the sample instead of being quoted from a different one.
    """
    total = len(rows)
    if not total:
        return {
            "n": 0,
            "distinct_components": 0,
            "majority_class": None,
            "majority_accuracy": 0.0,
            "random_weighted_accuracy": 0.0,
        }

    distribution = Counter(r["actual"] for r in rows)
    # Sort so a tie between two equally common components resolves the same way
    # on every run rather than following dict insertion order.
    majority, majority_support = max(
        distribution.items(), key=lambda kv: (kv[1], kv[0])
    )
    return {
        "n": total,
        "distinct_components": len(distribution),
        "majority_class": majority,
        "majority_support": majority_support,
        "majority_accuracy": majority_support / total,
        "random_weighted_accuracy": sum(
            (count / total) ** 2 for count in distribution.values()
        ),
    }


def _percentile(values: list[float], q: float) -> float:
    """Nearest-rank percentile, so p95 is always an observed value.

    No interpolation and no numpy: at n=150 the interpolated and nearest-rank
    p95 differ by less than the run-to-run noise, and reporting a latency that
    no request actually took invites questions the number can't answer.
    """
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = math.ceil(q * len(ordered))
    return ordered[min(max(rank, 1), len(ordered)) - 1]


def cost_and_latency(rows: list[dict]) -> dict:
    """Per-run latency distribution and what the whole eval cost to produce.

    Median and p95 rather than a mean: the distribution has a tail (retries,
    long issue bodies) and a mean hides it. p95 is the number to quote for a
    latency budget; the median is what a typical issue feels like.
    """
    latencies = [r["wall_ms"] for r in rows if "wall_ms" in r]
    input_tokens = sum(r.get("input_tokens", 0) for r in rows)
    output_tokens = sum(r.get("output_tokens", 0) for r in rows)
    total_cost = sum(r.get("cost_usd", 0.0) for r in rows)
    input_rate, output_rate = usage.rates()
    n = len(rows)

    return {
        "cost": {
            "total_usd": round(total_cost, 6),
            "per_issue_usd": round(total_cost / n, 6) if n else 0.0,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
            "input_tokens_per_issue": round(input_tokens / n, 1) if n else 0.0,
            "output_tokens_per_issue": round(output_tokens / n, 1) if n else 0.0,
            "rates_usd_per_mtok": {"input": input_rate, "output": output_rate},
        },
        "latency_ms": {
            "median": round(statistics.median(latencies), 1) if latencies else 0.0,
            "p95": round(_percentile(latencies, 0.95), 1),
            "min": round(min(latencies), 1) if latencies else 0.0,
            "max": round(max(latencies), 1) if latencies else 0.0,
            "measures": "extraction only; excludes the reporter-context fetch",
        },
    }


def wilson_interval(correct: int, total: int, z: float = 1.96) -> tuple[float, float]:
    """95% confidence interval for the accuracy figure (Wilson score).

    The reason this is printed next to accuracy: at n=15 the interval on 66.7%
    ran from roughly 42% to 85%, which is wide enough that the headline number
    was not evidence of much. Wilson rather than the normal approximation
    because it stays sane near 0% and 100%, which the per-component slices with
    single-digit support routinely are.
    """
    if not total:
        return (0.0, 0.0)
    p = correct / total
    denom = 1 + z**2 / total
    centre = (p + z**2 / (2 * total)) / denom
    spread = z * math.sqrt(p * (1 - p) / total + z**2 / (4 * total**2)) / denom
    return (max(0.0, centre - spread), min(1.0, centre + spread))


def summarise(
    rows: list[dict], repo: str, version: str, funnel: dict, requested_n: int
) -> dict:
    total = len(rows)
    correct = sum(1 for r in rows if r["correct"])
    abstained = sum(1 for r in rows if r["predicted"] == ABSTAIN)
    # Every scored issue has a real component label as ground truth, so an
    # abstention is always wrong: correct + abstained + mislabelled == total.
    mislabelled = sum(1 for r in rows if not r["correct"] and r["predicted"] != ABSTAIN)
    assert correct + abstained + mislabelled == total, "outcome buckets must partition"

    confusions = Counter(
        (r["actual"], r["predicted"]) for r in rows if not r["correct"]
    )

    def rate(count: int) -> float:
        return count / total if total else 0.0

    return {
        "version": version,
        "repo": repo,
        "requested_n": requested_n,
        "scored_n": total,
        "accuracy": rate(correct),
        "accuracy_ci95": list(wilson_interval(correct, total)),
        "abstention_rate": rate(abstained),
        "mislabel_rate": rate(mislabelled),
        "extraction_failures": sum(1 for r in rows if r["extraction_failed"]),
        "counts": {
            "correct": correct,
            "abstained": abstained,
            "mislabelled": mislabelled,
        },
        "funnel": funnel,
        "baselines": baselines(rows),
        **cost_and_latency(rows),
        "per_component": per_component_metrics(rows),
        "top_confusions": [
            {"want": want, "got": got, "count": count}
            for (want, got), count in confusions.most_common(10)
        ],
        "rows": rows,
    }


# --- output -----------------------------------------------------------


def _pct(value: float | None) -> str:
    return "     -" if value is None else f"{value:6.1%}"


def print_report(summary: dict) -> None:
    funnel = summary["funnel"]
    print()
    print(f"repo:    {summary['repo']}")
    print(f"version: {summary['version']}")
    print()
    print("sample funnel:")
    print(f"  fetched closed issues        {funnel['fetched']:5d}")
    print(f"  - bot-closed (invalid link/locked) {funnel['bot_closed']:5d}")
    print(f"  - no single component label  {funnel['no_single_component']:5d}")
    print(f"  = eligible to score          {funnel['eligible']:5d}")
    print(f"  scored                       {funnel['scored']:5d}")
    print()

    counts = summary["counts"]
    n = summary["scored_n"]
    print(f"scored n = {n}")
    lo, hi = summary["accuracy_ci95"]
    print(
        f"  accuracy        {summary['accuracy']:6.1%}  ({counts['correct']}/{n})"
        f"   95% CI [{lo:.1%}, {hi:.1%}]"
    )
    print(
        f"  abstention rate {summary['abstention_rate']:6.1%}  "
        f"({counts['abstained']}/{n} predicted {ABSTAIN})"
    )
    print(
        f"  mislabel rate   {summary['mislabel_rate']:6.1%}  "
        f"({counts['mislabelled']}/{n} wrong label applied)"
    )
    if summary["extraction_failures"]:
        print(f"  extraction failures {summary['extraction_failures']}")
    if n:
        print(f"  one issue moves accuracy by {1 / n:.1%}")
    print()

    base = summary["baselines"]
    print(
        f"baselines on the same {base['n']} issues "
        f"({base['distinct_components']} components present):"
    )
    # Built as (label, value) so the variable-length majority label can't knock
    # the value column out of alignment.
    lines = [
        (
            f"majority class (always {base['majority_class']!r})",
            base["majority_accuracy"],
        ),
        ("random, weighted by distribution", base["random_weighted_accuracy"]),
        ("agent", summary["accuracy"]),
    ]
    width = max(len(label) for label, _ in lines)
    for label, value in lines:
        print(f"  {label:<{width}}  {value:6.1%}")
    if base["majority_accuracy"]:
        lift = summary["accuracy"] / base["majority_accuracy"]
        if lift < 1.1:
            print("  -> no better than always guessing the most common component")
        else:
            print(f"  -> {lift:.1f}x the majority-class baseline")
    print()

    cost, latency = summary["cost"], summary["latency_ms"]
    print("cost and latency:")
    print(
        f"  total cost            ${cost['total_usd']:.4f}"
        f"  over {n} issues at "
        f"${cost['rates_usd_per_mtok']['input']:.2f}/"
        f"${cost['rates_usd_per_mtok']['output']:.2f} per Mtok in/out"
    )
    print(f"  cost per issue        ${cost['per_issue_usd']:.6f}")
    print(
        f"  tokens per issue      {cost['input_tokens_per_issue']:,.0f} in / "
        f"{cost['output_tokens_per_issue']:,.0f} out"
    )
    print(
        f"  latency per issue     median {latency['median'] / 1000:.2f}s"
        f"   p95 {latency['p95'] / 1000:.2f}s"
        f"   max {latency['max'] / 1000:.2f}s"
    )
    print(f"                        ({latency['measures']})")
    print()

    print("per-component (support = issues actually labelled this):")
    print(f"  {'component':<32} {'sup':>4} {'pred':>5} {'prec':>7} {'recall':>7}")
    for m in summary["per_component"]:
        print(
            f"  {m['component']:<32} {m['support']:>4} {m['predicted_n']:>5} "
            f"{_pct(m['precision'])} {_pct(m['recall'])}"
        )
    print()

    print("top confusions (want -> got):")
    for c in summary["top_confusions"]:
        print(f"  {c['want']!r} -> {c['got']!r}: {c['count']}")


def run_eval(
    repo: str,
    n: int,
    version: str,
    fetch_n: int,
    concurrency: int,
    refresh: bool,
) -> dict:
    issues, _ = load_or_fetch(repo, fetch_n, refresh=refresh)
    selected, funnel = select_scorable(issues, n)

    if not selected:
        raise SystemExit("no scorable issues found; try a larger --fetch")
    if len(selected) < n:
        print(
            f"warning: only {len(selected)} scorable issues in {len(issues)} fetched "
            f"(wanted {n}); raise --fetch for a larger sample",
            file=sys.stderr,
        )

    rows = score_issues(selected, concurrency)
    return summarise(rows, repo, version, funnel, requested_n=n)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--n", type=int, default=150, help="number of qualifying issues to score"
    )
    parser.add_argument(
        "--version", type=str, required=True, help="tag for the results file, e.g. v4"
    )
    parser.add_argument(
        "--fetch",
        type=int,
        default=800,
        help="closed issues to fetch before filtering (default 800; roughly "
        "1 in 5 survives both filters)",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=8,
        help="parallel extraction calls (default 8; the extractor backs off on "
        "429s, so raising this trades throughput for retries)",
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="ignore the cached issues and re-fetch from GitHub",
    )
    args = parser.parse_args()

    if args.concurrency < 1:
        raise SystemExit("--concurrency must be at least 1")

    repo = os.environ.get("REPO", "vercel/next.js")
    summary = run_eval(
        repo,
        args.n,
        args.version,
        fetch_n=max(args.fetch, args.n),
        concurrency=args.concurrency,
        refresh=args.refresh,
    )

    print_report(summary)

    RESULTS_DIR.mkdir(exist_ok=True)
    out_path = RESULTS_DIR / f"{args.version}.json"
    out_path.write_text(json.dumps(summary, indent=2))
    print()
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
