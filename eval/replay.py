"""Replay closed issues through the same extract + policy path used live, and
score the predicted component label against what a maintainer actually
applied.

Only issues carrying exactly one label from policy.COMPONENTS are scored --
see scripts/check_labels.py and the README for why that's a minority of
closed issues (most are bot-closed or never triaged to a single area) and
what fraction that leaves us.

Run: python -m eval.replay --n 20 --version v1
"""

import argparse
import json
import os
from collections import Counter
from pathlib import Path

from app import github, policy
from app.extract import extract_facts

RESULTS_DIR = Path(__file__).parent / "results"


def _single_component_label(labels: list[str]) -> str | None:
    matches = [name for name in labels if name in policy.COMPONENTS]
    return matches[0] if len(matches) == 1 else None


def run_eval(repo: str, n: int, version: str) -> dict:
    issues = github.fetch_closed(
        repo, n=max(n * 4, n)
    )  # over-fetch; most won't qualify

    rows = []
    for issue in issues:
        actual = _single_component_label(issue["labels"])
        if actual is None:
            continue

        facts = extract_facts(issue["title"], issue["body"])
        if not facts:
            # Extraction failed closed; still record it as a miss rather than
            # silently dropping it, so failure rate is visible in the output.
            predicted = "needs-triage"
        else:
            predicted = policy.component(facts)

        rows.append(
            {
                "number": issue["number"],
                "url": issue["url"],
                "actual": actual,
                "predicted": predicted,
                "correct": predicted == actual,
                "extraction_failed": not bool(facts),
            }
        )
        if len(rows) >= n:
            break

    total = len(rows)
    correct = sum(1 for r in rows if r["correct"])
    accuracy = correct / total if total else 0.0

    confusions = Counter(
        (r["actual"], r["predicted"]) for r in rows if not r["correct"]
    )

    summary = {
        "version": version,
        "repo": repo,
        "requested_n": n,
        "scored_n": total,
        "accuracy": accuracy,
        "top_confusions": [
            {"want": want, "got": got, "count": count}
            for (want, got), count in confusions.most_common(10)
        ],
        "rows": rows,
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--n", type=int, default=20, help="number of qualifying issues to score"
    )
    parser.add_argument(
        "--version", type=str, required=True, help="tag for the results file, e.g. v1"
    )
    args = parser.parse_args()

    repo = os.environ.get("REPO", "vercel/next.js")
    summary = run_eval(repo, args.n, args.version)

    print(f"repo: {summary['repo']}")
    print(f"scored {summary['scored_n']} issues (requested {summary['requested_n']})")
    print(f"accuracy: {summary['accuracy']:.1%}")
    print()
    print("top confusions (want -> got):")
    for c in summary["top_confusions"]:
        print(f"  {c['want']!r} -> {c['got']!r}: {c['count']}")

    RESULTS_DIR.mkdir(exist_ok=True)
    out_path = RESULTS_DIR / f"{args.version}.json"
    out_path.write_text(json.dumps(summary, indent=2))
    print()
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
