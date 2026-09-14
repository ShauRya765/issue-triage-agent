"""One-off check run before writing any graph code (see README).

Prints the most common labels on closed vercel/next.js issues, then reports
how many closed issues carry exactly one "component" label -- the number the
eval in eval/replay.py depends on to be meaningful at all.

Run: python scripts/check_labels.py
"""

import os
from collections import Counter

import httpx

# The bot-closed exclusion is defined once, in the eval that depends on it, and
# imported here so the vocabulary check and the eval can never disagree about
# which issues were never triaged in the first place.
from eval.replay import is_bot_closed

REPO = os.environ.get("REPO", "vercel/next.js")

# Additional generic labels to ignore when counting "component" labels.
GENERIC_LABELS = {
    "invalid link",
    "locked",
    "bug",
    "good first issue",
    "help wanted",
    "duplicate",
    "stale",
    "question",
}


def fetch_closed_issues(repo: str, pages: int = 8, per_page: int = 100) -> list[dict]:
    headers = {}
    if token := os.environ.get("GITHUB_TOKEN"):
        headers["Authorization"] = f"Bearer {token}"
    issues = []
    with httpx.Client(
        base_url="https://api.github.com", headers=headers, timeout=30
    ) as client:
        for page in range(1, pages + 1):
            resp = client.get(
                f"/repos/{repo}/issues",
                params={"state": "closed", "per_page": per_page, "page": page},
            )
            resp.raise_for_status()
            issues.extend(resp.json())
    # The issues endpoint also returns PRs; exclude them.
    return [i for i in issues if "pull_request" not in i]


def component_labels(issue: dict) -> list[str]:
    return [
        label["name"]
        for label in issue["labels"]
        if label["name"] not in GENERIC_LABELS
    ]


def main() -> None:
    issues = fetch_closed_issues(REPO)
    print(f"Fetched {len(issues)} non-PR closed issues from {REPO}\n")

    label_counts = Counter(
        label["name"] for issue in issues for label in issue["labels"]
    )
    print("=== Top 40 labels on closed issues ===")
    for name, count in label_counts.most_common(40):
        print(f"{count:4d}  {name}")

    # Bot-closed issues (only "invalid link"/"locked") were never triage
    # candidates -- exclude them before asking whether the label scheme is
    # dense enough to eval against.
    triaged = [
        i for i in issues if not is_bot_closed([la["name"] for la in i["labels"]])
    ]
    sample = triaged[:100]
    exactly_one = sum(1 for i in sample if len(component_labels(i)) == 1)

    print(
        f"\n=== Single-component-label coverage (n={len(sample)}, bot-closed excluded) ==="
    )
    print(f"Exactly one component label: {exactly_one} / {len(sample)}")


if __name__ == "__main__":
    main()
