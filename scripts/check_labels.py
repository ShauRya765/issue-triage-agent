"""One-off check run before writing any graph code (see README).

Prints the most common labels on closed vercel/next.js issues, then reports
how many closed issues carry exactly one "component" label -- the number the
eval in eval/replay.py depends on to be meaningful at all.

Run: python scripts/check_labels.py
"""

import os
from collections import Counter

import httpx

REPO = os.environ.get("REPO", "vercel/next.js")

# Labels that are applied by bots/moderation, not by a maintainer doing
# component triage. An issue whose only labels are these was never a
# candidate for a component label in the first place.
AUTOMATION_ONLY_LABELS = {"invalid link", "locked"}

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


def is_bot_closed(issue: dict) -> bool:
    names = {label["name"] for label in issue["labels"]}
    return bool(names) and names.issubset(AUTOMATION_ONLY_LABELS)


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
    triaged = [i for i in issues if not is_bot_closed(i)]
    sample = triaged[:100]
    exactly_one = sum(1 for i in sample if len(component_labels(i)) == 1)

    print(
        f"\n=== Single-component-label coverage (n={len(sample)}, bot-closed excluded) ==="
    )
    print(f"Exactly one component label: {exactly_one} / {len(sample)}")


if __name__ == "__main__":
    main()
