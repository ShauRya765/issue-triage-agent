"""Thin httpx client for the GitHub REST API.

All writes (add_label, comment) respect SHADOW_MODE: when set (the default),
they perform no network call and return a string describing what would have
happened instead. Reads always hit the real API -- there is no shadow mode
for reads, since reading a public repo's issues has no side effects.
"""

import os

import httpx

GITHUB_API = "https://api.github.com"


def _shadow_mode() -> bool:
    return os.environ.get("SHADOW_MODE", "true").lower() != "false"


def _headers() -> dict[str, str]:
    headers = {"Accept": "application/vnd.github+json"}
    if token := os.environ.get("GITHUB_TOKEN"):
        headers["Authorization"] = f"Bearer {token}"
    return headers


def _client() -> httpx.Client:
    return httpx.Client(base_url=GITHUB_API, headers=_headers(), timeout=30)


def _issue_to_dict(repo: str, raw: dict) -> dict:
    return {
        "repo": repo,
        "number": raw["number"],
        "title": raw["title"],
        "body": raw.get("body") or "",
        "labels": [label["name"] for label in raw["labels"]],
        "url": raw["html_url"],
        # user is null for deleted accounts; normalise to "" so callers can
        # treat "nobody to look up" as a plain falsy check.
        "reporter": (raw.get("user") or {}).get("login") or "",
        "author_association": raw.get("author_association") or "NONE",
    }


def fetch_open(repo: str, n: int = 20) -> list[dict]:
    """Fetch the n most recently updated open issues (PRs excluded)."""
    issues: list[dict] = []
    with _client() as client:
        page = 1
        while len(issues) < n:
            resp = client.get(
                f"/repos/{repo}/issues",
                params={"state": "open", "per_page": 100, "page": page},
            )
            resp.raise_for_status()
            batch = resp.json()
            if not batch:
                break
            issues.extend(raw for raw in batch if "pull_request" not in raw)
            page += 1
    return [_issue_to_dict(repo, raw) for raw in issues[:n]]


def fetch_closed(repo: str, n: int = 20) -> list[dict]:
    """Fetch the n most recently updated closed issues (PRs excluded), with labels."""
    issues: list[dict] = []
    with _client() as client:
        page = 1
        while len(issues) < n:
            resp = client.get(
                f"/repos/{repo}/issues",
                params={"state": "closed", "per_page": 100, "page": page},
            )
            resp.raise_for_status()
            batch = resp.json()
            if not batch:
                break
            issues.extend(raw for raw in batch if "pull_request" not in raw)
            page += 1
    return [_issue_to_dict(repo, raw) for raw in issues[:n]]


def fetch_issue(repo: str, number: int) -> dict:
    with _client() as client:
        resp = client.get(f"/repos/{repo}/issues/{number}")
        resp.raise_for_status()
    return _issue_to_dict(repo, resp.json())


def search_issue_count(repo: str, author: str) -> int | None:
    """Count issues this author has filed on this repo. None if the lookup failed.

    Returns None rather than raising or defaulting to 0: the caller has to be
    able to tell "this person has filed nothing" apart from "we could not find
    out", because those two warrant different triage behaviour.

    Note the search API is rate limited far harder than the REST API (10
    requests/minute unauthenticated, 30 authenticated), which is the usual
    reason this returns None. Setting GITHUB_TOKEN is worth it here.
    """
    try:
        with _client() as client:
            resp = client.get(
                "/search/issues",
                params={"q": f"repo:{repo} author:{author} type:issue", "per_page": 1},
            )
            resp.raise_for_status()
            return int(resp.json().get("total_count", 0))
    except (httpx.HTTPError, ValueError, KeyError):
        return None


def add_label(repo: str, number: int, label: str) -> str:
    if _shadow_mode():
        return f"[shadow] would add label {label!r} to {repo}#{number}"
    with _client() as client:
        resp = client.post(
            f"/repos/{repo}/issues/{number}/labels",
            json={"labels": [label]},
        )
        resp.raise_for_status()
    return f"added label {label!r} to {repo}#{number}"


def comment(repo: str, number: int, body: str) -> str:
    if _shadow_mode():
        return f"[shadow] would comment on {repo}#{number}: {body!r}"
    with _client() as client:
        resp = client.post(
            f"/repos/{repo}/issues/{number}/comments",
            json={"body": body},
        )
        resp.raise_for_status()
    return f"commented on {repo}#{number}"
