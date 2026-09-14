"""Thin httpx client for the GitHub REST API.

All writes (add_label, comment) respect SHADOW_MODE: when set (the default),
they perform no network call and return a string describing what would have
happened instead. Reads always hit the real API -- there is no shadow mode
for reads, since reading a public repo's issues has no side effects.
"""

import logging
import os
import random
import time

import httpx

GITHUB_API = "https://api.github.com"

logger = logging.getLogger(__name__)

# The REST API allows 5000 requests/hour authenticated but only 60
# unauthenticated, and it answers an exhausted budget with 403 (not 429) plus a
# reset timestamp. Both shapes are handled in _get: a large eval fetch is the
# one caller that can realistically hit either.
_MAX_RETRIES = 5
_MAX_SLEEP_SECONDS = 60.0


def _retry_after_seconds(resp: httpx.Response) -> float | None:
    """How long the API is asking us to wait, or None if it isn't asking."""
    if (retry_after := resp.headers.get("Retry-After")) is not None:
        try:
            return max(0.0, float(retry_after))
        except ValueError:
            return None
    # Secondary/primary rate limit: remaining hits 0 and reset is absolute epoch
    # seconds. Only meaningful on a 403/429, checked by the caller.
    if resp.headers.get("X-RateLimit-Remaining") == "0":
        try:
            reset = float(resp.headers["X-RateLimit-Reset"])
        except (KeyError, ValueError):
            return None
        return max(0.0, reset - time.time())
    return None


def _get(client: httpx.Client, path: str, params: dict | None = None) -> httpx.Response:
    """GET with backoff on rate limits and transient server errors.

    Retries 429/403-with-reset (rate limited) and 5xx. Anything else raises
    immediately -- a 404 is not worth waiting out.
    """
    for attempt in range(_MAX_RETRIES):
        resp = client.get(path, params=params)

        if resp.status_code in (403, 429):
            wait = _retry_after_seconds(resp)
            if wait is not None and attempt < _MAX_RETRIES - 1:
                # A primary-limit reset can be up to an hour out; don't silently
                # block the process for that long, surface it instead.
                if wait > _MAX_SLEEP_SECONDS:
                    resp.raise_for_status()
                logger.warning(
                    "github: rate limited on %s, sleeping %.1fs (attempt %d/%d)",
                    path,
                    wait,
                    attempt + 1,
                    _MAX_RETRIES,
                )
                time.sleep(wait + random.uniform(0, 1))
                continue

        if resp.status_code >= 500 and attempt < _MAX_RETRIES - 1:
            backoff = min(2.0**attempt, _MAX_SLEEP_SECONDS) + random.uniform(0, 1)
            logger.warning(
                "github: %d on %s, retrying in %.1fs", resp.status_code, path, backoff
            )
            time.sleep(backoff)
            continue

        resp.raise_for_status()
        return resp

    resp.raise_for_status()
    return resp


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


PER_PAGE = 100


def _fetch_issues(repo: str, state: str, n: int) -> list[dict]:
    """Page through the issues list until n non-PR issues are collected.

    Deduplicates by issue number. The default sort is `updated` descending, so
    on an active repo an issue touched mid-fetch can shift between pages and be
    returned twice (or skipped); the eval reads several hundred issues across
    many pages, which makes that likely rather than theoretical. Dedupe keeps
    the sample honest -- without it the same issue could be scored twice and
    quietly weight the accuracy figure.
    """
    seen: set[int] = set()
    issues: list[dict] = []
    with _client() as client:
        page = 1
        while len(issues) < n:
            resp = _get(
                client,
                f"/repos/{repo}/issues",
                params={"state": state, "per_page": PER_PAGE, "page": page},
            )
            batch = resp.json()
            if not batch:
                break  # ran out of issues before reaching n
            for raw in batch:
                if "pull_request" in raw or raw["number"] in seen:
                    continue
                seen.add(raw["number"])
                issues.append(_issue_to_dict(repo, raw))
            page += 1
    return issues[:n]


def fetch_open(repo: str, n: int = 20) -> list[dict]:
    """Fetch the n most recently updated open issues (PRs excluded)."""
    return _fetch_issues(repo, "open", n)


def fetch_closed(repo: str, n: int = 20) -> list[dict]:
    """Fetch the n most recently updated closed issues (PRs excluded), with labels."""
    return _fetch_issues(repo, "closed", n)


def fetch_issue(repo: str, number: int) -> dict:
    with _client() as client:
        resp = _get(client, f"/repos/{repo}/issues/{number}")
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
            resp = _get(
                client,
                "/search/issues",
                params={"q": f"repo:{repo} author:{author} type:issue", "per_page": 1},
            )
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
