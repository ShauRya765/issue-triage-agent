"""Fetches context about the person who filed an issue.

This is the "check customer context" step. It is a lookup, not a judgment:
every field here comes from the GitHub API, and nothing in this module calls
a model or decides anything. app.policy reads the result to weigh urgency,
exactly as a human triager would glance at who filed a ticket before deciding
how fast to move on it.

Why GitHub's author_association and prior issue count, specifically: on a
commercial desk this step would hit a CRM for plan tier, contract value and
open ticket count. This project triages a public OSS repo, where no such
record exists, so it uses the closest real equivalents the platform actually
exposes. The shape is what matters and it is the same either way -- a fetched
record about the requester, kept separate from the request text, feeding a
rule rather than a prompt. Swapping in a CRM call means rewriting this module
only; Facts, policy's signature and the graph stay as they are.

Every failure degrades to unknown=True rather than raising. A rate-limited
lookup must not take down a triage run, and policy treats unknown as "no
signal" -- never as "external", which would silently deprioritise a
maintainer's report the moment the API got slow.
"""

from app.github import search_issue_count
from app.state import Issue, ReporterContext

# GitHub's author_association values that mean the reporter is inside the
# project. These people can already label and close issues themselves, so a
# report from one of them has effectively been pre-triaged.
_INTERNAL = frozenset({"OWNER", "MEMBER", "COLLABORATOR"})

# Has landed a merged PR before. Knows the codebase; their reports tend to
# arrive with real reproductions.
_CONTRIBUTOR = frozenset({"CONTRIBUTOR"})


def _tier(author_association: str) -> str:
    """Normalise GitHub's association into the three bands policy cares about.

    Unrecognised values (GitHub has added values here before, e.g.
    FIRST_TIME_CONTRIBUTOR) fall through to "external", which is the
    conservative direction: it grants no escalation.
    """
    association = (author_association or "").upper()
    if association in _INTERNAL:
        return "internal"
    if association in _CONTRIBUTOR:
        return "contributor"
    return "external"


def unknown_context(login: str = "") -> ReporterContext:
    """The no-signal context. Used when there's nobody to look up, or on failure."""
    return ReporterContext(
        login=login,
        tier="external",
        prior_issues=0,
        unknown=True,
    )


def fetch_reporter_context(issue: Issue) -> ReporterContext:
    """Look up what we know about this issue's reporter. Never raises."""
    login = issue.get("reporter") or ""
    if not login:
        # Deleted account, or a payload without a user. Nothing to look up.
        return unknown_context()

    tier = _tier(issue.get("author_association", ""))

    # One extra request. If it fails we still return the tier, which came free
    # on the issue payload -- a partial context beats no context.
    prior = search_issue_count(issue["repo"], login)
    if prior is None:
        return ReporterContext(
            login=login,
            tier=tier,
            prior_issues=0,
            unknown=True,
        )

    # The issue being triaged is itself in the search results; don't count it.
    return ReporterContext(
        login=login,
        tier=tier,
        prior_issues=max(0, prior - 1),
        unknown=False,
    )
