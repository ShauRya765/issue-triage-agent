"""Unit tests for app.context.

Offline: the tier mapping is a pure function over GitHub's author_association
string, and the failure paths are exercised by stubbing the one network call.
"""

from unittest.mock import patch

from app import context


def make_issue(**overrides) -> dict:
    issue = {
        "repo": "vercel/next.js",
        "number": 1,
        "title": "t",
        "body": "b",
        "labels": [],
        "url": "u",
        "reporter": "octocat",
        "author_association": "NONE",
    }
    issue.update(overrides)
    return issue


def test_tier_maps_maintainers_to_internal():
    for association in ("OWNER", "MEMBER", "COLLABORATOR"):
        assert context._tier(association) == "internal"


def test_tier_maps_contributor():
    assert context._tier("CONTRIBUTOR") == "contributor"


def test_tier_defaults_unrecognised_to_external():
    """GitHub has added association values before; unknown must not escalate."""
    assert context._tier("FIRST_TIME_CONTRIBUTOR") == "external"
    assert context._tier("") == "external"
    assert context._tier(None) == "external"


def test_missing_reporter_is_unknown():
    """Deleted accounts leave no one to look up."""
    result = context.fetch_reporter_context(make_issue(reporter=""))
    assert result["unknown"] is True


def test_failed_lookup_keeps_tier_but_marks_unknown():
    """A rate-limited search still leaves the association, which came free."""
    with patch("app.context.search_issue_count", return_value=None):
        result = context.fetch_reporter_context(make_issue(author_association="MEMBER"))
    assert result["unknown"] is True
    assert result["tier"] == "internal"


def test_successful_lookup_excludes_the_current_issue():
    """The issue being triaged is in its own author's search results."""
    with patch("app.context.search_issue_count", return_value=4):
        result = context.fetch_reporter_context(make_issue())
    assert result["prior_issues"] == 3
    assert result["unknown"] is False


def test_first_ever_issue_reports_zero_prior():
    with patch("app.context.search_issue_count", return_value=1):
        result = context.fetch_reporter_context(make_issue())
    assert result["prior_issues"] == 0
