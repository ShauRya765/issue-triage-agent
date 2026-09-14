"""Makes the offline claim enforceable rather than a promise.

The README says triage decisions are made by app.policy, not by a model, and
that the tests prove it. That only means something if a test *can't* quietly
reach the network -- otherwise a future stubbed-out patch could rot into a real
API call and the suite would still be green (just slower, and only on a machine
that happened to have a key).

So every test runs with socket creation blocked. A test that tries to talk to
api.anthropic.com, api.github.com or Postgres fails with NetworkUsedInTest
naming the culprit, instead of hanging or silently succeeding in CI.
"""

import socket

import pytest


class NetworkUsedInTest(RuntimeError):
    """A test attempted a network connection. The suite must run offline."""


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def blocked(*args, **kwargs):
        raise NetworkUsedInTest(
            "This test tried to open a socket. The suite must run offline -- "
            "stub the call (see patches in tests/test_graph.py) instead of "
            "reaching api.anthropic.com, api.github.com or Postgres."
        )

    # socket.socket covers httpx/psycopg; create_connection and the getaddrinfo
    # path cover the helpers that don't go through the class directly.
    monkeypatch.setattr(socket, "socket", blocked)
    monkeypatch.setattr(socket, "create_connection", blocked)
    monkeypatch.setattr(socket, "getaddrinfo", blocked)
