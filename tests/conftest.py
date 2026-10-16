"""Test fakes. FakeTicketing mimics the mock API's shapes (mock-ticketing/app.py) in memory."""

import json
import os
import re
from collections.abc import Callable
from typing import Any

import httpx
import pytest
from pydantic import BaseModel, ConfigDict, Field

from ticket_agent.config import Settings
from ticket_agent.ticketing_api import TicketingClient

API_KEY = "test-key-not-real"

# Tests must not depend on the committed agent.toml: Settings in tests use the defaults
# plus what each test passes.
os.environ["AGENT_CONFIG"] = ""


@pytest.fixture(autouse=True)
def _no_settings_from_the_shell(monkeypatch: pytest.MonkeyPatch) -> None:
    """Nor on the shell: Settings reads the environment, so an exported GEMINI_API_KEY or
    LLM_MODEL would leak in. Each test starts without them and sets what it needs."""
    fields = set(Settings.model_fields)
    for name in list(os.environ):
        if name.lower().partition("__")[0] in fields:  # LLM_MODEL, TICKETING__URL, ...
            monkeypatch.delenv(name)


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


class Fail(BaseModel):
    """One scripted failure for the next request on a route.

    commit=False fails before the handler runs, like FLAKY in app.py:76-82.
    commit=True applies the write first and then fails: the case a real API can produce.
    """

    status: int | None = None
    retry_after: str | None = None
    exc: type[httpx.TransportError] | None = None
    commit: bool = False


class FakeTicketing(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    tickets: dict[str, dict[str, Any]] = Field(default_factory=dict)
    script: dict[str, list[Fail]] = Field(default_factory=dict)
    requests: list[tuple[str, str]] = Field(default_factory=list)
    always_has_more: bool = False
    # Runs after the request is recorded. Return a response to short-circuit, or None.
    intercept: Callable[[httpx.Request], httpx.Response | None] | None = None
    next_ticket: int = 2000
    next_comment: int = 0

    def add(self, **fields: Any) -> dict[str, Any]:
        self.next_ticket += 1
        ticket = {
            "id": f"TCK-{self.next_ticket}",
            "title": "Seeded ticket",
            "description": "",
            "status": "open",
            "priority": "medium",
            "category": "general",
            "requester": "Someone Else",
            "assignee": None,
            "created_at": "2026-09-01T09:00:00Z",
            "updated_at": "2026-09-01T09:00:00Z",
            "comments": [],
        }
        ticket.update(fields)
        self.tickets[ticket["id"]] = ticket
        return ticket

    def fail(self, route: str, *failures: Fail) -> None:
        self.script.setdefault(route, []).extend(failures)

    def count(self, method: str, path_prefix: str) -> int:
        return sum(1 for m, p in self.requests if m == method and p.startswith(path_prefix))

    def transport(self) -> httpx.MockTransport:
        return httpx.MockTransport(self.handle)

    def handle(self, request: httpx.Request) -> httpx.Response:
        self.requests.append((request.method, request.url.path))
        if self.intercept and (response := self.intercept(request)) is not None:
            return response
        if request.headers.get("x-api-key") != API_KEY:
            return httpx.Response(401, json={"error": "unauthorized"})
        route = _route(request)
        queue = self.script.get(route, [])
        failure = queue.pop(0) if queue else None
        if failure and not failure.commit:
            return _failure_response(failure, request)
        response = self._dispatch(route, request)
        if failure:  # commit=True: the write has landed, now report a failure anyway
            return _failure_response(failure, request)
        return response

    def _dispatch(self, route: str, request: httpx.Request) -> httpx.Response:
        if route == "search":
            return self._search(request.url.params)
        if route == "create":
            return httpx.Response(201, json=self._create(json.loads(request.content)))
        ticket_id = request.url.path.split("/")[2].upper()
        ticket = self.tickets.get(ticket_id)
        if ticket is None:
            return httpx.Response(404, json={"detail": f"ticket {ticket_id} not found"})
        if route == "get":
            return httpx.Response(200, json=ticket)
        if route == "update":
            changes = {k: v for k, v in json.loads(request.content).items() if v is not None}
            if not changes:
                return httpx.Response(400, json={"detail": "nothing to update"})
            ticket.update(changes)
            return httpx.Response(200, json=ticket)
        self.next_comment += 1
        comment = {"id": f"CMT-{self.next_comment:04d}", **json.loads(request.content)}
        ticket["comments"].append(comment)
        return httpx.Response(201, json=comment)

    def _create(self, body: dict[str, Any]) -> dict[str, Any]:
        return self.add(**body, status="open", assignee=None, comments=[])

    def _search(self, params: httpx.QueryParams) -> httpx.Response:
        items = list(reversed(self.tickets.values()))  # newest first, like the real sort
        for key in ("status", "priority", "category"):
            if key in params:
                items = [t for t in items if t[key] == params[key]]
        if "requester" in params:
            items = [t for t in items if t["requester"].lower() == params["requester"].lower()]
        if "q" in params:
            q = params["q"].lower()
            items = [t for t in items if q in t["title"].lower() or q in t["description"].lower()]
        page, size = int(params.get("page", 1)), int(params.get("page_size", 10))
        start = (page - 1) * size
        page_items = [
            {k: v for k, v in t.items() if k != "comments"} for t in items[start : start + size]
        ]
        has_more = self.always_has_more or start + size < len(items)
        body = {
            "items": page_items,
            "page": page,
            "page_size": size,
            "total": len(items),
            "has_more": has_more,
        }
        return httpx.Response(200, json=body)


def _route(request: httpx.Request) -> str:
    path, method = request.url.path, request.method
    if path == "/tickets":
        return "search" if method == "GET" else "create"
    if re.fullmatch(r"/tickets/[^/]+/comments", path):
        return "comment"
    return "get" if method == "GET" else "update"


def _failure_response(failure: Fail, request: httpx.Request) -> httpx.Response:
    if failure.exc is not None:
        raise failure.exc("scripted failure", request=request)
    headers = {"Retry-After": failure.retry_after} if failure.retry_after else {}
    return httpx.Response(failure.status or 500, headers=headers, json={"error": "scripted"})


class FakeTime:
    """Injected clock and sleep, so retries run instantly and waits can be asserted."""

    def __init__(self) -> None:
        self.now = 0.0
        self.sleeps: list[float] = []

    def clock(self) -> float:
        return self.now

    async def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


@pytest.fixture
def fake() -> FakeTicketing:
    return FakeTicketing()


@pytest.fixture
def fake_time() -> FakeTime:
    return FakeTime()


@pytest.fixture
def make_client(fake: FakeTicketing, fake_time: FakeTime):
    import random

    def _make(**kwargs: Any) -> TicketingClient:
        return TicketingClient(
            "http://ticketing.test",
            API_KEY,
            transport=fake.transport(),
            sleep=fake_time.sleep,
            clock=fake_time.clock,
            rng=random.Random(7),  # noqa: S311 - jitter, not crypto
            **kwargs,
        )

    return _make
