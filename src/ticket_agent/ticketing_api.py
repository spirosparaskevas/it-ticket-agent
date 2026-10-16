"""Async client for the mock ticketing API, with retries (CLAUDE.md invariant 4).

Endpoint shapes and failure behaviour: mock-ticketing/app.py.
"""

import asyncio
import random
import time
from collections.abc import Awaitable, Callable
from typing import Any, Literal
from urllib.parse import quote

import httpx
import structlog
from pydantic import BaseModel, Field

logger = structlog.get_logger()

Status = Literal["open", "in_progress", "resolved", "closed"]
Priority = Literal["low", "medium", "high", "urgent"]

MAX_ATTEMPTS = 5
BACKOFF_BASE_S = 0.25
BACKOFF_CAP_S = 2.0
RETRY_AFTER_CAP_S = 5.0
TIMEOUT = httpx.Timeout(connect=2.0, read=5.0, write=5.0, pool=2.0)
SEARCH_LIMIT = 50  # tickets a search returns; the total is always the full count

# Safe to show the user: no API key, no raw response body.
_NOT_RESPONDING = "The ticketing system is not responding right now."
_UNCONFIRMED = (
    "The ticketing system is not responding, so I can't confirm the request went through."
)
_REJECTED_KEY = "The ticketing service rejected this service's credentials."


class Telemetry(BaseModel):
    """What one call cost.

    `attempts` is 1 plus the retries of the operation's own requests. `requests` counts every
    HTTP request sent, the duplicate-check reads included. `unknown`: a write that failed but
    may have landed.
    """

    status: Literal["ok", "retried", "error", "unknown"] = "ok"
    attempts: int = 1
    requests: int = 0
    duration_ms: int = 0
    retry_reasons: list[str] = Field(default_factory=list)


class TicketingError(Exception):
    """A call that finally failed. `message` is safe to show the user. `unconfirmed`: a write
    may have landed, and nothing could rule it out."""

    def __init__(self, message: str, telemetry: Telemetry, *, unconfirmed: bool = False) -> None:
        super().__init__(message)
        self.message = message
        self.telemetry = telemetry
        self.unconfirmed = unconfirmed


class TicketingClient:
    def __init__(
        self,
        base_url: str,
        api_key: str,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
        page_size: int = 50,
        search_limit: int = SEARCH_LIMIT,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
        clock: Callable[[], float] = time.monotonic,
        rng: random.Random | None = None,
    ) -> None:
        self._http = httpx.AsyncClient(
            base_url=base_url, headers={"X-API-Key": api_key}, transport=transport, timeout=TIMEOUT
        )
        self._page_size = page_size
        self._search_limit = search_limit
        self._sleep = sleep
        self._clock = clock
        self._rng = rng or random.Random()  # noqa: S311 - backoff jitter, not cryptography

    async def __aenter__(self) -> "TicketingClient":
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self._http.aclose()

    # --- the five operations; each returns (data, Telemetry) or raises TicketingError ---

    async def search_tickets(self, **filters: str | None) -> tuple[dict[str, Any], Telemetry]:
        """Filters: q, status, priority, requester, category.

        Returns {"items", "total", "truncated"}: at most `search_limit` items, so one broad
        search cannot flood the LLM's context. `total` is still the full count.
        """
        return await self._run(lambda tel: self._search(filters, tel, limit=self._search_limit))

    async def get_ticket(self, ticket_id: str) -> tuple[dict[str, Any], Telemetry]:
        return await self._run(lambda tel: self._request("GET", _ticket_path(ticket_id), tel))

    async def update_ticket(self, ticket_id: str, **changes: str | None):
        # PATCH sets absolute values, so repeating it is harmless: no duplicate check.
        body = {key: value for key, value in changes.items() if value is not None}
        return await self._run(
            lambda tel: self._request("PATCH", _ticket_path(ticket_id), tel, body=body)
        )

    async def create_ticket(
        self,
        *,
        title: str,
        requester: str,
        description: str = "",
        priority: Priority = "medium",
        category: str = "general",
    ) -> tuple[dict[str, Any], Telemetry]:
        # Send every field (with the API's defaults) so a landed ticket can be matched on all.
        body = {"title": title, "requester": requester, "description": description}
        body |= {"priority": priority, "category": category}

        async def create(tel: Telemetry) -> dict[str, Any]:
            async def my_tickets() -> list[dict[str, Any]]:
                # Every page: the duplicate check must see all of this requester's tickets.
                found = await self._search({"requester": requester}, tel, primary=False)
                return found["items"]

            known_ids = {t["id"] for t in await my_tickets()}  # snapshot before the first attempt

            async def find_landed() -> dict[str, Any] | None:
                new = [t for t in await my_tickets() if t["id"] not in known_ids]
                return next((t for t in new if all(t[k] == v for k, v in body.items())), None)

            return await self._request("POST", "/tickets", tel, body=body, find_landed=find_landed)

        return await self._run(create)

    async def add_comment(
        self, ticket_id: str, *, author: str, body: str
    ) -> tuple[dict[str, Any], Telemetry]:
        path = _ticket_path(ticket_id)
        comment = {"author": author, "body": body}

        async def add(tel: Telemetry) -> dict[str, Any]:
            # Search results carry no comments, so read the ticket itself.
            async def comments() -> list[dict[str, Any]]:
                return (await self._request("GET", path, tel, primary=False))["comments"]

            known_ids = {c["id"] for c in await comments()}  # snapshot before the first attempt

            async def find_landed() -> dict[str, Any] | None:
                new = [c for c in await comments() if c["id"] not in known_ids]
                return next((c for c in new if (c["author"], c["body"]) == (author, body)), None)

            return await self._request(
                "POST", f"{path}/comments", tel, body=comment, find_landed=find_landed
            )

        return await self._run(add)

    # --- internals ---

    async def _run(self, operation: Callable[[Telemetry], Awaitable[Any]]) -> tuple[Any, Telemetry]:
        tel = Telemetry()
        start = self._clock()
        try:
            data = await operation(tel)
            tel.status = "retried" if tel.retry_reasons else "ok"
        except TicketingError as err:
            tel.status = "unknown" if err.unconfirmed else "error"
            raise
        finally:
            tel.duration_ms = round((self._clock() - start) * 1000)
        return data, tel

    async def _search(
        self,
        filters: dict[str, Any],
        tel: Telemetry,
        *,
        limit: int | None = None,
        primary: bool = True,
    ) -> dict[str, Any]:
        """Follows the pages until `limit` items are collected; `limit=None` reads them all."""
        params = {key: value for key, value in filters.items() if value is not None}
        items: list[dict[str, Any]] = []
        page = 1
        while True:
            body = await self._request(
                "GET", "/tickets", tel, primary=primary,
                params=params | {"page": page, "page_size": self._page_size},
            )  # fmt: skip
            items += body["items"]
            full = limit is not None and len(items) >= limit
            # An empty page ends it too: an API that always says has_more must not loop forever.
            if full or not body["has_more"] or not body["items"]:
                total = body["total"]
                shown = items[:limit] if limit is not None else items
                return {"items": shown, "total": total, "truncated": len(shown) < total}
            page += 1

    async def _request(
        self,
        method: str,
        path: str,
        tel: Telemetry,
        *,
        params: dict[str, Any] | None = None,
        body: dict[str, Any] | None = None,
        find_landed: Callable[[], Awaitable[dict[str, Any] | None]] | None = None,
        primary: bool = True,
    ) -> Any:
        """Send with retries. Returns the JSON body, or what `find_landed` says already landed.

        `primary=False` marks a helper read (a duplicate-check snapshot): it counts in
        `tel.requests` but not in `tel.attempts`.
        """
        unresolved = False  # a write may have landed and nothing has ruled it out
        for attempt in range(1, MAX_ATTEMPTS + 1):
            tel.requests += 1
            if primary and attempt > 1:
                tel.attempts += 1
            retry_after = None
            try:
                response = await self._http.request(method, path, params=params, json=body)
            except httpx.TransportError as exc:  # connect error, timeout, dropped connection
                reason, may_have_landed = type(exc).__name__, True
            else:
                if response.is_success:
                    return response.json()
                reason, may_have_landed = str(response.status_code), response.status_code >= 500
                if response.status_code == 429:
                    retry_after = _retry_after(response)
                elif response.status_code < 500:  # never retry other 4xx
                    message = _client_error(response.status_code)
                    raise TicketingError(message, tel, unconfirmed=unresolved)
            tel.retry_reasons.append(reason)
            # A 429 was rejected before doing anything. Anything else may have been saved.
            if may_have_landed and method != "GET":
                if find_landed is None:  # PATCH: a repeat is harmless, but it may have landed
                    unresolved = True
                else:
                    try:
                        landed = await find_landed()
                    except TicketingError:  # the check itself failed: never retry blindly
                        raise TicketingError(_UNCONFIRMED, tel, unconfirmed=True) from None
                    if landed is not None:
                        return landed
            if attempt < MAX_ATTEMPTS:
                delay = self._delay(attempt, retry_after)
                logger.warning(
                    "ticketing_retry",
                    method=method, path=path, attempt=attempt, reason=reason,
                    wait_s=round(delay, 2),
                )  # fmt: skip
                await self._sleep(delay)
        if unresolved:
            raise TicketingError(_UNCONFIRMED, tel, unconfirmed=True)
        raise TicketingError(_NOT_RESPONDING, tel)

    def _delay(self, attempt: int, retry_after: float | None) -> float:
        if retry_after is not None:
            return min(retry_after, RETRY_AFTER_CAP_S)
        return self._rng.uniform(0, min(BACKOFF_CAP_S, BACKOFF_BASE_S * 2 ** (attempt - 1)))


def _ticket_path(ticket_id: str) -> str:
    # Encoded as one path segment: "../admin" can never leave /tickets/.
    return f"/tickets/{quote(ticket_id, safe='')}"


def _retry_after(response: httpx.Response) -> float | None:
    try:
        return float(response.headers["Retry-After"])
    except (KeyError, ValueError):
        return None


def _client_error(code: int) -> str:
    if code in (401, 403):
        return _REJECTED_KEY
    if code == 404:
        return "The ticket was not found."
    return f"The ticketing system rejected the request (HTTP {code})."
