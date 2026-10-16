"""TicketingClient: retries, pagination and the duplicate check (CLAUDE.md invariant 4)."""

import httpx
import pytest
from conftest import Fail

from ticket_agent import ticketing_api as client_module
from ticket_agent.ticketing_api import TicketingClient, TicketingError

pytestmark = pytest.mark.anyio

NEW = {"title": "Laptop won't boot", "requester": "Priya Raman", "priority": "high"}


# --- pagination and filters ---


async def test_pagination_follows_has_more_and_returns_everything(fake, make_client):
    for _ in range(23):
        fake.add(requester="Tom Okafor")
    fake.add(requester="Someone Else")
    async with make_client(page_size=10) as client:
        result, tel = await client.search_tickets(requester="Tom Okafor")
    assert fake.count("GET", "/tickets") == 3
    assert len(result["items"]) == 23 and result["total"] == 23
    assert {t["requester"] for t in result["items"]} == {"Tom Okafor"}
    assert result["truncated"] is False
    # Three pages and no retry: one attempt, three requests.
    assert (tel.status, tel.attempts, tel.requests) == ("ok", 1, 3)


async def test_search_stops_at_the_limit_and_keeps_the_full_total(fake, make_client):
    for _ in range(30):
        fake.add(status="open")
    async with make_client(page_size=10, search_limit=15) as client:
        result, _ = await client.search_tickets(status="open")
    assert len(result["items"]) == 15 and result["total"] == 30 and result["truncated"]
    assert fake.count("GET", "/tickets") == 2  # stops paging once the limit is reached


async def test_search_never_pages_forever(fake, make_client):
    fake.always_has_more = True  # a broken API that always says there is more
    fake.add()
    async with make_client(page_size=1, search_limit=5) as client:
        result, _ = await client.search_tickets()
    assert len(result["items"]) == 1 and fake.count("GET", "/tickets") == 2  # the empty page


async def test_a_ticket_id_cannot_leave_the_tickets_path(fake, make_client):
    raw_paths: list[bytes] = []

    def record(request: httpx.Request) -> None:
        raw_paths.append(request.url.raw_path)

    fake.intercept = record
    async with make_client() as client:
        with pytest.raises(TicketingError):
            await client.get_ticket("../health")
    assert raw_paths == [b"/tickets/..%2Fhealth"]  # one encoded segment, not GET /health


async def test_duplicate_check_reads_count_as_requests_not_attempts(fake, make_client):
    ticket = fake.add()
    async with make_client() as client:
        _, tel = await client.add_comment(ticket["id"], author="Aisha Khan", body="Still down")
    assert (tel.status, tel.attempts, tel.requests) == ("ok", 1, 2)  # the snapshot + the POST


# --- retry policy ---


async def test_429_waits_for_retry_after_then_succeeds(fake, fake_time, make_client):
    ticket = fake.add()
    fake.fail("get", Fail(status=429, retry_after="1"))
    async with make_client() as client:
        _, tel = await client.get_ticket(ticket["id"])
    assert (tel.status, tel.attempts, tel.retry_reasons) == ("retried", 2, ["429"])
    assert fake_time.sleeps == [1]


async def test_retry_after_is_capped(fake, fake_time, make_client):
    ticket = fake.add()
    fake.fail("get", Fail(status=429, retry_after="30"))
    async with make_client() as client:
        await client.get_ticket(ticket["id"])
    assert fake_time.sleeps == [5]


async def test_500_twice_then_ok_backs_off_with_full_jitter(fake, fake_time, make_client):
    ticket = fake.add()
    fake.fail("get", Fail(status=500), Fail(status=500))
    async with make_client() as client:
        _, tel = await client.get_ticket(ticket["id"])
    assert (tel.status, tel.attempts) == ("retried", 3)
    base, cap = client_module.BACKOFF_BASE_S, client_module.BACKOFF_CAP_S
    for n, slept in enumerate(fake_time.sleeps, start=1):
        assert 0 <= slept <= min(cap, base * 2 ** (n - 1))


async def test_five_500s_is_an_error_after_five_attempts(fake, fake_time, make_client):
    ticket = fake.add()
    fake.fail("get", *[Fail(status=500)] * 5)
    async with make_client() as client:
        with pytest.raises(TicketingError) as err:
            await client.get_ticket(ticket["id"])
    assert "not responding" in err.value.message
    assert (err.value.telemetry.status, err.value.telemetry.attempts) == ("error", 5)
    assert len(fake_time.sleeps) == 4  # no pointless wait after the last attempt
    assert fake.count("GET", "/tickets/") == 5


@pytest.mark.parametrize("status", [400, 401, 404, 422])
async def test_other_4xx_are_never_retried(fake, make_client, status):
    ticket = fake.add()
    fake.fail("get", Fail(status=status))
    async with make_client() as client:
        with pytest.raises(TicketingError) as err:
            await client.get_ticket(ticket["id"])
    assert fake.count("GET", "/tickets/") == 1
    assert err.value.telemetry.status == "error"


async def test_unknown_ticket_is_not_found(fake, make_client):
    async with make_client() as client:
        with pytest.raises(TicketingError) as err:
            await client.get_ticket("TCK-9999")
    assert err.value.message == "The ticket was not found."


async def test_bad_key_is_config_and_never_leaks_the_key(fake):
    async with TicketingClient(
        "http://ticketing.test", "wrong-key", transport=fake.transport()
    ) as client:
        with pytest.raises(TicketingError) as err:
            await client.get_ticket("TCK-1")
    assert "credentials" in err.value.message
    assert "wrong-key" not in f"{err.value} {err.value!r}"


async def test_connect_error_is_retried(fake, make_client):
    ticket = fake.add()
    fake.fail("get", Fail(exc=httpx.ConnectError))
    async with make_client() as client:
        _, tel = await client.get_ticket(ticket["id"])
    assert tel.retry_reasons == ["ConnectError"]


async def test_duration_comes_from_the_injected_clock(fake, fake_time, make_client):
    ticket = fake.add()
    fake.fail("get", Fail(status=429, retry_after="2"))
    async with make_client() as client:
        _, tel = await client.get_ticket(ticket["id"])
    assert tel.duration_ms == 2000


# --- create_ticket: the duplicate check ---


async def test_create_landed_then_500_returns_existing_without_second_post(fake, make_client):
    fake.fail("create", Fail(status=500, commit=True))
    async with make_client() as client:
        ticket, tel = await client.create_ticket(**NEW)
    assert fake.count("POST", "/tickets") == 1
    assert len(fake.tickets) == 1 and ticket["id"] in fake.tickets
    assert tel.status == "retried"


async def test_create_landed_then_read_timeout_returns_existing(fake, make_client):
    fake.fail("create", Fail(exc=httpx.ReadTimeout, commit=True))
    async with make_client() as client:
        ticket, tel = await client.create_ticket(**NEW)
    assert fake.count("POST", "/tickets") == 1 and len(fake.tickets) == 1
    assert tel.retry_reasons[0] == "ReadTimeout"


async def test_create_500_before_landing_retries_and_creates_exactly_one(fake, make_client):
    fake.fail("create", Fail(status=500))
    async with make_client() as client:
        ticket, tel = await client.create_ticket(**NEW)
    assert fake.count("POST", "/tickets") == 2
    assert len(fake.tickets) == 1 and ticket["title"] == NEW["title"]
    assert tel.status == "retried"


async def test_create_429_makes_no_duplicate_check(fake, make_client):
    fake.fail("create", Fail(status=429, retry_after="1"))
    async with make_client() as client:
        await client.create_ticket(**NEW)
    assert fake.requests == [("GET", "/tickets"), ("POST", "/tickets"), ("POST", "/tickets")]


async def test_create_connect_error_is_checked_like_any_other_non_429(fake, make_client):
    fake.fail("create", Fail(exc=httpx.ConnectError))
    async with make_client() as client:
        await client.create_ticket(**NEW)
    gets_and_posts = ["GET", "POST", "GET", "POST"]  # snapshot, try, check (finds nothing), retry
    assert [method for method, _ in fake.requests] == gets_and_posts


async def test_create_check_get_500_is_retried_not_treated_as_not_found(fake, make_client):
    fake.fail("create", Fail(status=500, commit=True))

    def first_check_fails(request: httpx.Request) -> httpx.Response | None:
        # GET #1 is the snapshot; GET #2 is the first duplicate-check attempt.
        is_first_check = request.method == "GET" and fake.count("GET", "/tickets") == 2
        return httpx.Response(500) if is_first_check else None

    fake.intercept = first_check_fails
    async with make_client() as client:
        ticket, tel = await client.create_ticket(**NEW)
    assert fake.count("POST", "/tickets") == 1, "a failed check must not look like 'not landed'"
    assert len(fake.tickets) == 1 and ticket["id"] in fake.tickets
    assert tel.retry_reasons == ["500", "500"]  # the write, then the failed check


async def test_create_without_snapshot_fails_closed(fake, make_client):
    fake.fail("search", *[Fail(status=500)] * 5)
    async with make_client() as client:
        with pytest.raises(TicketingError) as err:
            await client.create_ticket(**NEW)
    assert "not responding" in err.value.message
    assert fake.count("POST", "/tickets") == 0 and not fake.tickets


async def test_create_is_not_retried_when_the_check_cannot_run(fake, make_client):
    fake.fail("create", Fail(exc=httpx.ReadTimeout, commit=True))

    def checks_fail(request: httpx.Request) -> httpx.Response | None:
        after_post = fake.count("POST", "/tickets") > 0
        return httpx.Response(500) if request.method == "GET" and after_post else None

    fake.intercept = checks_fail
    async with make_client() as client:
        with pytest.raises(TicketingError) as err:
            await client.create_ticket(**NEW)
    assert "can't confirm" in err.value.message
    assert err.value.telemetry.status == "unknown"
    assert fake.count("POST", "/tickets") == 1, "never retry a write we could not rule out"


async def test_create_ignores_an_identical_ticket_that_existed_before(fake, make_client):
    fake.add(**NEW, description="", category="general")  # already there before this call
    fake.fail("create", Fail(status=500))
    async with make_client() as client:
        ticket, _ = await client.create_ticket(**NEW)
    assert len(fake.tickets) == 2, "the user asked for a new one; the old one is not ours"
    assert fake.count("POST", "/tickets") == 2


async def test_create_needs_every_sent_field_to_match(fake, make_client):
    # Someone else's ticket with our title and requester lands between snapshot and check,
    # and our own first POST fails before it is written.
    def someone_else_writes_first(request: httpx.Request) -> httpx.Response | None:
        if request.method == "POST" and fake.count("POST", "/tickets") == 1:
            fake.add(**(NEW | {"priority": "low"}), description="", category="general")
            return httpx.Response(500)
        return None

    fake.intercept = someone_else_writes_first
    async with make_client() as client:
        ticket, _ = await client.create_ticket(**NEW)
    assert ticket["priority"] == "high" and len(fake.tickets) == 2


# --- add_comment and update_ticket ---


async def test_comment_landed_then_500_is_not_posted_twice(fake, make_client):
    ticket = fake.add()
    fake.fail("comment", Fail(status=500, commit=True))
    async with make_client() as client:
        comment, tel = await client.add_comment(ticket["id"], author="Aisha", body="Still down")
    assert fake.count("POST", f"/tickets/{ticket['id']}/comments") == 1
    assert len(ticket["comments"]) == 1 and comment["id"] == ticket["comments"][0]["id"]
    assert tel.status == "retried"


async def test_comment_500_before_landing_retries_and_adds_exactly_one(fake, make_client):
    ticket = fake.add()
    fake.fail("comment", Fail(status=500))
    async with make_client() as client:
        await client.add_comment(ticket["id"], author="Aisha Khan", body="Still down")
    assert fake.count("POST", f"/tickets/{ticket['id']}/comments") == 2
    assert len(ticket["comments"]) == 1


async def test_comment_on_unknown_ticket_is_not_found_and_never_posts(fake, make_client):
    async with make_client() as client:
        with pytest.raises(TicketingError) as err:
            await client.add_comment("TCK-9999", author="Aisha Khan", body="hello")
    assert "not found" in err.value.message and fake.count("POST", "/tickets") == 0


async def test_an_update_that_kept_failing_with_500_is_unknown_not_error(fake, make_client):
    # PATCH has no duplicate check, so a 500 may have been applied: nobody can say it wasn't.
    ticket = fake.add()
    fake.fail("update", *[Fail(status=500)] * 5)
    async with make_client() as client:
        with pytest.raises(TicketingError) as err:
            await client.update_ticket(ticket["id"], priority="urgent")
    assert err.value.unconfirmed and err.value.telemetry.status == "unknown"
    assert "can't confirm" in err.value.message


async def test_a_write_rejected_only_with_429s_is_an_error(fake, make_client):
    # A 429 is rejected before anything is written: certainly not landed.
    ticket = fake.add()
    fake.fail("update", *[Fail(status=429, retry_after="1")] * 5)
    async with make_client() as client:
        with pytest.raises(TicketingError) as err:
            await client.update_ticket(ticket["id"], priority="urgent")
    assert not err.value.unconfirmed and err.value.telemetry.status == "error"


async def test_update_500_is_retried_without_a_duplicate_check(fake, make_client):
    ticket = fake.add()
    fake.fail("update", Fail(status=500))
    async with make_client() as client:
        updated, tel = await client.update_ticket(ticket["id"], priority="urgent")
    assert updated["priority"] == "urgent" and tel.status == "retried"
    assert fake.requests == [("PATCH", f"/tickets/{ticket['id']}")] * 2
