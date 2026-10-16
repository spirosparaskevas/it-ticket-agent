"""The MCP server, through the SDK's in-memory Client (no subprocess, no network)."""

import httpx
import pytest
from conftest import API_KEY, Fail
from mcp import Client

from ticket_agent.mcp_server.server import build_server
from ticket_agent.ticketing_api import TicketingClient

pytestmark = pytest.mark.anyio

TOOLS = {"search_tickets", "get_ticket", "create_ticket", "update_ticket", "add_comment"}


@pytest.fixture
def server(make_client):
    return build_server(make_client=make_client)


async def test_lists_exactly_the_five_tools_with_descriptions(server):
    async with Client(server) as mcp:
        tools = {tool.name: tool for tool in (await mcp.list_tools()).tools}
    assert set(tools) == TOOLS
    assert all(tool.description for tool in tools.values())
    for name in ("create_ticket", "update_ticket", "add_comment"):
        assert "confirm" in tools[name].description
    assert "comments" in tools["search_tickets"].description  # says results carry none


async def test_input_schemas_carry_the_api_limits(server):
    async with Client(server) as mcp:
        tools = {tool.name: tool for tool in (await mcp.list_tools()).tools}
    create = tools["create_ticket"].input_schema
    assert set(create["required"]) == {"title", "requester"}
    title = create["properties"]["title"]
    assert (title["minLength"], title["maxLength"]) == (3, 120)
    assert "ctx" not in create["properties"]
    assert tools["search_tickets"].input_schema.get("required", []) == []
    for prop in tools["search_tickets"].input_schema["properties"].values():
        assert prop.get("description"), "every parameter tells the LLM what it is for"


async def test_search_returns_data_summary_and_telemetry_in_meta(fake, server):
    for _ in range(3):
        fake.add(requester="Priya Raman")
    async with Client(server) as mcp:
        result = await mcp.call_tool("search_tickets", {"requester": "Priya Raman"})
    assert not result.is_error
    assert result.structured_content["total"] == 3 and len(result.structured_content["items"]) == 3
    assert result.content[0].text == "Found 3 tickets."
    assert result.meta["telemetry"]["status"] == "ok"


async def test_retried_call_reports_retried_telemetry(fake, server):
    ticket = fake.add(title="VPN drops")
    fake.fail("get", Fail(status=500))
    async with Client(server) as mcp:
        result = await mcp.call_tool("get_ticket", {"ticket_id": ticket["id"]})
    assert result.structured_content["id"] == ticket["id"]
    telemetry = result.meta["telemetry"]
    assert (telemetry["status"], telemetry["attempts"]) == ("retried", 2)


async def test_final_failure_is_an_error_result_with_telemetry_and_exact_message(server):
    async with Client(server) as mcp:
        result = await mcp.call_tool("get_ticket", {"ticket_id": "TCK-9999"})
    assert result.is_error
    assert result.content[0].text == "The ticket was not found."  # no SDK prefix
    assert result.meta["telemetry"]["status"] == "error"


async def test_create_update_and_comment_round_trip(fake, server):
    async with Client(server) as mcp:
        created = await mcp.call_tool(
            "create_ticket", {"title": "Printer jammed", "requester": "Tom Okafor"}
        )
        ticket_id = created.structured_content["id"]
        updated = await mcp.call_tool("update_ticket", {"ticket_id": ticket_id, "priority": "high"})
        comment = await mcp.call_tool(
            "add_comment", {"ticket_id": ticket_id, "author": "Tom Okafor", "body": "Still jammed"}
        )
    assert created.content[0].text == f'Created {ticket_id} "Printer jammed" for Tom Okafor.'
    assert updated.structured_content["priority"] == "high"
    assert comment.structured_content["body"] == "Still jammed"
    assert fake.tickets[ticket_id]["comments"][0]["author"] == "Tom Okafor"


async def test_invalid_arguments_never_reach_the_api(fake, server):
    async with Client(server) as mcp:
        result = await mcp.call_tool("create_ticket", {"title": "ab", "requester": "Tom Okafor"})
    assert result.is_error
    assert fake.requests == []


@pytest.mark.parametrize("ticket_id", ["../health", "TCK-1002/../../admin/reset", "1002"])
async def test_a_ticket_id_must_look_like_one(fake, server, ticket_id):
    async with Client(server) as mcp:
        tools = {tool.name: tool for tool in (await mcp.list_tools()).tools}
        result = await mcp.call_tool("get_ticket", {"ticket_id": ticket_id})
    assert result.is_error and fake.requests == []
    assert tools["get_ticket"].input_schema["properties"]["ticket_id"]["pattern"] == r"^TCK-\d+$"


async def test_a_broad_search_is_capped_and_says_so(fake, make_client):
    for _ in range(60):
        fake.add()
    async with Client(build_server(make_client=make_client)) as mcp:
        result = await mcp.call_tool("search_tickets", {})
    assert len(result.structured_content["items"]) == 50
    assert result.structured_content["total"] == 60 and result.structured_content["truncated"]
    assert "narrow the filters" in result.content[0].text


async def test_a_crash_shows_no_internals(fake, server):
    fake.intercept = lambda request: httpx.Response(200, json={"unexpected": "shape"})
    async with Client(server) as mcp:
        result = await mcp.call_tool("search_tickets", {})
    assert result.is_error
    assert result.content[0].text == "Error executing tool search_tickets"


async def test_lifespan_creates_one_client_and_closes_it(fake):
    class ClosingTransport(httpx.MockTransport):
        closed = False

        async def aclose(self) -> None:
            self.closed = True

    transport = ClosingTransport(fake.handle)
    created: list[TicketingClient] = []

    def make_client() -> TicketingClient:
        created.append(TicketingClient("http://ticketing.test", API_KEY, transport=transport))
        return created[-1]

    ticket = fake.add()
    async with Client(build_server(make_client=make_client)) as mcp:
        for _ in range(3):
            await mcp.call_tool("get_ticket", {"ticket_id": ticket["id"]})
        assert not transport.closed
    assert len(created) == 1
    assert transport.closed
