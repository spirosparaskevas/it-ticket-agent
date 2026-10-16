"""MCP server exposing the ticketing API as five tools.

Every tool returns a CallToolResult: data in structured_content, a one-line summary in content,
and the client's telemetry in _meta["telemetry"] (the agent strips _meta before the LLM).
The read tools carry readOnlyHint; the agent treats every other tool as a write that needs the
user's approval.
"""

from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from typing import Annotated, Any

from mcp.server.mcpserver import Context, MCPServer
from mcp.types import CallToolResult, TextContent, ToolAnnotations
from pydantic import Field

from ticket_agent.config import Settings
from ticket_agent.ticketing_api import (
    SEARCH_LIMIT,
    Priority,
    Status,
    Telemetry,
    TicketingClient,
    TicketingError,
)

# The agent runs only these without approval; every other tool is a write (fail closed).
READ_ONLY = ToolAnnotations(read_only_hint=True)

# The pattern guides the model and keeps a crafted id ("../admin") out of the URL path.
TicketId = Annotated[
    str, Field(pattern=r"^TCK-\d+$", description="Ticket id in capitals, e.g. TCK-1002.")
]
Person = Annotated[
    str, Field(min_length=2, max_length=80, description="Full name, e.g. 'Priya Raman'.")
]


def build_server(make_client: Callable[[], TicketingClient] | None = None) -> MCPServer:
    """`make_client` is called once, when the server starts. Default: built from Settings."""
    factory = make_client or _client_from_settings

    @asynccontextmanager
    async def lifespan(_: MCPServer) -> AsyncIterator[TicketingClient]:
        async with factory() as client:  # closed when the server stops
            yield client

    server = MCPServer("ticket-agent", version="0.1.0", lifespan=lifespan)

    @server.tool(
        annotations=READ_ONLY,
        description="Search IT support tickets. All filters are optional and combine with AND. "
        f"Returns up to {SEARCH_LIMIT} matching tickets and `total`, the full count; if "
        "`truncated` is true, narrow the filters to see the rest. Results do not include "
        "comments; use get_ticket for those. Use it to list or count tickets, e.g. "
        '"my open tickets" = requester + status.',
    )
    async def search_tickets(
        ctx: Context,
        q: Annotated[str | None, Field(description="Words in the title or description.")] = None,
        status: Annotated[Status | None, Field(description="Only this status.")] = None,
        priority: Annotated[Priority | None, Field(description="Only this priority.")] = None,
        requester: Annotated[
            str | None,
            Field(description="Full name of who raised it, e.g. 'Priya Raman' (any letter case)."),
        ] = None,
        category: Annotated[
            str | None, Field(description="e.g. network, access, hardware, software.")
        ] = None,
    ) -> CallToolResult:
        return await _call(
            ctx,
            TicketingClient.search_tickets,
            _search_line,
            q=q, status=status, priority=priority, requester=requester, category=category,
        )  # fmt: skip

    @server.tool(
        annotations=READ_ONLY,
        description="Get one ticket by id (e.g. TCK-1002), including all its comments. "
        "Use it for the status or details of a specific ticket.",
    )
    async def get_ticket(ctx: Context, ticket_id: TicketId) -> CallToolResult:
        return await _call(ctx, TicketingClient.get_ticket, _ticket_line, ticket_id=ticket_id)

    @server.tool(
        description="Create a new IT support ticket. It starts as open, with no assignee. "
        "This needs the user's confirmation: calling it proposes the ticket, and it is created "
        "only after the user confirms."
    )
    async def create_ticket(
        ctx: Context,
        title: Annotated[
            str, Field(min_length=3, max_length=120, description="Short summary of the problem.")
        ],
        requester: Person,
        description: Annotated[
            str, Field(max_length=2000, description="Details: what happened, since when.")
        ] = "",
        priority: Annotated[Priority, Field(description="How urgent it is.")] = "medium",
        category: Annotated[
            str, Field(max_length=40, description="e.g. network, access, hardware, software.")
        ] = "general",
    ) -> CallToolResult:
        return await _call(
            ctx,
            TicketingClient.create_ticket,
            lambda t: f'Created {t["id"]} "{t["title"]}" for {t["requester"]}.',
            title=title, requester=requester, description=description,
            priority=priority, category=category,
        )  # fmt: skip

    @server.tool(
        description="Change a ticket's status, priority or assignee; give at least one. It cannot "
        "change title, description, category or requester, and cannot remove an assignee. This "
        "needs the user's confirmation: calling it proposes the change, and it is applied only "
        "after the user confirms."
    )
    async def update_ticket(
        ctx: Context,
        ticket_id: TicketId,
        status: Annotated[Status | None, Field(description="New status.")] = None,
        priority: Annotated[Priority | None, Field(description="New priority.")] = None,
        assignee: Annotated[
            str | None, Field(max_length=80, description="Full name of the new assignee.")
        ] = None,
    ) -> CallToolResult:
        return await _call(
            ctx, TicketingClient.update_ticket, _ticket_line,
            ticket_id=ticket_id, status=status, priority=priority, assignee=assignee,
        )  # fmt: skip

    @server.tool(
        description="Add a comment to a ticket. This needs the user's confirmation: calling it "
        "proposes the comment, and it is added only after the user confirms."
    )
    async def add_comment(
        ctx: Context,
        ticket_id: TicketId,
        author: Person,
        body: Annotated[str, Field(min_length=1, max_length=2000, description="The comment text.")],
    ) -> CallToolResult:
        return await _call(
            ctx,
            TicketingClient.add_comment,
            lambda c: f"Added comment {c['id']} to {ticket_id}.",
            ticket_id=ticket_id, author=author, body=body,
        )  # fmt: skip

    return server


def _client_from_settings() -> TicketingClient:
    settings = Settings()
    return TicketingClient(settings.ticketing.url, settings.ticketing_api_key.get_secret_value())


async def _call(
    ctx: Context,
    operation: Callable[..., Awaitable[tuple[Any, Telemetry]]],
    summarize: Callable[[Any], str],
    **arguments: Any,
) -> CallToolResult:
    client: TicketingClient = ctx.request_context.lifespan_context
    try:
        data, telemetry = await operation(client, **arguments)
    except TicketingError as err:
        # Returned, not raised as ToolError: the SDK's error path would drop our telemetry.
        return _result(err.message, None, err.telemetry, is_error=True)
    return _result(summarize(data), data, telemetry)


def _result(
    text: str, data: Any, telemetry: Telemetry, *, is_error: bool = False
) -> CallToolResult:
    return CallToolResult(
        content=[TextContent(type="text", text=text)],
        structured_content=data,
        is_error=is_error,
        _meta={"telemetry": telemetry.model_dump()},
    )


def _search_line(found: dict[str, Any]) -> str:
    total = found["total"]
    line = f"Found {total} ticket{'' if total == 1 else 's'}."
    if found["truncated"]:
        line += f" Showing the first {len(found['items'])}; narrow the filters to see the rest."
    return line


def _ticket_line(ticket: dict[str, Any]) -> str:
    n = len(ticket["comments"])
    return (
        f'{ticket["id"]} "{ticket["title"]}" is {ticket["status"]} '
        f"(priority {ticket['priority']}, {n} comment{'' if n == 1 else 's'})."
    )
