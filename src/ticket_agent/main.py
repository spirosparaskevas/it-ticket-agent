"""The AgentCore Runtime HTTP service: GET /ping and POST /invocations (CONTRACT.md), plus the UI
at GET / and what the UI shows about the agent at GET /info."""

import time
import uuid
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from importlib.resources import files
from typing import Literal

import structlog
from fastapi import FastAPI, Header, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import HTMLResponse, JSONResponse
from mcp import Client
from pydantic import BaseModel, Field, model_validator

from ticket_agent.agent.graph import Agent, ToolInfo, TurnResult
from ticket_agent.agent.llm import LLM, LLMError, OpenAIChatClient
from ticket_agent.agent.state import SessionStore
from ticket_agent.config import ModelChoice, Settings
from ticket_agent.mcp_server.server import build_server
from ticket_agent.observability import configure_logging
from ticket_agent.ticketing_api import TicketingClient

logger = structlog.get_logger()

SESSION_HEADER = "X-Amzn-Bedrock-AgentCore-Runtime-Session-Id"
LOCAL_SESSION = "default-session"  # local dev only: AgentCore always sends the header


class InvocationRequest(BaseModel):
    """CONTRACT.md's {"prompt"}, or {"approval"} from the UI's Accept and Decline buttons."""

    prompt: str | None = Field(default=None, min_length=1)
    approval: Literal["accept", "decline"] | None = None

    @model_validator(mode="after")
    def _one_of_them(self) -> "InvocationRequest":
        if (self.prompt is None) == (self.approval is None):
            raise ValueError("send either prompt or approval")
        return self


class AgentInfo(BaseModel):
    """For the UI's status bar. Never keys or endpoints."""

    models: list[ModelChoice]  # the model chain, in order
    tools: list[ToolInfo]
    started_at: str


def create_app(
    *,
    llm: LLM | None = None,
    make_client: Callable[[], TicketingClient] | None = None,
    settings: Settings | None = None,
) -> FastAPI:
    """Defaults come from the environment; tests pass a ScriptedLLM and a fake ticketing client."""

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        config = settings or Settings()
        configure_logging(config.agent.log_level)
        model = llm or OpenAIChatClient.from_settings(config)
        # One in-process MCP client for the whole service; it owns the one TicketingClient.
        async with Client(build_server(make_client=make_client)) as mcp:
            app.state.agent = await Agent.create(
                model, mcp, SessionStore(), max_steps=config.agent.max_steps
            )
            main_model = config.main_model
            chain = [main_model, *config.llm_fallback_models] if main_model else []
            app.state.info = AgentInfo(
                models=chain,
                tools=app.state.agent.tool_info(),
                started_at=datetime.now(UTC).isoformat(timespec="seconds"),
            )
            yield

    app = FastAPI(title="Ticket agent", lifespan=lifespan)

    @app.exception_handler(RequestValidationError)
    async def invalid_request(request: Request, exc: RequestValidationError) -> JSONResponse:
        # A rejected body gets a request id and a log line too, like every other request.
        request_id = str(uuid.uuid4())
        session_id = request.headers.get(SESSION_HEADER, LOCAL_SESSION)
        errors = jsonable_encoder(exc.errors())
        logger.warning(
            "invalid_request", request_id=request_id, session_id=session_id,
            errors=[error["msg"] for error in errors],
        )  # fmt: skip
        payload = {"error": "invalid_request", "message": "Send a JSON body with a prompt.",
                   "detail": errors, "request_id": request_id}  # fmt: skip
        return JSONResponse(payload, status_code=422, headers={"X-Request-Id": request_id})

    @app.get("/", response_class=HTMLResponse)
    async def ui() -> str:
        # Read through importlib.resources so it also works from the installed package.
        return files("ticket_agent").joinpath("ui", "index.html").read_text()

    @app.get("/ping")
    async def ping() -> dict[str, str]:
        # Static on purpose: never report a changing time_of_last_update.
        return {"status": "Healthy"}

    @app.get("/info")
    async def info(request: Request) -> AgentInfo:
        return request.app.state.info

    @app.post("/invocations")
    async def invocations(
        body: InvocationRequest,
        request: Request,
        session_id: str = Header(default=LOCAL_SESSION, alias=SESSION_HEADER),
    ) -> JSONResponse:
        request_id = str(uuid.uuid4())
        structlog.contextvars.clear_contextvars()
        structlog.contextvars.bind_contextvars(request_id=request_id, session_id=session_id)
        headers = {"X-Request-Id": request_id}
        start = time.monotonic()
        try:
            result = await request.app.state.agent.run_turn(
                session_id, body.prompt or "", approval=body.approval
            )
        except LLMError:
            logger.exception("llm_unavailable", request_id=request_id, session_id=session_id)
            message = "The language model is not available right now. Please try again shortly."
            payload = {"error": "llm_unavailable", "message": message, "request_id": request_id}
            return JSONResponse(payload, status_code=503, headers=headers)
        except Exception:
            # Not hidden: the traceback is logged. The caller only gets a safe message and an id.
            logger.exception("invocation_failed", request_id=request_id, session_id=session_id)
            payload = {"error": "internal_error", "message": "Something went wrong.",
                       "request_id": request_id}  # fmt: skip
            return JSONResponse(payload, status_code=500, headers=headers)
        _log_invocation(request_id, session_id, start, result)
        return JSONResponse(result.model_dump(mode="json"), headers=headers)

    return app


def _log_invocation(request_id: str, session_id: str, start: float, result: TurnResult) -> None:
    logger.info(
        "invocation",
        request_id=request_id,
        session_id=session_id,
        latency_ms=round((time.monotonic() - start) * 1000),
        tool_calls=[
            {
                "name": call.name,
                "status": call.status,
                "attempts": call.attempts,
                "duration_ms": call.duration_ms,
            }
            for call in result.tool_calls
        ],
        pending_action=result.pending_approval.action if result.pending_approval else None,
        token_usage=result.usage.model_dump(),
    )


app = create_app()
