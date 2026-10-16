"""The HTTP service (CONTRACT.md) with ScriptedLLM injected and the fake ticketing API."""

import asyncio
import json

import httpx
import pytest
import structlog
from conftest import Fail
from structlog.testing import capture_logs

from ticket_agent.agent.llm import LLMError, LLMResponse, ScriptedLLM, ToolCall, Usage
from ticket_agent.config import Settings
from ticket_agent.main import create_app
from ticket_agent.observability import configure_logging

pytestmark = pytest.mark.anyio

SESSION = "X-Amzn-Bedrock-AgentCore-Runtime-Session-Id"
SECRET = "SECRET-ticketing-key-do-not-log"  # noqa: S105 - a fake key, to prove it is not logged
NEW = {"title": "Laptop won't boot", "requester": "Priya Raman", "priority": "high"}


def say(text: str = "", *calls: ToolCall, tokens: int = 0) -> LLMResponse:
    return LLMResponse(text=text, tool_calls=list(calls), usage=Usage(input_tokens=tokens))


def call(name: str, **input) -> ToolCall:
    return ToolCall(id=f"call-{name}", name=name, input=input)


@pytest.fixture(autouse=True)
def _reset_logging():
    yield
    structlog.reset_defaults()


@pytest.fixture
def run_app(make_client):
    """Start the app (lifespan included) with the given LLM; yields an httpx client."""

    async def start(llm, stack, settings=None):
        settings = settings or Settings(ticketing_api_key=SECRET)
        app = create_app(llm=llm, make_client=make_client, settings=settings)
        await stack.enter_async_context(app.router.lifespan_context(app))
        transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
        return await stack.enter_async_context(
            httpx.AsyncClient(transport=transport, base_url="http://agent.test")
        )

    return start


@pytest.fixture
async def stack():
    from contextlib import AsyncExitStack

    async with AsyncExitStack() as exit_stack:
        yield exit_stack


async def test_ping(run_app, stack):
    http = await run_app(ScriptedLLM([]), stack)
    response = await http.get("/ping")
    assert (response.status_code, response.json()) == (200, {"status": "Healthy"})


async def test_info_lists_the_model_chain_and_the_tools_and_never_a_key(run_app, stack):
    settings = Settings(
        ticketing_api_key=SECRET,
        llm_provider="openai",
        llm_model="gpt-6-luna",
        llm_reasoning="medium",
        llm_fallback_models="gemini/gemini-3.5-flash",
        openai_api_key="sk-SECRET-openai",
    )
    http = await run_app(ScriptedLLM([]), stack, settings=settings)
    body = (await http.get("/info")).json()
    assert [(m["provider"], m["model"], m["reasoning"]) for m in body["models"]] == [
        ("openai", "gpt-6-luna", "medium"), ("gemini", "gemini-3.5-flash", None),
    ]  # fmt: skip
    tools = {tool["name"]: tool["read_only"] for tool in body["tools"]}
    assert tools["search_tickets"] is True and tools["create_ticket"] is False
    assert "SECRET" not in json.dumps(body) and body["started_at"]


async def test_invocation_follows_the_contract_shape(fake, run_app, stack):
    fake.add(id="TCK-1002", status="in_progress")
    get = call("get_ticket", ticket_id="TCK-1002")
    llm = ScriptedLLM([say("", get), say("It is in progress.")])
    http = await run_app(llm, stack)
    prompt = {"prompt": "status of TCK-1002?"}
    response = await http.post("/invocations", json=prompt, headers={SESSION: "s" * 40})
    body = response.json()
    assert response.status_code == 200
    assert body["output"] == "It is in progress."
    assert body["pending_approval"] is None
    assert [(c["name"], c["input"], c["status"]) for c in body["tool_calls"]] == [
        ("get_ticket", {"ticket_id": "TCK-1002"}, "ok")
    ]
    assert isinstance(body["tool_calls"][0]["duration_ms"], int)
    assert [event["type"] for event in body["trace"]] == ["llm", "tool", "llm"]  # extra field
    assert response.headers["X-Request-Id"]


async def test_a_staged_write_comes_back_as_pending_approval(fake, run_app, stack):
    llm = ScriptedLLM([say("", call("create_ticket", **NEW)), say("Shall I?")])
    http = await run_app(llm, stack)
    body = (await http.post("/invocations", json={"prompt": "create it"})).json()
    assert body["tool_calls"] == [] and fake.requests == []
    assert body["pending_approval"] == {"action": "create_ticket", "input": NEW}


async def test_the_approval_field_answers_a_pending_write_without_a_prompt(fake, run_app, stack):
    llm = ScriptedLLM([say("", call("create_ticket", **NEW)), say("Shall I?"), say("Created.")])
    http = await run_app(llm, stack)
    await http.post("/invocations", json={"prompt": "create it"}, headers={SESSION: "s" * 40})
    response = await http.post(
        "/invocations", json={"approval": "accept"}, headers={SESSION: "s" * 40}
    )
    body = response.json()
    assert response.status_code == 200 and body["pending_approval"] is None
    assert [c["name"] for c in body["tool_calls"]] == ["create_ticket"]


async def test_sessions_are_routed_by_the_header(run_app, stack):
    llm = ScriptedLLM(
        [
            say("", call("create_ticket", **NEW)),
            say("A?"),
            say("hello B"),
            say("UNRELATED"),
            say("hello A again"),
        ]  # A's "hi again" goes to the classifier first
    )
    http = await run_app(llm, stack)
    a = {SESSION: "a" * 40}
    b = {SESSION: "b" * 40}
    await http.post("/invocations", json={"prompt": "create"}, headers=a)
    other = (await http.post("/invocations", json={"prompt": "hi"}, headers=b)).json()
    again = (await http.post("/invocations", json={"prompt": "hi again"}, headers=a)).json()
    assert other["pending_approval"] is None  # B never sees A's pending action
    assert again["pending_approval"]["action"] == "create_ticket"  # A's survives across requests


async def test_a_missing_session_header_uses_the_local_default_session(run_app, stack):
    llm = ScriptedLLM(
        [say("", call("create_ticket", **NEW)), say("?"), say("UNRELATED"), say("still here")]
    )
    http = await run_app(llm, stack)
    await http.post("/invocations", json={"prompt": "create"})
    again = (await http.post("/invocations", json={"prompt": "hello"})).json()
    assert again["pending_approval"] is not None


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"prompt": ""},
        {"prompt": 5},
        {"other": "x"},
        {"approval": "maybe"},
        {"prompt": "yes", "approval": "accept"},  # one or the other, never both
    ],
)
async def test_a_malformed_body_is_a_4xx(run_app, stack, payload):
    http = await run_app(ScriptedLLM([]), stack)
    assert (await http.post("/invocations", json=payload)).status_code == 422


async def test_invalid_json_is_a_4xx(run_app, stack):
    http = await run_app(ScriptedLLM([]), stack)
    headers = {"Content-Type": "application/json"}
    response = await http.post("/invocations", content=b"{nope", headers=headers)
    assert response.status_code == 422


async def test_a_rejected_body_still_gets_a_request_id_and_a_log_line(run_app, stack):
    http = await run_app(ScriptedLLM([]), stack)
    with capture_logs() as logs:
        response = await http.post("/invocations", json={}, headers={SESSION: "s" * 40})
    request_id = response.headers["X-Request-Id"]
    assert response.status_code == 422 and response.json()["request_id"] == request_id
    line = next(log for log in logs if log["event"] == "invalid_request")
    assert (line["request_id"], line["session_id"]) == (request_id, "s" * 40)


def json_lines(text: str) -> list[dict]:
    return [json.loads(line) for line in text.splitlines() if line.startswith("{")]


async def test_tool_calls_and_approvals_are_logged_with_the_request_id(
    fake, run_app, stack, capsys
):
    configure_logging("INFO")
    llm = ScriptedLLM([say("", call("create_ticket", **NEW)), say("Shall I?"), say("Created.")])
    http = await run_app(llm, stack)
    headers = {SESSION: "s" * 40}
    await http.post("/invocations", json={"prompt": "create"}, headers=headers)
    second = await http.post("/invocations", json={"prompt": "yes"}, headers=headers)
    lines = json_lines(capsys.readouterr().out)
    request_id = second.headers["X-Request-Id"]
    decision = next(line for line in lines if line["event"] == "approval_decision")
    assert decision["request_id"] == request_id and decision["input"] == NEW
    assert (decision["answer"], decision["decided_by"]) == ("CONFIRM", "exact_match")
    tool = next(line for line in lines if line["event"] == "tool_call")
    assert tool["request_id"] == request_id and tool["name"] == "create_ticket"
    assert (tool["status"], tool["attempts"], tool["requests"]) == ("ok", 1, 2)


async def test_a_write_is_logged_even_if_the_turn_fails_afterwards(fake, run_app, stack, capsys):
    configure_logging("INFO")
    llm = ScriptedLLM(
        [say("", call("create_ticket", **NEW)), say("Shall I?"), RuntimeError("bug after it")]
    )
    http = await run_app(llm, stack)
    headers = {SESSION: "s" * 40}
    await http.post("/invocations", json={"prompt": "create"}, headers=headers)
    failed = await http.post("/invocations", json={"prompt": "yes"}, headers=headers)
    assert failed.status_code == 500 and len(fake.tickets) == 1
    events = [line["event"] for line in json_lines(capsys.readouterr().out)]
    assert events.index("tool_call") < events.index("invocation_failed")


async def test_ticketing_retries_are_logged_with_the_request_id(fake, run_app, stack, capsys):
    configure_logging("INFO")
    fake.add(id="TCK-1002")
    fake.fail("get", Fail(status=500))
    llm = ScriptedLLM([say("", call("get_ticket", ticket_id="TCK-1002")), say("ok")])
    http = await run_app(llm, stack)
    response = await http.post("/invocations", json={"prompt": "status?"})
    lines = json_lines(capsys.readouterr().out)
    retry = next(line for line in lines if line["event"] == "ticketing_retry")
    assert retry["request_id"] == response.headers["X-Request-Id"]
    assert (retry["method"], retry["reason"], retry["attempt"]) == ("GET", "500", 1)


async def test_the_llm_being_down_is_a_safe_503(run_app, stack):
    class Down:
        async def complete(self, messages, tools):
            raise LLMError("every model failed: internal detail")

    http = await run_app(Down(), stack)
    response = await http.post("/invocations", json={"prompt": "hi"})
    assert response.status_code == 503
    assert "internal detail" not in response.text and "language model" in response.json()["message"]


async def test_an_unexpected_error_is_a_safe_500_and_the_traceback_goes_to_the_log(run_app, stack):
    class Broken:
        async def complete(self, messages, tools):
            raise RuntimeError("secret internals")

    http = await run_app(Broken(), stack)
    with capture_logs() as logs:
        response = await http.post("/invocations", json={"prompt": "hi"})
    assert response.status_code == 500
    assert "secret internals" not in response.text and response.json()["request_id"]
    error = next(log for log in logs if log["event"] == "invocation_failed")
    assert error["exc_info"] and error["request_id"] == response.json()["request_id"]


class Probe:
    """An LLM that records how many requests are inside it at once."""

    def __init__(self) -> None:
        self.active = 0
        self.max_active = 0

    async def complete(self, messages, tools):
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        await asyncio.sleep(0.02)
        self.active -= 1
        return say("ok")


async def test_two_concurrent_requests_on_one_session_are_serialized(run_app, stack):
    probe = Probe()
    http = await run_app(probe, stack)
    same_session = {SESSION: "s" * 40}
    await asyncio.gather(
        *[http.post("/invocations", json={"prompt": "hi"}, headers=same_session) for _ in range(2)]
    )
    assert probe.max_active == 1


async def test_requests_on_different_sessions_run_in_parallel(run_app, stack):
    probe = Probe()
    http = await run_app(probe, stack)
    sessions = [{SESSION: "1" * 40}, {SESSION: "2" * 40}]
    await asyncio.gather(
        *[http.post("/invocations", json={"prompt": "hi"}, headers=h) for h in sessions]
    )
    assert probe.max_active == 2


async def test_one_log_line_per_invocation_with_what_is_needed_to_follow_it(fake, run_app, stack):
    fake.add(id="TCK-1002")
    llm = ScriptedLLM(
        [say("", call("get_ticket", ticket_id="TCK-1002"), tokens=10), say("done", tokens=5),
         say("", call("create_ticket", **NEW)), say("Shall I?")]
    )  # fmt: skip
    http = await run_app(llm, stack)
    headers = {SESSION: "s" * 40}
    with capture_logs() as logs:
        first = await http.post("/invocations", json={"prompt": "status?"}, headers=headers)
        second = await http.post("/invocations", json={"prompt": "create"}, headers=headers)
    lines = [log for log in logs if log["event"] == "invocation"]
    assert len(lines) == 2
    one, two = lines
    assert one["request_id"] == first.headers["X-Request-Id"] and one["session_id"] == "s" * 40
    assert one["tool_calls"] == [
        {"name": "get_ticket", "status": "ok", "attempts": 1, "duration_ms": 0}
    ]
    assert one["token_usage"] == {"input_tokens": 15, "output_tokens": 0}
    assert one["pending_action"] is None
    assert isinstance(one["latency_ms"], int)
    assert two["request_id"] == second.headers["X-Request-Id"]
    assert two["tool_calls"] == [] and two["pending_action"] == "create_ticket"


async def test_secrets_never_reach_the_logs(fake, run_app, stack, capsys):
    configure_logging("DEBUG")
    http = await run_app(ScriptedLLM([say("hello")]), stack)
    configure_logging("DEBUG")  # the lifespan reconfigured it; make sure stdout rendering is on
    await http.post("/invocations", json={"prompt": "hi"})
    out = capsys.readouterr().out
    assert "invocation" in out and SECRET not in out and "api_key" not in out


def test_logs_are_one_json_object_per_line(capsys):
    configure_logging("INFO")
    structlog.get_logger().info("hello", answer=42)
    line = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert line["event"] == "hello" and line["answer"] == 42
    assert line["level"] == "info" and line["timestamp"].endswith("Z")


def test_the_log_level_is_respected(capsys):
    configure_logging("WARNING")
    structlog.get_logger().info("quiet")
    structlog.get_logger().warning("loud")
    out = capsys.readouterr().out
    assert "loud" in out and "quiet" not in out
