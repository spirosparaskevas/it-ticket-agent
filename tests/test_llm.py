"""OpenAIChatClient with an injected create function: no network, no real model."""

import random

import httpx2
import openai
import pytest
from openai.types.chat import ChatCompletion
from pydantic import SecretStr
from structlog.testing import capture_logs

from ticket_agent.agent.llm import (
    GEMINI_BASE_URL,
    GEMINI_PLACEHOLDER,
    LLMError,
    LLMTarget,
    Message,
    OpenAIChatClient,
    ToolCall,
    ToolSpec,
)
from ticket_agent.config import ModelChoice, Settings

pytestmark = pytest.mark.anyio

KEY = "sk-not-a-real-key"
PRIMARY = LLMTarget(choice=ModelChoice.parse("openai/gpt-6-luna@medium"), api_key=SecretStr(KEY))
NO_REASONING = LLMTarget(choice=ModelChoice.parse("openai/gpt-4o-mini"), api_key=SecretStr(KEY))
GEMINI = LLMTarget(
    choice=ModelChoice.parse("gemini/gemini-3.5-flash@low"),
    api_key=SecretStr("g-not-a-real-key"),
    base_url=GEMINI_BASE_URL,
)
TOOLS = [
    ToolSpec(name="get_ticket", description="Get one ticket.", input_schema={"type": "object"})
]
REQUEST = httpx2.Request("POST", "https://llm.test/v1/chat/completions")
SIGNATURE = {"google": {"thought_signature": "sig-from-gemini"}}


def reply(text: str | None = "", tool_calls: list | None = None) -> ChatCompletion:
    message = {"role": "assistant", "content": text, "tool_calls": tool_calls}
    return ChatCompletion.model_validate(
        {
            "id": "r1", "object": "chat.completion", "created": 0, "model": "served-model",
            "choices": [{"index": 0, "finish_reason": "stop", "message": message}],
            "usage": {"prompt_tokens": 11, "completion_tokens": 5, "total_tokens": 16},
        }
    )  # fmt: skip


def tool_call(arguments: str = '{"ticket_id": "TCK-1002"}', **extra) -> dict:
    function = {"name": "get_ticket", "arguments": arguments}
    return {"id": "c1", "type": "function", "function": function, **extra}


def status_error(cls: type[openai.APIStatusError], code: int, message: str = "x", **headers):
    response = httpx2.Response(code, headers=headers, request=REQUEST)
    return cls(message, response=response, body=None)


class FakeCreate:
    """Stands in for the SDK call: returns or raises the scripted outcomes in order."""

    def __init__(self, *outcomes: object) -> None:
        self.outcomes = list(outcomes)
        self.calls: list[tuple[str, dict]] = []

    async def __call__(self, target: LLMTarget, request: dict):
        self.calls.append((target.choice.label, request))
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


class FakeSleep:
    def __init__(self) -> None:
        self.sleeps: list[float] = []

    async def __call__(self, seconds: float) -> None:
        self.sleeps.append(seconds)


def client(*outcomes: object, targets=(PRIMARY,)):
    create, sleep = FakeCreate(*outcomes), FakeSleep()
    llm = OpenAIChatClient(
        list(targets),
        create=create,
        sleep=sleep,
        rng=random.Random(7),  # noqa: S311
    )
    return llm, create, sleep


USER = [Message(role="user", content="status of TCK-1002?")]


# --- request and reply ---


async def test_request_uses_the_chat_format_and_the_reasoning_setting():
    llm, create, _ = client(reply("hi"))
    history = [
        Message(role="system", content="be brief"),
        Message(role="user", content="status?"),
        Message(
            role="assistant", tool_calls=[ToolCall(id="c1", name="get_ticket", input={"a": 1})]
        ),
        Message(role="tool", content="open", tool_call_id="c1"),
    ]
    await llm.complete(history, TOOLS)
    _, sent = create.calls[0]
    assert sent["model"] == "gpt-6-luna" and sent["reasoning_effort"] == "medium"
    assert sent["messages"] == [
        {"role": "system", "content": "be brief"},
        {"role": "user", "content": "status?"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "c1",
                    "type": "function",
                    "function": {"name": "get_ticket", "arguments": '{"a": 1}'},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "c1", "content": "open"},
    ]
    function = {
        "name": "get_ticket",
        "description": "Get one ticket.",
        "parameters": {"type": "object"},
    }
    assert sent["tools"] == [{"type": "function", "function": function}]


async def test_no_reasoning_setting_sends_no_reasoning_effort_and_no_tools_sends_none():
    llm, create, _ = client(reply("hi"), targets=[NO_REASONING])
    await llm.complete(USER, [])
    _, sent = create.calls[0]
    assert "reasoning_effort" not in sent and "tools" not in sent


async def test_reply_text_tool_calls_usage_and_model_are_parsed():
    llm, _, _ = client(reply("Looking.", [tool_call()]))
    response = await llm.complete(USER, TOOLS)
    assert response.text == "Looking."
    assert response.tool_calls == [
        ToolCall(id="c1", name="get_ticket", input={"ticket_id": "TCK-1002"})
    ]
    assert (response.usage.input_tokens, response.usage.output_tokens) == (11, 5)
    assert (response.model, response.attempts, response.errors) == ("openai/gpt-6-luna", 1, [])
    assert response.fallback_from is None


async def test_unparseable_tool_arguments_become_empty_input():
    llm, _, _ = client(reply("", [tool_call("{not json")]))
    response = await llm.complete(USER, TOOLS)
    assert response.tool_calls[0].input == {}  # the tool then reports the missing arguments


# --- retries and fallback ---


async def test_429_waits_for_retry_after_then_succeeds():
    llm, _, sleep = client(
        status_error(openai.RateLimitError, 429, **{"retry-after": "2"}), reply("ok")
    )
    response = await llm.complete(USER, [])
    assert response.text == "ok" and sleep.sleeps == [2]
    assert (response.attempts, response.errors) == (2, ["429"])


async def test_server_errors_and_timeouts_back_off_and_are_recorded():
    llm, _, sleep = client(
        status_error(openai.InternalServerError, 500),
        openai.APITimeoutError(request=REQUEST),
        reply("ok"),
    )
    response = await llm.complete(USER, [])
    assert response.errors == ["500", "APITimeoutError"] and response.attempts == 3
    assert len(sleep.sleeps) == 2 and all(0 <= s <= 8 for s in sleep.sleeps)


async def test_after_three_failed_attempts_the_next_model_answers():
    down = status_error(openai.InternalServerError, 503)
    llm, create, _ = client(down, down, down, reply("from the fallback"), targets=[PRIMARY, GEMINI])
    with capture_logs() as logs:
        response = await llm.complete(USER, [])
    assert [label for label, _ in create.calls] == ["openai/gpt-6-luna"] * 3 + [
        "gemini/gemini-3.5-flash"
    ]
    assert response.text == "from the fallback" and response.model == "gemini/gemini-3.5-flash"
    assert response.fallback_from == "openai/gpt-6-luna" and response.errors == ["503"] * 3
    assert any(log["event"] == "llm_fallback" for log in logs)


@pytest.mark.parametrize(
    ("error", "code"),
    [
        (openai.BadRequestError, 400),
        (openai.AuthenticationError, 401),
        (openai.NotFoundError, 404),
    ],
)
async def test_a_configuration_error_fails_at_once_and_is_never_hidden_by_a_fallback(error, code):
    llm, create, _ = client(status_error(error, code), reply("never"), targets=[PRIMARY, GEMINI])
    with capture_logs() as logs, pytest.raises(LLMError):
        await llm.complete(USER, [])
    assert len(create.calls) == 1
    problem = next(log for log in logs if log["event"] == "llm_config_error")
    assert problem["model"] == "openai/gpt-6-luna" and problem["error"] == str(code)


async def test_every_model_failing_raises_llm_error():
    down = openai.APIConnectionError(request=REQUEST)
    llm, create, _ = client(*[down] * 6, targets=[PRIMARY, GEMINI])
    with pytest.raises(LLMError):
        await llm.complete(USER, [])
    assert len(create.calls) == 6


async def test_a_real_bug_is_not_swallowed_by_the_fallback():
    llm, create, _ = client(TypeError("bug"), reply("never"), targets=[PRIMARY, GEMINI])
    with pytest.raises(TypeError):
        await llm.complete(USER, [])
    assert len(create.calls) == 1


async def test_logs_never_show_a_key():
    failure = status_error(openai.AuthenticationError, 401, f"bad key {KEY}")
    llm, _, _ = client(failure)
    with capture_logs() as logs, pytest.raises(LLMError):
        await llm.complete(USER, [])
    assert KEY not in repr(logs) and "***" in repr(logs)


# --- Gemini thought signatures ---


async def test_a_gemini_signature_is_kept_from_the_reply():
    llm, _, _ = client(reply("", [tool_call(extra_content=SIGNATURE)]), targets=[GEMINI])
    response = await llm.complete(USER, TOOLS)
    assert response.tool_calls[0].extra_content == SIGNATURE


async def test_signatures_go_to_gemini_only_with_a_placeholder_for_other_models_calls():
    history = [
        *USER,
        Message(
            role="assistant",
            tool_calls=[
                ToolCall(id="g", name="get_ticket", extra_content=SIGNATURE),  # made by Gemini
                ToolCall(id="o", name="get_ticket"),  # made by an OpenAI model
            ],
        ),
        Message(role="tool", content="x", tool_call_id="g"),
        Message(role="tool", content="y", tool_call_id="o"),
    ]
    to_gemini, create_g, _ = client(reply("ok"), targets=[GEMINI])
    await to_gemini.complete(history, TOOLS)
    calls = create_g.calls[0][1]["messages"][1]["tool_calls"]
    assert [c["extra_content"] for c in calls] == [SIGNATURE, GEMINI_PLACEHOLDER]
    to_openai, create_o, _ = client(reply("ok"))
    await to_openai.complete(history, TOOLS)
    calls = create_o.calls[0][1]["messages"][1]["tool_calls"]
    assert all("extra_content" not in c for c in calls)  # OpenAI gets the plain format


# --- built from settings ---


def test_built_from_settings_with_keys_and_endpoints():
    settings = Settings(
        llm_provider="openai",
        llm_model="gpt-6-luna",
        llm_reasoning="medium",
        llm_fallback_models="openai/gpt-4o-mini, gemini/gemini-3.5-flash@low",
        openai_api_key="sk-o",
        gemini_api_key="g-k",
    )
    llm = OpenAIChatClient.from_settings(settings)
    assert [t.choice.label for t in llm.targets] == [
        "openai/gpt-6-luna", "openai/gpt-4o-mini", "gemini/gemini-3.5-flash",
    ]  # fmt: skip
    assert llm.targets[0].base_url is None  # the SDK default, or OPENAI_BASE_URL
    assert llm.targets[2].base_url == GEMINI_BASE_URL
    assert llm.targets[2].api_key.get_secret_value() == "g-k"


def test_the_fallback_is_optional_and_one_model_needs_only_its_own_key():
    llm = OpenAIChatClient.from_settings(
        Settings(llm_provider="openai", llm_model="gpt-6-luna", openai_api_key="sk-o")
    )  # no LLM_FALLBACK_MODELS, no GEMINI_API_KEY
    assert [t.choice.label for t in llm.targets] == ["openai/gpt-6-luna"]


async def test_without_a_fallback_an_unavailable_model_is_an_llm_error():
    down = status_error(openai.InternalServerError, 503)
    llm, create, _ = client(down, down, down)
    with pytest.raises(LLMError):
        await llm.complete(USER, [])
    assert len(create.calls) == 3  # retried, then no other model to try


def test_a_missing_model_or_key_fails_at_startup_and_names_the_variable():
    with pytest.raises(ValueError, match="LLM_PROVIDER and LLM_MODEL"):
        OpenAIChatClient.from_settings(Settings())
    with pytest.raises(ValueError, match="GEMINI_API_KEY"):
        OpenAIChatClient.from_settings(
            Settings(llm_provider="gemini", llm_model="gemini-3.5-flash")
        )
