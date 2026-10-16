"""The LLM interface the agent talks to, the OpenAI-protocol client, and ScriptedLLM for tests.

All LLM access goes through here, so the provider can be swapped or faked in tests. Every model
is called through the official openai SDK and the Chat Completions protocol: OpenAI directly,
Gemini through its OpenAI-compatible endpoint. Another OpenAI-compatible endpoint (Amazon
Bedrock, a platform gateway) is OPENAI_BASE_URL, not a code change.
"""

import asyncio
import json
import random
from collections.abc import Awaitable, Callable
from typing import Any, Literal, Protocol

import openai
import structlog
from pydantic import BaseModel, Field, SecretStr

from ticket_agent.config import ModelChoice, Settings

logger = structlog.get_logger()

GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
# Gemini 3 rejects a tool call without a thought signature (HTTP 400). A call that another model
# made has none; this placeholder tells Gemini to skip the check (tested in the first build).
GEMINI_PLACEHOLDER = {"google": {"thought_signature": "skip_thought_signature_validator"}}

# Only these move on to a retry, then to the next model. Anything else is a configuration
# error (a wrong key, model or parameter) and must fail loudly, not hide behind a fallback.
UNAVAILABLE = (openai.APIConnectionError, openai.RateLimitError, openai.InternalServerError)
MAX_ATTEMPTS = 3  # per model
BACKOFF_BASE_S = 0.5
BACKOFF_CAP_S = 8.0
RETRY_AFTER_CAP_S = 10.0


class ToolSpec(BaseModel):
    name: str
    description: str
    input_schema: dict[str, Any]


class ToolCall(BaseModel):
    id: str
    name: str
    input: dict[str, Any] = Field(default_factory=dict)
    extra_content: dict[str, Any] | None = None  # Gemini's thought signature, sent back as-is


class Message(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: str = ""
    tool_calls: list[ToolCall] = Field(default_factory=list)  # assistant messages only
    tool_call_id: str | None = None  # tool messages only


class Usage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0


class LLMResponse(BaseModel):
    text: str = ""
    model: str = ""  # the model that answered, as provider/model
    tool_calls: list[ToolCall] = Field(default_factory=list)
    usage: Usage = Field(default_factory=Usage)
    attempts: int = 1  # requests sent, retries and fallbacks included
    errors: list[str] = Field(default_factory=list)  # one label per failed request, e.g. "429"
    fallback_from: str | None = None  # the first model in the chain, when another one answered


class LLM(Protocol):
    async def complete(self, messages: list[Message], tools: list[ToolSpec]) -> LLMResponse: ...


class LLMCall(BaseModel):
    """What ScriptedLLM was sent on one call."""

    messages: list[Message]
    tools: list[str]  # tool names offered


class ScriptedLLM:
    """Returns pre-programmed responses in order (or raises a scripted exception), and records
    what it was sent."""

    def __init__(self, responses: list[LLMResponse | Exception]) -> None:
        self._responses = list(responses)
        self.calls: list[LLMCall] = []

    async def complete(self, messages: list[Message], tools: list[ToolSpec]) -> LLMResponse:
        copied = [message.model_copy(deep=True) for message in messages]
        self.calls.append(LLMCall(messages=copied, tools=[tool.name for tool in tools]))
        if not self._responses:
            raise AssertionError("ScriptedLLM was asked for more responses than were scripted")
        step = self._responses.pop(0)
        if isinstance(step, Exception):
            raise step
        return step


class LLMError(Exception):
    """No model could answer. The message is for the logs, not the user."""


class LLMTarget(BaseModel):
    """One model in the chain, with its key and endpoint."""

    choice: ModelChoice
    api_key: SecretStr
    base_url: str | None = None  # None: the SDK default (OPENAI_BASE_URL, else api.openai.com)


Create = Callable[[LLMTarget, dict[str, Any]], Awaitable[Any]]


class OpenAIChatClient:
    """Tries each model in order. A model is retried while it is unavailable, then the next one
    is used. Our loop owns the retries (the SDK's are off), so each one shows in the trace."""

    def __init__(
        self,
        targets: list[LLMTarget],
        *,
        timeout_s: float = 60.0,
        create: Create | None = None,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
        rng: random.Random | None = None,
    ) -> None:
        self.targets = targets
        self._timeout_s = timeout_s
        self._create = create or self._sdk_create
        self._sleep = sleep
        self._rng = rng or random.Random()  # noqa: S311 - backoff jitter, not cryptography
        self._clients: dict[str, openai.AsyncOpenAI] = {}

    @classmethod
    def from_settings(cls, settings: Settings) -> "OpenAIChatClient":
        if settings.main_model is None:
            raise ValueError("LLM_PROVIDER and LLM_MODEL are not set; see .env.example")
        chain = [settings.main_model, *settings.llm_fallback_models]
        return cls(
            [_target(settings, choice) for choice in chain], timeout_s=settings.llm.timeout_s
        )

    async def complete(self, messages: list[Message], tools: list[ToolSpec]) -> LLMResponse:
        errors: list[str] = []
        for index, target in enumerate(self.targets):
            request = _request(target, messages, tools)
            for attempt in range(1, MAX_ATTEMPTS + 1):
                try:
                    reply = await self._create(target, request)
                except UNAVAILABLE as exc:
                    errors.append(_label(exc))
                    if attempt == MAX_ATTEMPTS:
                        break
                    logger.warning("llm_retry", model=target.choice.label, error=errors[-1])
                    await self._sleep(self._delay(attempt, exc))
                    continue
                except openai.APIError as exc:
                    logger.error(
                        "llm_config_error",
                        model=target.choice.label,
                        error=_label(exc),
                        detail=self._redact(str(exc)),
                    )
                    raise LLMError(f"{target.choice.label}: {_label(exc)}") from None
                return _parse(reply).model_copy(
                    update={
                        "model": target.choice.label,
                        "attempts": len(errors) + 1,
                        "errors": errors,
                        "fallback_from": self.targets[0].choice.label if index else None,
                    }
                )
            logger.warning("llm_fallback", failed_model=target.choice.label, errors=errors)
        raise LLMError(f"every model failed: {[t.choice.label for t in self.targets]}")

    def _delay(self, attempt: int, exc: Exception) -> float:
        retry_after = getattr(getattr(exc, "response", None), "headers", {}).get("retry-after")
        try:
            return min(float(retry_after), RETRY_AFTER_CAP_S)
        except (TypeError, ValueError):  # no header, or an HTTP date
            cap = min(BACKOFF_CAP_S, BACKOFF_BASE_S * 2 ** (attempt - 1))
            return self._rng.uniform(0, cap)  # full jitter

    async def _sdk_create(self, target: LLMTarget, request: dict[str, Any]) -> Any:
        key = f"{target.base_url}|{target.choice.provider}"
        if key not in self._clients:
            self._clients[key] = openai.AsyncOpenAI(
                api_key=target.api_key.get_secret_value(),
                base_url=target.base_url,  # None: the SDK reads OPENAI_BASE_URL itself
                timeout=self._timeout_s,
                max_retries=0,  # retries happen in complete(), where they are recorded
            )
        return await self._clients[key].chat.completions.create(**request)

    def _redact(self, text: str) -> str:
        """Error text can repeat a key: never log one."""
        for target in self.targets:
            text = text.replace(target.api_key.get_secret_value(), "***")
        return text[:300]


def _target(settings: Settings, choice: ModelChoice) -> LLMTarget:
    variable = f"{choice.provider.upper()}_API_KEY"
    key: SecretStr | None = getattr(settings, f"{choice.provider}_api_key")
    if key is None or not key.get_secret_value().strip():
        raise ValueError(f"{variable} is not set: add it to .env for {choice.label}")
    base_url = GEMINI_BASE_URL if choice.provider == "gemini" else None
    return LLMTarget(choice=choice, api_key=key, base_url=base_url)


def _request(target: LLMTarget, messages: list[Message], tools: list[ToolSpec]) -> dict[str, Any]:
    gemini = target.choice.provider == "gemini"
    request: dict[str, Any] = {
        "model": target.choice.model,
        "messages": [_to_chat(message, gemini=gemini) for message in messages],
    }
    if tools:
        request["tools"] = [_tool_to_chat(tool) for tool in tools]
    if target.choice.reasoning:
        request["reasoning_effort"] = target.choice.reasoning
    return request


def _to_chat(message: Message, *, gemini: bool) -> dict[str, Any]:
    if message.role == "tool":
        return {"role": "tool", "tool_call_id": message.tool_call_id, "content": message.content}
    converted: dict[str, Any] = {"role": message.role, "content": message.content}
    if message.tool_calls:
        converted["tool_calls"] = [
            _call_to_chat(call, gemini=gemini) for call in message.tool_calls
        ]
    return converted


def _call_to_chat(call: ToolCall, *, gemini: bool) -> dict[str, Any]:
    converted: dict[str, Any] = {
        "id": call.id,
        "type": "function",
        "function": {"name": call.name, "arguments": json.dumps(call.input)},
    }
    if gemini:  # Gemini needs its signature back; other models' calls get the placeholder
        converted["extra_content"] = call.extra_content or GEMINI_PLACEHOLDER
    return converted


def _tool_to_chat(tool: ToolSpec) -> dict[str, Any]:
    function = {"name": tool.name, "description": tool.description, "parameters": tool.input_schema}
    return {"type": "function", "function": function}


def _parse(reply: Any) -> LLMResponse:
    message = reply.choices[0].message
    calls = [
        ToolCall(
            id=call.id,
            name=call.function.name,
            input=_arguments(call.function.arguments),
            extra_content=(call.model_extra or {}).get("extra_content"),
        )
        for call in message.tool_calls or []
    ]
    usage = Usage(
        input_tokens=getattr(reply.usage, "prompt_tokens", 0) or 0,
        output_tokens=getattr(reply.usage, "completion_tokens", 0) or 0,
    )
    return LLMResponse(text=message.content or "", tool_calls=calls, usage=usage)


def _arguments(raw: str | None) -> dict[str, Any]:
    # A model can emit broken JSON. Empty input makes the tool report what is missing.
    try:
        parsed = json.loads(raw or "{}")
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _label(exc: Exception) -> str:
    status = getattr(exc, "status_code", None)
    return str(status) if status else type(exc).__name__
