"""The graph state (what the LangGraph checkpointer keeps per session) and the session store."""

import asyncio
import json
import operator
from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field

from ticket_agent.agent.llm import Message, ToolCall, Usage

Answer = Literal["CONFIRM", "DECLINE", "CHANGE", "UNRELATED"]


class PendingAction(BaseModel):
    """A write the user has not confirmed yet. `input` is exactly what the user was shown."""

    action: str
    input: dict[str, Any]

    def describe(self) -> str:
        """Exact, for the LLM: the action and its JSON input."""
        return f"{self.action} {json.dumps(self.input)}"

    def summary(self) -> str:
        """Plain language, for the user."""
        i = self.input
        if self.action == "create_ticket":
            details = ", ".join(f"{key} {i[key]}" for key in ("priority", "category") if i.get(key))
            text = f'create a ticket "{i.get("title")}" for {i.get("requester")}'
            return f"{text} ({details})" if details else text
        if self.action == "update_ticket":
            changes = [f"{k} to {v}" for k, v in i.items() if k != "ticket_id" and v is not None]
            return f"update {i.get('ticket_id')}: {', '.join(changes)}"
        if self.action == "add_comment":
            return f'add a comment to {i.get("ticket_id")} as {i.get("author")}: "{i.get("body")}"'
        return self.describe()


class ToolCallRecord(BaseModel):
    """One tool that actually ran this turn (CONTRACT.md tool_calls[], plus attempts)."""

    name: str
    input: dict[str, Any]
    status: str  # ok, retried, error, or unknown: a write that failed but may have landed
    duration_ms: int = 0
    attempts: int = 0  # 1 plus the retries of the tool's own requests
    requests: int = 0  # every HTTP request sent, the duplicate-check reads included
    retry_reasons: list[str] = Field(default_factory=list)
    error: str | None = None  # the safe message, when status is "error"


class LlmEvent(BaseModel):
    """One LLM call, for the trace in the response and the UI."""

    type: Literal["llm"] = "llm"
    purpose: Literal["agent", "classifier"]
    model: str
    duration_ms: int
    input_tokens: int = 0
    output_tokens: int = 0
    attempts: int = 1  # requests sent, retries and fallbacks included
    errors: list[str] = Field(default_factory=list)  # e.g. ["429", "503"]
    fallback_from: str | None = None  # set when a fallback model answered


class ToolEvent(BaseModel):
    """One tool call the LLM asked for, in the order it happened."""

    type: Literal["tool"] = "tool"
    name: str
    status: str  # ok, retried, error, unknown, or staged: saved for approval, not run
    duration_ms: int = 0
    attempts: int = 0


class ApprovalEvent(BaseModel):
    """How the user's answer to a pending action was read (CLAUDE.md invariant 3)."""

    type: Literal["approval"] = "approval"
    action: str
    answer: Answer
    decided_by: Literal["button", "exact_match", "classifier"]


TraceEvent = Annotated[LlmEvent | ToolEvent | ApprovalEvent, Field(discriminator="type")]


class Reply(BaseModel):
    """The user's answer to a pending action, decided before the graph resumes."""

    text: str
    answer: Answer
    decided_by: Literal["button", "exact_match", "classifier"]
    trace: list[TraceEvent] = Field(default_factory=list)  # the classifier's LLM call, if any
    usage: Usage = Field(default_factory=Usage)


class Turn(BaseModel):
    """Bookkeeping for the current turn. Replaced by a fresh one when a turn starts."""

    steps: int = 0  # agent LLM calls so far
    staged: bool = False  # a write was staged this turn
    prompt: str = ""  # the user's message that started this turn
    answer: Answer | None = None  # what that answer meant
    reply: str = ""  # the agent's latest text
    records: list[ToolCallRecord] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)  # fixed sentences added to the reply
    trace: list[TraceEvent] = Field(default_factory=list)
    usage: Usage = Field(default_factory=Usage)


class AgentState(BaseModel):
    # operator.add: a node returns only its new messages; LangGraph appends them.
    messages: Annotated[list[Message], operator.add] = Field(default_factory=list)
    pending: PendingAction | None = None
    turn: Turn = Field(default_factory=Turn)


# The checkpointer deserialises only these types (LangGraph's msgpack allow-list).
CHECKPOINT_TYPES = (
    Message, ToolCall, Usage, PendingAction, ToolCallRecord, LlmEvent, ToolEvent, ApprovalEvent,
    Turn,
)  # fmt: skip


class SessionStore:
    """Per session: one lock (one turn at a time), the LangGraph thread, and its last good state.

    The conversation itself lives in the LangGraph checkpointer. A failed turn moves the session
    to a fresh thread seeded with the last good messages, so nothing from that turn, including a
    stored resume value, can ever be replayed.
    """

    def __init__(self) -> None:
        self._locks: dict[str, asyncio.Lock] = {}
        self._generation: dict[str, int] = {}
        self._last_good: dict[str, dict[str, Any]] = {}

    def lock(self, session_id: str) -> asyncio.Lock:
        return self._locks.setdefault(session_id, asyncio.Lock())

    def thread(self, session_id: str) -> dict[str, Any]:
        generation = self._generation.get(session_id, 0)
        return {"configurable": {"thread_id": f"{session_id}#{generation}"}}

    def new_thread(self, session_id: str) -> dict[str, Any]:
        self._generation[session_id] = self._generation.get(session_id, 0) + 1
        return self.thread(session_id)

    def last_good(self, session_id: str) -> dict[str, Any] | None:
        return self._last_good.get(session_id)

    def remember(self, session_id: str, config: dict[str, Any]) -> None:
        self._last_good[session_id] = config
