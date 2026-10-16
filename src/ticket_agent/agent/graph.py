"""The agent as a LangGraph graph, with the approval step.

    START -> agent --tool calls--> tools -> agent
               |--no tool calls, a write waits--> wait_for_reply  (interrupt)
               |                                     |--CONFIRM--> execute -> agent
               |                                     `--otherwise----------> agent
               `--no tool calls, nothing waits--> END

A write never runs when the LLM calls it: `tools` saves it as pending, and the turn pauses in
`wait_for_reply` with interrupt(). The user's next message resumes it.

Two rules keep interrupt() safe. LangGraph stores a resume value on the paused checkpoint, and a
run that fails after resuming would replay that stored value next time (tested):
1. Code decides what the user's answer means *before* resuming (run_turn). If that fails, the
   graph was never touched.
2. A resumed turn never raises on an LLM failure: it ends with a fixed reply. Anything else that
   fails (a bug) moves the session to a fresh thread seeded with the last good messages.
"""

import json
import time
from collections.abc import Callable
from typing import Any, Literal

import structlog
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt
from mcp import Client
from mcp.types import CallToolResult, Tool
from pydantic import BaseModel

from ticket_agent.agent import approval
from ticket_agent.agent.llm import LLM, LLMError, LLMResponse, Message, ToolCall, ToolSpec, Usage
from ticket_agent.agent.prompts import build_system_prompt
from ticket_agent.agent.state import (
    CHECKPOINT_TYPES,
    AgentState,
    ApprovalEvent,
    LlmEvent,
    PendingAction,
    Reply,
    SessionStore,
    ToolCallRecord,
    ToolEvent,
    TraceEvent,
    Turn,
)

logger = structlog.get_logger()

LLM_DOWN = "I could not finish my reply: the language model is not available right now."
NOTHING_WAITING = "Nothing is waiting for your approval."
ONLY_FIRST = "I only saved the first change; the others were not saved."
CANCELLED = "Cancelled. Nothing was changed."


class ToolInfo(BaseModel):
    name: str
    read_only: bool  # False: the agent asks the user before it runs


class TurnResult(BaseModel):
    output: str
    tool_calls: list[ToolCallRecord]
    pending_approval: PendingAction | None
    usage: Usage
    trace: list[TraceEvent]


class Agent:
    def __init__(
        self,
        llm: LLM,
        mcp: Client,
        tools: list[Tool],
        store: SessionStore,
        *,
        max_steps: int = 6,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._llm = llm
        self._mcp = mcp
        self._store = store
        self._max_steps = max_steps
        self._clock = clock
        self._tools = {tool.name: tool for tool in tools}
        # Fail closed: only a tool that says it is read-only runs without approval.
        self._read_only = {
            tool.name for tool in tools if tool.annotations and tool.annotations.read_only_hint
        }
        self._specs = [
            ToolSpec(name=t.name, description=t.description or "", input_schema=t.input_schema)
            for t in tools
        ]
        self._graph = self._build_graph()

    def tool_info(self) -> list[ToolInfo]:
        return [ToolInfo(name=name, read_only=name in self._read_only) for name in self._tools]

    @classmethod
    async def create(cls, llm: LLM, mcp: Client, store: SessionStore, **options: Any) -> "Agent":
        """The MCP tools are listed once, at startup."""
        return cls(llm, mcp, (await mcp.list_tools()).tools, store, **options)

    def _build_graph(self):
        graph = StateGraph(AgentState)
        graph.add_node("agent", self._agent)
        graph.add_node("tools", self._run_tools)
        graph.add_node("wait_for_reply", self._wait_for_reply)
        graph.add_node("execute", self._execute)
        graph.add_edge(START, "agent")
        graph.add_conditional_edges("agent", _after_agent, ["tools", "wait_for_reply", END])
        graph.add_edge("tools", "agent")
        graph.add_conditional_edges("wait_for_reply", _after_reply, ["execute", "agent"])
        graph.add_edge("execute", "agent")
        serde = JsonPlusSerializer(allowed_msgpack_modules=CHECKPOINT_TYPES)
        self._checkpointer = InMemorySaver(serde=serde)
        return graph.compile(checkpointer=self._checkpointer)

    async def run_turn(
        self, session_id: str, prompt: str, approval: Literal["accept", "decline"] | None = None
    ) -> TurnResult:
        """`approval` is the UI's Accept or Decline button: a decision, with no text to read."""
        async with self._store.lock(session_id):  # one turn at a time per session
            config = self._store.thread(session_id)
            paused = await self._graph.aget_state(config)
            if paused.interrupts:  # a write waits: this message is the user's answer
                pending = PendingAction.model_validate(paused.values["pending"])
                if approval:
                    answer = "CONFIRM" if approval == "accept" else "DECLINE"
                    reply = Reply(text="", answer=answer, decided_by="button")
                else:
                    reply = await self._decide(pending, prompt)  # may raise: graph untouched
                # The audit trail: what the user was asked to approve and how the answer was read.
                logger.info(
                    "approval_decision",
                    action=pending.action, input=pending.input,
                    answer=reply.answer, decided_by=reply.decided_by,
                )  # fmt: skip
                payload: Any = Command(resume=reply.model_dump())
            elif approval:  # a button for something already answered or gone
                return TurnResult(
                    output=NOTHING_WAITING, tool_calls=[], pending_approval=None,
                    usage=Usage(), trace=[],
                )  # fmt: skip
            else:
                user = Message(role="user", content=prompt)
                payload = {"messages": [user], "turn": Turn(prompt=prompt)}
            try:
                await self._graph.ainvoke(
                    payload, config | {"recursion_limit": 25 + 4 * self._max_steps}
                )
            except Exception:
                await self._restart_from_last_good(session_id)
                raise
            snapshot = await self._graph.aget_state(config)
            self._store.remember(session_id, snapshot.config)
            return self._result(AgentState.model_validate(snapshot.values))

    async def _decide(self, pending: PendingAction, text: str) -> Reply:
        """What the user's answer means: an exact phrase, else one classifier call."""
        answer = approval.match_exact(text)
        if answer is not None:
            return Reply(text=text, answer=answer, decided_by="exact_match")
        turn = Turn()  # collects the classifier's trace event and usage
        response = await self._call_llm(
            turn, "classifier", approval.classifier_messages(pending, text), []
        )
        return Reply(
            text=text,
            answer=approval.parse_label(response.text),
            decided_by="classifier",
            trace=turn.trace,
            usage=turn.usage,
        )

    async def _restart_from_last_good(self, session_id: str) -> None:
        """A failed turn: continue on a fresh thread with the last good messages, nothing
        pending. The failed thread, with any stored resume value, is never used again."""
        last_good = self._store.last_good(session_id)
        config = self._store.new_thread(session_id)
        if last_good is not None:
            messages = (await self._graph.aget_state(last_good)).values.get("messages", [])
            await self._graph.aupdate_state(config, {"messages": messages}, as_node="agent")

    # --- nodes ---

    async def _agent(self, state: AgentState) -> dict[str, Any]:
        turn = state.turn.model_copy(deep=True)
        if turn.steps >= self._max_steps:
            return _reply(turn, f"I stopped after {self._max_steps} steps without finishing.")
        turn.steps += 1
        messages = [Message(role="system", content=build_system_prompt(state.pending))]
        try:
            response = await self._call_llm(turn, "agent", messages + state.messages, self._specs)
        except LLMError:
            if not (turn.records or turn.staged or turn.answer):
                raise  # an ordinary turn where nothing happened yet: a clean 503
            # A resumed turn must not raise (its stored answer would be replayed), and a turn
            # where something happened must still report it.
            return _reply(turn, LLM_DOWN)
        turn.reply = response.text
        message = Message(role="assistant", content=response.text, tool_calls=response.tool_calls)
        return {"messages": [message], "turn": turn}

    async def _run_tools(self, state: AgentState) -> dict[str, Any]:
        turn = state.turn.model_copy(deep=True)
        pending = state.pending
        results = []
        for call in state.messages[-1].tool_calls:
            if call.name not in self._tools:
                text = f"There is no tool called {call.name}."
            elif call.name in self._read_only:
                text, _ = await self._call_tool(turn, call.name, call.input)
            else:
                text, pending = self._stage(turn, pending, call)
            results.append(Message(role="tool", content=text, tool_call_id=call.id))
        return {"messages": results, "turn": turn, "pending": pending}

    async def _wait_for_reply(self, state: AgentState) -> dict[str, Any]:
        pending = state.pending
        if pending is None:  # unreachable: routed here only while a write waits
            raise RuntimeError("wait_for_reply ran with nothing pending")
        # The turn pauses here. On resume LangGraph re-runs this node and interrupt() returns
        # the Reply that run_turn decided, so nothing above this line may have side effects.
        reply = Reply.model_validate(interrupt(pending.model_dump()))
        approved = ApprovalEvent(
            action=pending.action, answer=reply.answer, decided_by=reply.decided_by
        )
        turn = Turn(
            prompt=reply.text,
            answer=reply.answer,
            trace=[*reply.trace, approved],
            usage=reply.usage,
        )
        if reply.answer == "CONFIRM":
            return {"turn": turn}  # execute adds the user's message with the real result
        described = pending.describe()
        notes = {
            "DECLINE": f"The user declined {described}. It was cancelled. Nothing changed.",
            "CHANGE": f"The user wants to change {described}. The proposal was withdrawn. Call "
            "the tool again with the new details.",
            "UNRELATED": f"{described} still waits for the user's approval. Answer the user.",
        }
        if reply.answer == "DECLINE":
            turn.notes.append(CANCELLED)
        return {
            "turn": turn,
            "pending": pending if reply.answer == "UNRELATED" else None,
            "messages": [_user_with_note(reply.text, notes[reply.answer])],
        }

    async def _execute(self, state: AgentState) -> dict[str, Any]:
        """Runs the stored input, exactly as the user saw it. Never the LLM's arguments."""
        turn = state.turn.model_copy(deep=True)
        pending = state.pending
        if pending is None:  # unreachable: routed here only on a confirm
            raise RuntimeError("execute ran with nothing pending")
        text, record = await self._call_tool(turn, pending.action, pending.input)
        confirmed = f"The user confirmed {pending.describe()}."
        if record.status in ("ok", "retried"):
            note = f"{confirmed} It ran. Result: {text}"
        elif record.status == "unknown":  # the write may have landed; nothing ruled it out
            note = f"{confirmed} It may or may not have gone through: {text}"
            turn.notes.append(
                f"That change may or may not have gone through: {text} "
                "Ask me to check before trying again."
            )
        else:  # rejected or never sent: nothing was written
            note = f"{confirmed} It failed and nothing was changed: {text}"
            turn.notes.append(f"That change was not made: {text}")
        # Cleared either way: a failed write is never retried blindly.
        return {"turn": turn, "pending": None, "messages": [_user_with_note(turn.prompt, note)]}

    # --- helpers ---

    def _stage(
        self, turn: Turn, pending: PendingAction | None, call: ToolCall
    ) -> tuple[str, PendingAction | None]:
        if turn.staged:
            turn.notes.append(ONLY_FIRST)
            return "Not saved: only one change can wait for approval at a time.", pending
        required = self._tools[call.name].input_schema.get("required", [])
        missing = [field for field in required if not call.input.get(field)]
        if missing:
            return f"Not saved: missing {', '.join(missing)}. Ask the user for it.", pending
        turn.staged = True
        turn.trace.append(ToolEvent(name=call.name, status="staged"))
        return "Saved, NOT executed. The user is now asked to approve it.", PendingAction(
            action=call.name, input=call.input
        )

    async def _call_llm(
        self,
        turn: Turn,
        purpose: Literal["agent", "classifier"],
        messages: list[Message],
        tools: list[ToolSpec],
    ) -> LLMResponse:
        start = self._clock()
        response = await self._llm.complete(messages, tools)
        turn.usage.input_tokens += response.usage.input_tokens
        turn.usage.output_tokens += response.usage.output_tokens
        turn.trace.append(
            LlmEvent(
                purpose=purpose,
                model=response.model,
                duration_ms=round((self._clock() - start) * 1000),
                input_tokens=response.usage.input_tokens,
                output_tokens=response.usage.output_tokens,
                attempts=response.attempts,
                errors=response.errors,
                fallback_from=response.fallback_from,
            )
        )
        return response

    async def _call_tool(
        self, turn: Turn, name: str, tool_input: dict[str, Any]
    ) -> tuple[str, ToolCallRecord]:
        result = await self._mcp.call_tool(name, tool_input)
        telemetry = (result.meta or {}).get("telemetry", {})  # kept here, never sent to the LLM
        text = _text_for_llm(result)
        record = ToolCallRecord(
            name=name,
            input=tool_input,
            status=telemetry.get("status", "error" if result.is_error else "ok"),
            duration_ms=telemetry.get("duration_ms", 0),
            attempts=telemetry.get("attempts", 0),
            requests=telemetry.get("requests", 0),
            retry_reasons=telemetry.get("retry_reasons", []),
            error=text if result.is_error else None,
        )
        turn.records.append(record)
        # Logged as it happens, so a write is on record even if the turn fails afterwards.
        logger.info("tool_call", **record.model_dump(exclude={"input"}))
        turn.trace.append(
            ToolEvent(
                name=name,
                status=record.status,
                duration_ms=record.duration_ms,
                attempts=record.attempts,
            )
        )
        return text, record

    def _result(self, state: AgentState) -> TurnResult:
        turn = state.turn
        parts = [turn.reply.strip(), *turn.notes]
        # Invariant 6 in code: a failed read is named even if the LLM's reply skips it.
        parts += [
            f"Note: {record.name} did not work: {record.error}"
            for record in turn.records
            if record.status == "error" and record.name in self._read_only
        ]
        output = "\n".join(part for part in parts if part)
        if state.pending:  # its own paragraph, so it stands out from the reply
            reminder = (
                f"Waiting for your approval: {state.pending.summary()}. "
                'Use Accept or Decline, or reply "yes" to go ahead.'
            )
            output = f"{output}\n\n{reminder}" if output else reminder
        return TurnResult(
            output=output,
            tool_calls=turn.records,
            pending_approval=state.pending,
            usage=turn.usage,
            trace=turn.trace,
        )


def _after_agent(state: AgentState) -> str:
    if state.messages[-1].tool_calls:
        return "tools"
    return "wait_for_reply" if state.pending else END


def _after_reply(state: AgentState) -> str:
    return "execute" if state.turn.answer == "CONFIRM" else "agent"


def _reply(turn: Turn, text: str) -> dict[str, Any]:
    turn.reply = text
    return {"messages": [Message(role="assistant", content=text)], "turn": turn}


def _user_with_note(prompt: str, note: str) -> Message:
    # A button sends no text: then the note is the whole message.
    text = f"{prompt}\n\n[System note: {note}]" if prompt else f"[System note: {note}]"
    return Message(role="user", content=text)


def _text_for_llm(result: CallToolResult) -> str:
    """Only the text and the data: _meta (telemetry) is dropped here."""
    texts = [block.text for block in result.content if hasattr(block, "text")]
    if result.structured_content is not None:
        texts.append(json.dumps(result.structured_content))
    return "\n".join(texts)
