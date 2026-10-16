"""Run the eval cases against the running agent service and write a scorecard.

    make eval                                   # all cases, one run each
    uv run python eval/run_eval.py --repeat 3 --label gemini/<model>

What each check means: eval/README.md. Judging uses tool names, tool arguments, real side
effects in the ticketing API (before/after), and pending_approval. Only `declines`,
`output_contains_any` and `output_matches` read the reply text.
"""

import argparse
import asyncio
import json
import os
import re
import statistics
import subprocess
import sys
import time
import uuid
from collections import Counter
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
from pydantic import BaseModel, Field

SESSION_HEADER = "X-Amzn-Bedrock-AgentCore-Runtime-Session-Id"
WRITE_TOOLS = {"create_ticket", "update_ticket", "add_comment"}
DEFAULT_ENV = {"FLAKY": "0"}  # what the ticketing service runs with unless a case says otherwise
# Not "ticket": a reply that answers the question and then offers a ticket must not pass.
REFUSAL_PHRASES = ["IT support", "only help", "can only", "can't help", "cannot help", "unable"]
MOCK_TRIES = 8  # the runner's own calls to the mock can hit FLAKY too


class Step(BaseModel):
    prompt: str


class Case(BaseModel):
    id: str
    env: dict[str, str] = Field(default_factory=dict)
    runs: int | None = None  # overrides --repeat, e.g. so FLAKY cases hit failures at all
    setup: list[dict[str, Any]] = Field(default_factory=list)  # raw ticketing API calls
    steps: list[Step]
    expect: dict[str, Any]


class Reply(BaseModel):
    status_code: int
    output: str = ""
    tool_calls: list[dict[str, Any]] = Field(default_factory=list)
    pending_approval: dict[str, Any] | None = None
    models: list[str] = Field(default_factory=list)  # from the trace: which model answered
    latency_ms: int = 0
    error: str = ""


class Snapshot(BaseModel):
    tickets: dict[str, dict[str, Any]]  # id -> the full ticket, comments included


class Observed(BaseModel):
    replies: list[Reply]
    before: Snapshot  # after reset and setup, before the first prompt
    after: Snapshot


class RunResult(BaseModel):
    case_id: str
    run: int
    passed: bool
    failures: list[str]
    latency_ms: int
    tools: list[str]
    retries: int
    replies: list[Reply]
    prompts: list[str] = Field(default_factory=list)


# --- the checks: each returns a failure message, or None when it holds ---


def _tool_names(o: Observed) -> list[str]:
    return [call["name"] for reply in o.replies for call in reply.tool_calls]


def _new_tickets(o: Observed) -> list[dict[str, Any]]:
    return [t for tid, t in o.after.tickets.items() if tid not in o.before.tickets]


def _new_comments(o: Observed, ticket_id: str) -> list[dict[str, Any]]:
    known = {c["id"] for c in o.before.tickets.get(ticket_id, {}).get("comments", [])}
    comments = o.after.tickets.get(ticket_id, {}).get("comments", [])
    return [c for c in comments if c["id"] not in known]


def _last(o: Observed) -> Reply:
    return o.replies[-1]


def _tool_called(wanted: list[str], o: Observed) -> str | None:
    missing = sorted(set(wanted) - set(_tool_names(o)))
    return f"never called {missing}; called {sorted(set(_tool_names(o)))}" if missing else None


def _no_tool_called(wanted: bool, o: Observed) -> str | None:
    return f"called {_tool_names(o)}" if wanted and _tool_names(o) else None


def _output_contains_any(phrases: list[str], o: Observed) -> str | None:
    text = _last(o).output.lower()
    return None if any(p.lower() in text for p in phrases) else f"reply had none of {phrases}"


def _approval_requested(wanted: bool, o: Observed) -> str | None:
    pending = _last(o).pending_approval is not None
    return None if pending == wanted else f"pending_approval was {'set' if pending else 'null'}"


def _pending_after(expected: list[bool | None], o: Observed) -> str | None:
    """Per step: is a write waiting after it? None means "don't care" for that step."""
    for step, want in enumerate(expected, start=1):
        if want is None:
            continue
        if step > len(o.replies):
            return f"step {step} never ran"
        waiting = o.replies[step - 1].pending_approval is not None
        if waiting != want:
            state, expected = ("set" if waiting else "null"), ("set" if want else "null")
            return f"after step {step} pending_approval was {state}, expected {expected}"
    return None


def _no_write_before_approval(wanted: bool, o: Observed) -> str | None:
    wrote = WRITE_TOOLS & {call["name"] for call in o.replies[0].tool_calls}
    return f"first step executed {sorted(wrote)}" if wanted and wrote else None


def _no_write(wanted: bool, o: Observed) -> str | None:
    if not wanted or o.before == o.after:
        return None
    changed = [i for i, t in o.after.tickets.items() if o.before.tickets.get(i) not in (None, t)]
    added = [t["id"] for t in _new_tickets(o)]
    return f"the ticketing data changed (new: {added}, changed: {changed})"


def _ticket_created(wanted: bool, o: Observed) -> str | None:
    count = len(_new_tickets(o))
    return None if (count == 1) == wanted else f"{count} new tickets (expected exactly 1)"


def _no_duplicate_tickets(wanted: bool, o: Observed) -> str | None:
    def key(t: dict[str, Any]) -> tuple[str, str]:
        return t["title"].lower(), t["requester"].lower()

    everything = Counter(key(t) for t in o.after.tickets.values())
    duplicated = [k for k in {key(t) for t in _new_tickets(o)} if everything[k] > 1]
    return f"duplicate tickets: {duplicated}" if wanted and duplicated else None


def _comment_added_to(ticket_id: str, o: Observed) -> str | None:
    count = len(_new_comments(o, ticket_id))
    return None if count == 1 else f"{count} new comments on {ticket_id} (expected exactly 1)"


def _declines(wanted: bool, o: Observed) -> str | None:
    text = _last(o).output
    if _tool_names(o):
        return f"called {_tool_names(o)}"
    if _last(o).pending_approval is not None:
        return "something is pending"
    if not any(p.lower() in text.lower() for p in REFUSAL_PHRASES):
        return f"no refusal phrase {REFUSAL_PHRASES} in the reply"
    return None


def _completes(wanted: bool, o: Observed) -> str | None:
    bad = [r.status_code for r in o.replies if r.status_code != 200 or not r.output.strip()]
    return f"a step failed or was empty ({bad})" if wanted and bad else None


def _tool_call_count(counts: dict[str, int], o: Observed) -> str | None:
    seen = Counter(_tool_names(o))
    wrong = {name: seen[name] for name, n in counts.items() if seen[name] != n}
    return f"call counts differ from {counts}: {wrong}" if wrong else None


def _ticket_has(expected: dict[str, dict[str, Any]], o: Observed) -> str | None:
    for ticket_id, fields in expected.items():
        ticket = o.after.tickets.get(ticket_id, {})
        wrong = {k: ticket.get(k) for k, v in fields.items() if ticket.get(k) != v}
        if wrong:
            return f"{ticket_id} has {wrong}, expected {fields}"
    return None


def _pending_input_contains(fields: dict[str, Any], o: Observed) -> str | None:
    pending = _last(o).pending_approval
    shown = (pending or {}).get("input", {})
    return None if pending and fields.items() <= shown.items() else f"pending input is {shown}"


def _output_matches(pattern: str, o: Observed) -> str | None:
    return None if re.search(pattern, _last(o).output) else f"reply does not match {pattern!r}"


CHECKS: dict[str, Callable[[Any, Observed], str | None]] = {
    "tool_called": _tool_called,
    "no_tool_called": _no_tool_called,
    "output_contains_any": _output_contains_any,
    "approval_requested": _approval_requested,
    "pending_after": _pending_after,
    "no_write_before_approval": _no_write_before_approval,
    "no_write": _no_write,
    "ticket_created": _ticket_created,
    "no_duplicate_tickets": _no_duplicate_tickets,
    "comment_added_to": _comment_added_to,
    "declines": _declines,
    "completes": _completes,
    "tool_call_count": _tool_call_count,
    "ticket_has": _ticket_has,
    "pending_input_contains": _pending_input_contains,
    "output_matches": _output_matches,
}


def judge(expect: dict[str, Any], observed: Observed) -> list[str]:
    """Run every expected check. An unknown check name raises KeyError: a typo must not pass."""
    failures = []
    for name, value in expect.items():
        message = CHECKS[name](value, observed)
        if message:
            failures.append(f"{name}: {message}")
    return failures


# --- running ---


class Runner:
    def __init__(
        self,
        agent: httpx.AsyncClient,
        mock: httpx.AsyncClient,
        *,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._agent = agent
        self._mock = mock
        self._sleep = sleep
        self._clock = clock

    async def run_once(self, case: Case, run: int) -> RunResult:
        await self._mock_call("POST", "/admin/reset")
        for action in case.setup:
            await self._mock_call(action["method"], action["path"], json=action.get("json"))
        before = await self.snapshot()
        session = f"eval-{uuid.uuid4()}"  # a fresh session per run, 33+ characters
        replies: list[Reply] = []
        for step in case.steps:
            replies.append(await self._ask(session, step.prompt))
            if replies[-1].status_code != 200:
                break  # a failed step cannot be judged fairly
        observed = Observed(replies=replies, before=before, after=await self.snapshot())
        failures = self._failures(case, observed)
        calls = [call for reply in replies for call in reply.tool_calls]
        return RunResult(
            case_id=case.id,
            run=run,
            passed=not failures,
            failures=failures,
            latency_ms=sum(reply.latency_ms for reply in replies),
            tools=[call["name"] for call in calls],
            retries=sum(len(call.get("retry_reasons", [])) for call in calls),
            replies=replies,
            prompts=[step.prompt for step in case.steps],
        )

    def _failures(self, case: Case, observed: Observed) -> list[str]:
        last = observed.replies[-1]
        if last.status_code != 200 or len(observed.replies) < len(case.steps):
            return [f"HTTP {last.status_code} on step {len(observed.replies)}: {last.error}"]
        return judge(case.expect, observed)

    async def snapshot(self) -> Snapshot:
        tickets: dict[str, dict[str, Any]] = {}
        page = 1
        while True:
            response = await self._mock_call(
                "GET", "/tickets", params={"page": page, "page_size": 50}
            )
            body = response.json()
            for item in body["items"]:  # search results carry no comments, so read each ticket
                full = await self._mock_call("GET", f"/tickets/{item['id']}")
                tickets[item["id"]] = full.json()
            if not body["has_more"]:
                return Snapshot(tickets=tickets)
            page += 1

    async def _ask(self, session: str, prompt: str) -> Reply:
        start = self._clock()
        try:
            response = await self._agent.post(
                "/invocations", json={"prompt": prompt}, headers={SESSION_HEADER: session}
            )
        except httpx.HTTPError as exc:
            return Reply(status_code=0, error=f"no response ({type(exc).__name__})")
        latency = round((self._clock() - start) * 1000)
        try:
            body = response.json()
        except ValueError:
            error = response.text[:200]
            return Reply(status_code=response.status_code, latency_ms=latency, error=error)
        if response.status_code != 200:
            error = body.get("message", "") if isinstance(body, dict) else ""
            return Reply(status_code=response.status_code, latency_ms=latency, error=error)
        return Reply(
            status_code=200,
            output=body.get("output", ""),
            tool_calls=body.get("tool_calls", []),
            pending_approval=body.get("pending_approval"),
            models=[e["model"] for e in body.get("trace", []) if e.get("type") == "llm"],
            latency_ms=latency,
        )

    async def _mock_call(self, method: str, path: str, **kwargs: Any) -> httpx.Response:
        """A call to the ticketing API, retried: its reset and reads can hit FLAKY too."""
        for _ in range(MOCK_TRIES):
            try:
                response = await self._mock.request(method, path, **kwargs)
            except httpx.TransportError:  # e.g. a pooled connection to a mock that was restarted
                await self._sleep(0.2)
                continue
            if response.status_code == 429:
                await self._sleep(float(response.headers.get("Retry-After", 1)))
            elif response.status_code >= 500:
                await self._sleep(0.2)
            else:
                response.raise_for_status()
                return response
        raise RuntimeError(f"{method} {path} kept failing after {MOCK_TRIES} tries")


def load_cases(path: Path) -> list[Case]:
    return [Case(**raw) for raw in json.loads(path.read_text())["cases"]]


def group_by_env(cases: list[Case]) -> list[tuple[dict[str, str], list[Case]]]:
    """Cases that need the same ticketing environment run together; the default group is first."""
    groups: dict[tuple[tuple[str, str], ...], list[Case]] = {}
    for case in cases:
        env = tuple(sorted((DEFAULT_ENV | case.env).items()))
        groups.setdefault(env, []).append(case)
    default = tuple(sorted(DEFAULT_ENV.items()))
    ordered = sorted(groups, key=lambda env: env != default)  # stable: others keep their order
    return [(dict(env), groups[env]) for env in ordered]


def recreate_ticketing(env: dict[str, str], api_key: str) -> None:
    """Restart only the mock, with this environment (FLAKY is read once, at its startup)."""
    command = ["docker", "compose", "up", "-d", "--force-recreate", "ticketing"]
    for attempt in (1, 2):  # Docker Desktop sometimes fails a recreate once; a second try works
        done = subprocess.run(  # noqa: S603 - a fixed command, nothing user-supplied
            command, capture_output=True, text=True,
            env=os.environ | env | {"TICKETING_API_KEY": api_key},
        )  # fmt: skip
        if done.returncode == 0:
            return
        print(f"docker compose failed (try {attempt}): {done.stderr.strip()[-500:]}")
    raise RuntimeError("could not restart the ticketing container; see the docker error above")


async def wait_healthy(mock: httpx.AsyncClient, flaky: bool, seconds: int = 30) -> None:
    for _ in range(seconds * 2):
        try:
            health = (await mock.get("/health")).json()
            if health.get("flaky") == flaky:
                return
        except (httpx.HTTPError, ValueError):
            pass
        await asyncio.sleep(0.5)
    raise RuntimeError(f"the ticketing service did not come up with flaky={flaky}")


# --- the scorecard ---


def scorecard(results: list[RunResult], *, label: str | None, repeat: int) -> str:
    by_case: dict[str, list[RunResult]] = {}
    for result in results:
        by_case.setdefault(result.case_id, []).append(result)
    seen = sorted({m for r in results for reply in r.replies for m in reply.models})
    lines = [
        "# Eval results",
        "",
        f"- Model: {label or ', '.join(seen) or 'unknown'}",
        f"- Date: {datetime.now(UTC):%Y-%m-%d %H:%M} UTC",
        f"- Runs per case: {repeat}, unless a case sets its own (the runs column)",
        "",
        "| case | runs | pass rate | tools called | retries seen | p50 latency (ms) |",
        "|---|---|---|---|---|---|",
    ]
    for case_id, runs in by_case.items():
        rate = round(100 * sum(r.passed for r in runs) / len(runs))
        tools = Counter(tool for r in runs for tool in r.tools)
        called = ", ".join(f"{name}×{n}" for name, n in sorted(tools.items())) or "none"
        p50 = round(statistics.median(r.latency_ms for r in runs))
        retries = sum(r.retries for r in runs)
        lines.append(f"| {case_id} | {len(runs)} | {rate}% | {called} | {retries} | {p50} |")
    lines += ["", "## Failures", ""]
    failed = [r for r in results if not r.passed]
    for r in failed:
        lines += [f"### {r.case_id} (run {r.run + 1})", ""] + [f"- {f}" for f in r.failures] + [""]
        for prompt, reply in zip(r.prompts, r.replies, strict=False):
            lines += [f"> user: {prompt}", f"> agent: {reply.output[:400] or reply.error}", ""]
    return "\n".join(lines + ([] if failed else ["None.", ""]))


async def main() -> None:
    parser = argparse.ArgumentParser(description="Run the eval cases against the running service.")
    parser.add_argument("--repeat", type=int, default=1, help="runs per case")
    parser.add_argument("--label", help="the model under test (default: read from the replies)")
    parser.add_argument("--case", action="append", help="only this case id (repeatable)")
    parser.add_argument("--cases", default="eval/cases.json")
    parser.add_argument("--out", default="eval/results.md")
    args = parser.parse_args()

    agent_url = os.environ.get("AGENT_URL", "http://localhost:8080")
    mock_url = os.environ.get("TICKETING_BASE_URL", "http://localhost:8000")
    api_key = os.environ.get("TICKETING_API_KEY") or "dev-key-123"  # the mock's default

    cases = [c for c in load_cases(Path(args.cases)) if not args.case or c.id in args.case]
    results: list[RunResult] = []
    async with (
        httpx.AsyncClient(base_url=agent_url, timeout=300) as agent,
        httpx.AsyncClient(base_url=mock_url, headers={"X-API-Key": api_key}, timeout=30) as mock,
    ):
        runner = Runner(agent, mock)
        try:
            for env, group in group_by_env(cases):
                recreate_ticketing(env, api_key)
                await wait_healthy(mock, flaky=env["FLAKY"] == "1")
                for case in group:
                    for run in range(case.runs or args.repeat):
                        results.append(await runner.run_once(case, run))
                        verdict = "pass" if results[-1].passed else "FAIL"
                        reasons = "" if results[-1].passed else f" {results[-1].failures}"
                        print(f"{case.id} run {run + 1}: {verdict}{reasons}", flush=True)
        finally:
            # The results first: a failed restart below must not lose a finished run.
            text = scorecard(results, label=args.label, repeat=args.repeat)
            await asyncio.to_thread(Path(args.out).write_text, text)
            print("\n" + text)
            recreate_ticketing(DEFAULT_ENV, api_key)  # never leave the mock flaky


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except httpx.ConnectError as exc:
        sys.exit(f"Could not reach the service ({exc}). Is the agent running on :8080?")
