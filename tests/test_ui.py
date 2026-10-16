"""The web UI is one static file. pytest cannot run JavaScript, so these check what it can:
that the page is served, that it ships inside the package, and that its source follows the rules
(no innerHTML, no external resources). The click-through gate is done by hand."""

import re
from importlib.resources import files

import httpx
import pytest

from ticket_agent.agent.llm import ScriptedLLM
from ticket_agent.config import Settings
from ticket_agent.main import create_app

pytestmark = pytest.mark.anyio

PAGE = files("ticket_agent").joinpath("ui", "index.html")


async def test_the_page_is_served_at_the_root(make_client):
    settings = Settings(ticketing_api_key="k")
    app = create_app(llm=ScriptedLLM([]), make_client=make_client, settings=settings)
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://agent.test") as http:
            response = await http.get("/")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    assert response.text == PAGE.read_text()


def test_the_page_ships_inside_the_package():
    # The container image installs the package without the source tree.
    assert PAGE.is_file()


def test_no_server_text_is_ever_rendered_as_html():
    source = PAGE.read_text()
    for forbidden in ("innerHTML", "outerHTML", "insertAdjacentHTML", "document.write", "eval("):
        assert forbidden not in source
    assert "textContent" in source


def test_no_external_resources():
    source = PAGE.read_text()
    assert not re.search(r"https?://", source), "no CDN, font or script from the internet"
    assert not re.search(r"<script[^>]+src=", source) and "<link" not in source


def test_it_stays_small_enough_to_explain():
    assert len(PAGE.read_text().splitlines()) <= 350


def test_it_only_calls_the_service_contract_and_its_status_endpoints():
    source = PAGE.read_text()
    called = set(re.findall(r'fetch\("([^"]+)"', source))
    assert called == {"/invocations", "/ping", "/info"}  # no special endpoint for approvals
    assert "X-Amzn-Bedrock-AgentCore-Runtime-Session-Id" in source


def test_accept_and_decline_send_a_decision_not_a_typed_message():
    source = PAGE.read_text()
    assert 'decide(box, "accept")' in source and 'decide(box, "decline")' in source
    assert "{ approval }" in source  # the request carries the decision, no prompt text
    assert 'send("Approve")' not in source


def test_replies_render_markdown_by_building_elements():
    source = PAGE.read_text()
    assert "function md(" in source and "function inline(" in source
    for tag in ('"strong"', '"code"', '"table"', '"li"'):
        assert tag in source


def test_the_trace_shows_tool_calls():
    source = PAGE.read_text()
    assert 't.type === "tool"' in source and ".k-tool" in source


def test_the_controls_are_accessible():
    source = PAGE.read_text()
    assert 'aria-live="polite"' in source  # new replies are announced
    assert '"aria-label", "Approval request"' in source
    assert "onclick=" not in source and ".onclick" not in source  # handlers via addEventListener
