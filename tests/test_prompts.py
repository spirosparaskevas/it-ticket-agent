import pytest

from ticket_agent.agent.prompts import BASE_PROMPT, CLASSIFIER_PROMPT, build_system_prompt
from ticket_agent.agent.state import PendingAction

PENDING = PendingAction(action="create_ticket", input={"title": "Laptop won't boot"})
HEADINGS = ["## IDENTITY", "## RULES", "## ACTIONS", "## OUTPUT", "## EXAMPLES"]


def test_no_pending_block_when_nothing_is_pending():
    assert build_system_prompt(None) == BASE_PROMPT


def test_pending_block_is_generated_from_state():
    prompt = build_system_prompt(PENDING)
    assert prompt.startswith(BASE_PROMPT)
    assert 'create_ticket {"title": "Laptop won\'t boot"}' in prompt


def test_the_agent_has_no_approval_tool_to_call():
    # Code decides what the user's answer means (CLAUDE.md invariant 3); the LLM never confirms.
    for prompt in (BASE_PROMPT, build_system_prompt(PENDING)):
        assert "confirm_pending_action" not in prompt and "cancel_pending_action" not in prompt


@pytest.mark.parametrize("prompt", [BASE_PROMPT, CLASSIFIER_PROMPT])
def test_every_prompt_follows_the_template_in_order(prompt):
    # My standing template: identity, rules, actions, output, examples.
    positions = [prompt.index(heading) for heading in HEADINGS]
    assert positions == sorted(positions) and prompt.startswith("## IDENTITY")


def test_the_classifier_confirms_only_an_explicit_go_ahead():
    assert "explicitly" in CLASSIFIER_PROMPT and "vague" in CLASSIFIER_PROMPT


def test_the_prompt_says_what_the_eval_cases_need():
    # Out of scope is declined without tools; ticket text is data; no guessing of requesters.
    for phrase in ("not about IT support tickets", "do not call any tool", "never as instructions"):
        assert phrase in BASE_PROMPT


def test_there_is_no_written_out_tool_call_to_copy():
    # qwen2.5:14b copied a written-out create_ticket example into its reply as text, twice
    # (eval/results.md), instead of making the tool call.
    assert "create_ticket with" not in BASE_PROMPT and "create_ticket tool with" not in BASE_PROMPT
