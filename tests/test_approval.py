"""What the user's answer to a pending action means (CLAUDE.md invariant 3)."""

import pytest

from ticket_agent.agent.approval import classifier_messages, match_exact, parse_label
from ticket_agent.agent.state import PendingAction


@pytest.mark.parametrize(
    "text",
    [
        "yes",
        "Yes!",
        "yes, go ahead",
        "Approve",
        "confirm",
        "yes please",
        "go ahead please",
        "Do it.",
    ],
)
def test_confirm_phrases(text):
    assert match_exact(text) == "CONFIRM"


@pytest.mark.parametrize("text", ["no", "No.", "Decline", "cancel that", "never mind", "don't"])
def test_decline_phrases(text):
    assert match_exact(text) == "DECLINE"


@pytest.mark.parametrize(
    "text",
    ["ok, but make it urgent", "yes and also show me TCK-1002", "actually no, don't", "how many?"],
)
def test_anything_else_is_left_to_the_classifier(text):
    # No prefix matching: "ok, but make it urgent" must never count as a confirmation.
    assert match_exact(text) is None


@pytest.mark.parametrize("text", ["ok", "okay", "sure", "yeah", "y", "fine"])
def test_vague_agreement_is_not_an_exact_confirm(text):
    # Only an explicit "yes" (or the Accept button) approves without asking the classifier.
    assert match_exact(text) is None


@pytest.mark.parametrize(
    ("action", "input", "summary"),
    [
        (
            "create_ticket",
            {"title": "Printer jammed", "requester": "Tom Okafor", "priority": "high"},
            'create a ticket "Printer jammed" for Tom Okafor (priority high)',
        ),
        (
            "update_ticket",
            {"ticket_id": "TCK-1002", "status": "resolved", "assignee": "Ana"},
            "update TCK-1002: status to resolved, assignee to Ana",
        ),
        (
            "add_comment",
            {"ticket_id": "TCK-1012", "author": "Aisha Khan", "body": "Still down"},
            'add a comment to TCK-1012 as Aisha Khan: "Still down"',
        ),
        ("close_ticket", {"ticket_id": "TCK-1"}, 'close_ticket {"ticket_id": "TCK-1"}'),
    ],
)
def test_a_pending_action_reads_as_plain_language(action, input, summary):
    assert PendingAction(action=action, input=input).summary() == summary


@pytest.mark.parametrize(
    ("reply", "label"),
    [
        ("CONFIRM", "CONFIRM"),
        ("decline", "DECLINE"),
        ("The answer is CHANGE.", "CHANGE"),
        ("UNRELATED", "UNRELATED"),
        ("UNCONFIRMED", "UNRELATED"),  # whole words only
        ("I am not sure", "UNRELATED"),  # unclear means nothing runs
        ("", "UNRELATED"),
    ],
)
def test_parse_label(reply, label):
    assert parse_label(reply) == label


def test_the_classifier_sees_the_action_and_the_reply_only():
    pending = PendingAction(action="create_ticket", input={"title": "Printer jammed"})
    system, user = classifier_messages(pending, "make it high priority")
    assert system.role == "system" and "## IDENTITY" in system.content
    assert 'create_ticket {"title": "Printer jammed"}' in user.content
    assert "make it high priority" in user.content
