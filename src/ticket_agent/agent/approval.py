"""What a typed answer to a pending action means.

The UI's Accept and Decline buttons need no interpretation: they send a decision, not text. A
typed answer approves only when it is explicit: an exact phrase such as "yes" or "approve", or
the classifier saying CONFIRM for a clear go-ahead. Vague agreement ("ok", "sure") and anything
unclear count as UNRELATED, so nothing runs by mistake.
"""

import re

from ticket_agent.agent.llm import Message
from ticket_agent.agent.prompts import CLASSIFIER_PROMPT
from ticket_agent.agent.state import Answer, PendingAction

CONFIRM_PHRASES = {
    "yes",
    "yes go ahead",
    "yes do it",
    "confirm",
    "confirmed",
    "approve",
    "approved",
    "go ahead",
    "do it",
    "please do",
    "proceed",
    "ok go ahead",
}  # fmt: skip  (no "ok", "sure" or "yeah": those are not an explicit go-ahead)
DECLINE_PHRASES = {
    "no", "n", "nope", "cancel", "cancel it", "cancel that", "stop", "dont", "do not",
    "no thanks", "never mind", "nevermind", "no cancel", "no cancel that", "no dont",
    "decline", "declined",
}  # fmt: skip
LABELS: tuple[Answer, ...] = ("CONFIRM", "DECLINE", "CHANGE", "UNRELATED")


def match_exact(text: str) -> Answer | None:
    """Whole phrases only: "ok, but make it urgent" must not count as a confirmation."""
    normalized = re.sub(r"[^a-z ]", "", text.lower().replace("'", ""))
    normalized = re.sub(r"\b(please|thanks|thank you)\b", "", normalized)
    normalized = " ".join(normalized.split())
    if normalized in CONFIRM_PHRASES:
        return "CONFIRM"
    if normalized in DECLINE_PHRASES:
        return "DECLINE"
    return None


def classifier_messages(pending: PendingAction, text: str) -> list[Message]:
    question = (
        f"Proposed action: {pending.describe()}\n"
        f"Reply of the user: {text}\n"
        "Give one word: CONFIRM, DECLINE, CHANGE or UNRELATED."
    )
    return [
        Message(role="system", content=CLASSIFIER_PROMPT),
        Message(role="user", content=question),
    ]


def parse_label(reply: str) -> Answer:
    # Whole words: "UNCONFIRMED" is not CONFIRM.
    words = re.findall(r"[A-Z]+", reply.upper())
    found = [label for word in words for label in LABELS if word == label]
    return found[0] if found else "UNRELATED"
