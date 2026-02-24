from __future__ import annotations

import re

from .stages import Answer


STEP_TAG_RE = re.compile(r"<step\b", re.IGNORECASE)


def coerce_answer_stepwise(answer: Answer) -> Answer:
    """Coerce an answer in the population to stepwise format enclosed in XMLs."""
    reasoning = answer.reasoning
    choice = answer.choice

    if not reasoning:
        reasoning = f"<step>Proposed answer: {choice or 'N/A'}</step>"
    elif STEP_TAG_RE.search(reasoning) is None:
        reasoning = (
            "<step>"
            + "</step>\n\n<step>".join(
                s.strip("- ") for s in reasoning.split("\n\n") if s.strip("- ")
            )
            + "</step>"
        )

    return Answer(reasoning=reasoning, choice=choice)
