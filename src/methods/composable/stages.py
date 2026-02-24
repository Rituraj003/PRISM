from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from pydantic_ai.models import Model
from pydantic_ai.settings import ModelSettings
from pydantic_ai.usage import RunUsage

from shared import (
    EvaluationExample,
    QuestionType,
    SharedSettings,
    TokenUsage,
    extract_usage_tokens,
)


@dataclass
class Answer:
    """Represents a single answer with reasoning and choice."""

    reasoning: str
    choice: str | None


@dataclass
class StageContext:
    """Context passed to all stages."""

    settings: SharedSettings
    model: Model
    model_settings: ModelSettings
    example: EvaluationExample
    question_index: int
    question_type: QuestionType
    token_usage: TokenUsage = field(default_factory=TokenUsage)
    prm_temperature: float = 0.0
    # Cache for cross-depth data (e.g., PRM scores to avoid re-grading unchanged answers)
    cache: dict[str, Any] = field(default_factory=dict)

    def record_usage(self, usage: RunUsage) -> None:
        """Accumulate token usage from a model run."""
        self.token_usage.add(extract_usage_tokens(usage))

    def record_secondary_usage(self, usage: RunUsage) -> None:
        """Accumulate token usage from a secondary model run."""
        self.token_usage.add_secondary(extract_usage_tokens(usage))


class CreatePopulation(Protocol):
    """
    Stage 1: Create initial population of answers.

    Takes an example and generates N initial answers.
    """

    async def __call__(self, context: StageContext) -> list[Answer]:
        """Generate initial population of answers."""
        ...


class PopulationToPopulation(Protocol):
    """
    Stage 2: Transform one population into another.

    Takes a population of answers and refines/transforms them.
    Can be chained multiple times.
    """

    async def __call__(
        self, context: StageContext, population: list[Answer]
    ) -> list[Answer]:
        """Transform population to improved population."""
        ...


class PopulationToAnswer(Protocol):
    """
    Stage 3: Reduce population to final answer.

    Takes a population and produces a single final answer.
    """

    async def __call__(self, context: StageContext, population: list[Answer]) -> Answer:
        """Reduce population to single final answer."""
        ...


__all__ = [
    "Answer",
    "StageContext",
    "CreatePopulation",
    "PopulationToPopulation",
    "PopulationToAnswer",
]
