from __future__ import annotations

import asyncio

from pydantic_ai import Agent

from answer_verification import parse_answer_with_reasoning
from prompts import SAMPLE_N_MATH, SAMPLE_N_MCQ, SAMPLE_N_TEXT
from shared import QuestionType

from ..stages import Answer, CreatePopulation, StageContext


def parse_answer(response: str, question_type: QuestionType) -> Answer:
    """Parse answer from response using unified extraction."""
    reasoning, choice_str = parse_answer_with_reasoning(
        response, question_type=question_type
    )
    return Answer(reasoning, choice_str)


class SampleN(CreatePopulation):
    """
    Create population by sampling N independent answers.

    Uses settings.samples or settings.width to determine N.
    """

    async def __call__(self, context: StageContext) -> list[Answer]:
        question_type = context.question_type
        prompt_family = question_type.prompt_family
        if prompt_family == "mcq":
            prompts = SAMPLE_N_MCQ
        elif prompt_family == "math":
            prompts = SAMPLE_N_MATH
        else:
            prompts = SAMPLE_N_TEXT

        agent: Agent[None, str] = Agent(
            context.model,
            system_prompt=prompts["solver"],
            model_settings=context.model_settings,
        )

        prompt = context.example.to_prompt()

        async def solve_one() -> Answer:
            try:
                result = await agent.run(f"Problem:\n{prompt}")
                context.record_usage(result.usage())
                return parse_answer(result.output, question_type=question_type)
            except Exception as exc:
                return Answer(
                    reasoning=f"Exception during sampling: {exc}",
                    choice=None,
                )

        # Use width as the population size (can also be configured as samples)
        n = max(1, context.settings.width)
        tasks = [solve_one() for _ in range(n)]
        answers = await asyncio.gather(*tasks)
        return list(answers)


__all__ = ["SampleN", "parse_answer"]
