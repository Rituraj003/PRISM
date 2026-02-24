from __future__ import annotations

import random

from pydantic_ai import Agent
from tqdm.asyncio import tqdm as atqdm

from answer_verification import parse_answer_with_reasoning
from prompts import (
    RECURSIVE_AGGREGATE_MATH,
    RECURSIVE_AGGREGATE_MCQ,
    RECURSIVE_AGGREGATE_TEXT,
)
from shared import QuestionType

from ..stages import Answer, PopulationToPopulation, StageContext


def parse_answer(response: str, question_type: QuestionType) -> Answer:
    """Parse answer from response using unified extraction."""
    reasoning, choice_str = parse_answer_with_reasoning(
        response, question_type=question_type
    )
    return Answer(reasoning, choice_str)


class RecursiveAggregate(PopulationToPopulation):
    """
    Recursively aggregate population by sampling and combining.

    For each new answer, sample a pool of existing answers and aggregate them.
    This is the core logic from recursive_aggregation method.
    """

    async def __call__(
        self, context: StageContext, population: list[Answer]
    ) -> list[Answer]:
        question_type = context.question_type
        prompt_family = question_type.prompt_family
        if prompt_family == "mcq":
            prompts = RECURSIVE_AGGREGATE_MCQ
        elif prompt_family == "math":
            prompts = RECURSIVE_AGGREGATE_MATH
        else:
            prompts = RECURSIVE_AGGREGATE_TEXT
        width = max(1, context.settings.width)
        agg_pool = max(1, context.settings.agg_pool_size)

        if agg_pool > len(population):
            # Can't sample more than we have
            agg_pool = len(population)

        aggregator_agent: Agent[None, str] = Agent(
            context.model,
            system_prompt=prompts["aggregator"],
            model_settings=context.model_settings,
        )

        question = context.example.to_prompt()
        rng = random.Random(context.settings.seed + context.question_index)

        async def aggregate_subset(subset: list[Answer]) -> Answer:
            try:
                answers_text = "\n\n".join(
                    f"Answer {idx + 1}:\n{ans.reasoning}\nChoice: {ans.choice}"
                    for idx, ans in enumerate(subset)
                )
                prompt = (
                    f"Problem:\n{question}\n\n"
                    f"Proposed answers:\n{answers_text}\n\n"
                    "Create an improved answer that reasons carefully before giving the final choice."
                )
                result = await aggregator_agent.run(prompt)
                context.record_usage(result.usage())
                return parse_answer(result.output, question_type=question_type)
            except Exception as exc:
                return Answer(
                    reasoning=f"Exception during aggregation: {exc}",
                    choice=None,
                )

        # Create new population by aggregating random subsets
        subsets: list[list[Answer]] = []
        for _ in range(width):
            indices = rng.sample(range(len(population)), agg_pool)
            subset = [population[i] for i in indices]
            subsets.append(subset)

        tasks = [aggregate_subset(subset) for subset in subsets]
        new_population = await atqdm.gather(
            *tasks, desc="rsa-aggregate", total=len(tasks), dynamic_ncols=True
        )
        return list(new_population)


__all__ = ["RecursiveAggregate", "parse_answer"]
