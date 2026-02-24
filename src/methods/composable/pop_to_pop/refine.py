from __future__ import annotations

from pydantic_ai import Agent
from tqdm.asyncio import tqdm as atqdm

from answer_verification import parse_answer_with_reasoning
from prompts import REFINE_MATH, REFINE_MCQ, REFINE_TEXT
from shared import QuestionType

from ..stages import Answer, PopulationToPopulation, StageContext

CRITIC_INSTRUCTIONS = (
    "Role: Critic.\n"
    "Given a problem and a proposed answer, list concrete issues and actionable fixes. Be terse and specific."
)


def parse_answer(response: str, question_type: QuestionType) -> Answer:
    """Parse answer from response using unified extraction."""
    reasoning, choice_str = parse_answer_with_reasoning(
        response, question_type=question_type
    )
    return Answer(reasoning, choice_str)


class Refine(PopulationToPopulation):
    """
    Refine each answer in population using critique-iterate loop.

    This is the core logic from the parallel method:
    For each answer, generate critique, then iterate to improve.
    Can be run multiple times (controlled by settings.depth).
    """

    async def __call__(
        self, context: StageContext, population: list[Answer]
    ) -> list[Answer]:
        question_type = context.question_type
        prompt_family = question_type.prompt_family
        if prompt_family == "mcq":
            prompts = REFINE_MCQ
        elif prompt_family == "math":
            prompts = REFINE_MATH
        else:
            prompts = REFINE_TEXT

        critic_agent: Agent[None, str] = Agent(
            context.model,
            system_prompt=CRITIC_INSTRUCTIONS,
            model_settings=context.model_settings,
        )

        iterator_agent: Agent[None, str] = Agent(
            context.model,
            system_prompt=prompts["iterator"],
            model_settings=context.model_settings,
        )

        question = context.example.to_prompt()

        async def refine_one(answer: Answer) -> Answer:
            try:
                # Generate critique
                critique_prompt = (
                    f"Problem:\n{question}\n\n"
                    f"Proposed answer reasoning:\n{answer.reasoning}\n"
                    f"Proposed choice: {answer.choice}"
                )
                critique_result = await critic_agent.run(critique_prompt)
                context.record_usage(critique_result.usage())
                critique = critique_result.output.strip()

                # Iterate to improve
                iterate_prompt = (
                    f"Problem:\n{question}\n\n"
                    f"Previous reasoning:\n{answer.reasoning}\n"
                    f"Previous choice: {answer.choice}\n\n"
                    f"Critique:\n{critique}\n"
                    "Revise and end with the answer tag."
                )
                iterate_result = await iterator_agent.run(iterate_prompt)
                context.record_usage(iterate_result.usage())
                return parse_answer(iterate_result.output, question_type=question_type)
            except Exception as exc:
                return Answer(
                    reasoning=f"Exception during refinement: {exc}",
                    choice=answer.choice,
                )

        # Refine all answers in parallel
        tasks = [refine_one(answer) for answer in population]
        refined = await atqdm.gather(
            *tasks, desc="refine", total=len(tasks), dynamic_ncols=True
        )
        return list(refined)


__all__ = ["Refine", "parse_answer"]
