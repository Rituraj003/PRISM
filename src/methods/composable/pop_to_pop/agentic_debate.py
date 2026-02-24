from __future__ import annotations

from pydantic_ai import Agent
from tqdm.asyncio import tqdm as atqdm

from answer_verification import parse_answer_with_reasoning
from prompts import AGENTIC_DEBATE_MATH, AGENTIC_DEBATE_MCQ, AGENTIC_DEBATE_TEXT
from shared import QuestionType

from ..stages import Answer, PopulationToPopulation, StageContext


def parse_answer(response: str, question_type: QuestionType) -> Answer:
    """Parse answer from response using unified extraction."""
    reasoning, choice_str = parse_answer_with_reasoning(
        response, question_type=question_type
    )
    return Answer(reasoning, choice_str)


class AgenticDebate(PopulationToPopulation):
    """
    Refine each answer in population based on all the answers in the previous iteration.

    This is the core logic from the parallel method:
    Concatenate all the answers from the previous iteration as context, then iterate to refine each answer.
    Can be run multiple times (controlled by settings.depth).
    """

    async def __call__(
        self, context: StageContext, population: list[Answer]
    ) -> list[Answer]:
        question_type = context.question_type
        prompt_family = question_type.prompt_family
        if prompt_family == "mcq":
            prompts = AGENTIC_DEBATE_MCQ
        elif prompt_family == "math":
            prompts = AGENTIC_DEBATE_MATH
        else:
            prompts = AGENTIC_DEBATE_TEXT

        iterator_agent: Agent[None, str] = Agent(
            context.model,
            system_prompt=prompts["iterator"],
            model_settings=context.model_settings,
        )

        question = context.example.to_prompt()

        all_evidence = "\n\n".join(
            f"Reasoning:\n{answer.reasoning}\nChoice: {answer.choice}"
            for answer in population
        )

        async def refine_one(answer: Answer) -> Answer:
            try:
                # Iterate to improve
                iterate_prompt = (
                    f"Problem:\n{question}\n\n"
                    f"All previous answers and reasonings:\n{all_evidence}\n\n"
                    f"Previous reasoning:\n{answer.reasoning}\n"
                    f"Previous choice: {answer.choice}\n\n"
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
            *tasks, desc="agentic_debate", total=len(tasks), dynamic_ncols=True
        )
        return list(refined)


__all__ = ["AgenticDebate", "parse_answer"]
