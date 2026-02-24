from __future__ import annotations

from pydantic_ai import Agent

from answer_verification import parse_answer_with_reasoning
from prompts import LLM_AGGREGATE_MATH, LLM_AGGREGATE_MCQ, LLM_AGGREGATE_TEXT
from shared import QuestionType

from ..stages import Answer, PopulationToAnswer, StageContext


def parse_answer(response: str, question_type: QuestionType) -> Answer:
    """
    Parse answer from LLM response using unified extraction.

    Args:
        response: The LLM response text
        question_type: Question modality to guide parsing

    Returns:
        Answer object with reasoning and choice
    """
    reasoning, choice_str = parse_answer_with_reasoning(
        response, question_type=question_type
    )
    return Answer(reasoning, choice_str)


class LLMAggregate(PopulationToAnswer):
    """
    Reduce population to final answer using LLM aggregation.

    Shows all answers to an LLM and asks it to determine the best choice.
    """

    async def __call__(self, context: StageContext, population: list[Answer]) -> Answer:
        if not population:
            return Answer(
                reasoning="No answers in population for LLM aggregation",
                choice=None,
            )

        question_type = context.question_type
        prompt_family = question_type.prompt_family
        if prompt_family == "mcq":
            prompts = LLM_AGGREGATE_MCQ
        elif prompt_family == "math":
            prompts = LLM_AGGREGATE_MATH
        else:
            prompts = LLM_AGGREGATE_TEXT

        aggregator_agent: Agent[None, str] = Agent(
            context.model,
            system_prompt=prompts["aggregator"],
            model_settings=context.model_settings,
        )

        question = context.example.to_prompt()

        # Format all answers
        answers_text = "\n\n".join(
            f"Answer {idx + 1}:\n{ans.reasoning}\nChoice: {ans.choice}"
            for idx, ans in enumerate(population)
        )

        prompt = (
            f"Problem:\n{question}\n\n"
            f"Proposed answers from multiple sources:\n{answers_text}\n\n"
            "Analyze all answers and determine the best final answer. "
            "Consider which reasoning is most sound and which choice is most likely correct."
        )

        try:
            result = await aggregator_agent.run(prompt)
            context.record_usage(result.usage())
            return parse_answer(result.output, question_type=question_type)
        except Exception as exc:
            return Answer(
                reasoning=f"Exception during LLM aggregation: {exc}",
                choice=None,
            )


__all__ = ["LLMAggregate", "parse_answer"]
