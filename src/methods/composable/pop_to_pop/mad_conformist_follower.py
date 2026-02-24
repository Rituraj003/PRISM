from __future__ import annotations

import random

from pydantic_ai import Agent
from tqdm.asyncio import tqdm as atqdm

from answer_verification import parse_answer_with_reasoning
from prompts import (
    MAD_CONFORMIST_FOLLOWER_MATH,
    MAD_CONFORMIST_FOLLOWER_MCQ,
    MAD_CONFORMIST_FOLLOWER_TEXT,
)
from shared import QuestionType

from ..pop_to_answer.majority_vote import MajorityVote

from ..stages import Answer, PopulationToPopulation, StageContext


def parse_answer(response: str, question_type: QuestionType) -> Answer:
    """Parse answer from response using unified extraction."""
    reasoning, choice_str = parse_answer_with_reasoning(
        response, question_type=question_type
    )
    return Answer(reasoning, choice_str)


class MadConformist(PopulationToPopulation):
    """
    Preserves answers matching majority, refines non-conformists.
    """

    async def __call__(
        self, context: StageContext, population: list[Answer]
    ) -> list[Answer]:
        question_type = context.question_type
        prompt_family = question_type.prompt_family
        if prompt_family == "mcq":
            prompts = MAD_CONFORMIST_FOLLOWER_MCQ
        elif prompt_family == "math":
            prompts = MAD_CONFORMIST_FOLLOWER_MATH
        else:
            prompts = MAD_CONFORMIST_FOLLOWER_TEXT

        population_choices = [ans.choice for ans in population]
        majority_choice = MajorityVote.vote_majority(
            population_choices,
            question_type=question_type,
        )

        conformist_answers: list[Answer] = []
        non_conformist_answers: list[Answer] = []

        if majority_choice is not None:
            for ans in population:
                if ans.choice == majority_choice:
                    conformist_answers.append(ans)
                else:
                    non_conformist_answers.append(ans)
        else:
            non_conformist_answers = population

        if not non_conformist_answers:
            return conformist_answers

        iterator_agent: Agent[None, str] = Agent(
            context.model,
            system_prompt=prompts["iterator"],
            model_settings=context.model_settings,
        )

        question = context.example.to_prompt()

        async def refine_one(answer: Answer) -> Answer:
            try:
                iterate_prompt = (
                    f"Problem:\n{question}\n\n"
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

        tasks = [refine_one(answer) for answer in non_conformist_answers]
        refined_answers = await atqdm.gather(
            *tasks, desc="mad-refine", total=len(tasks), dynamic_ncols=True
        )

        return conformist_answers + list(refined_answers)


class MadFollower(PopulationToPopulation):
    """
    Randomly selects ~30% to become followers of majority, refines the rest.
    """

    def __init__(
        self,
        follower_ratio: float = 0.3,
    ):
        self.follower_ratio = follower_ratio

    async def __call__(
        self, context: StageContext, population: list[Answer]
    ) -> list[Answer]:
        question_type = context.question_type
        prompt_family = question_type.prompt_family
        if prompt_family == "mcq":
            prompts = MAD_CONFORMIST_FOLLOWER_MCQ
        elif prompt_family == "math":
            prompts = MAD_CONFORMIST_FOLLOWER_MATH
        else:
            prompts = MAD_CONFORMIST_FOLLOWER_TEXT
        width = max(1, context.settings.width)
        follower_ratio = self.follower_ratio

        population_choices = [ans.choice for ans in population]
        majority_choice = MajorityVote.vote_majority(
            population_choices,
            question_type=question_type,
        )

        rng = random.Random(context.settings.seed + context.question_index)
        shuffled_population = population.copy()
        rng.shuffle(shuffled_population)

        num_followers = int(width * follower_ratio)

        if majority_choice is not None and num_followers > 0:
            # Use list comprehension to create distinct instances (not references to same object)
            follower_answers = [
                Answer(
                    reasoning="Following the majority choice from the population.",
                    choice=majority_choice,
                )
                for _ in range(num_followers)
            ]
            revisor_answers = shuffled_population[num_followers:]
        else:
            follower_answers = []
            revisor_answers = shuffled_population

        if not revisor_answers:
            return follower_answers

        iterator_agent: Agent[None, str] = Agent(
            context.model,
            system_prompt=prompts["iterator"],
            model_settings=context.model_settings,
        )

        question = context.example.to_prompt()

        async def refine_one(answer: Answer) -> Answer:
            try:
                iterate_prompt = (
                    f"Problem:\n{question}\n\n"
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

        tasks = [refine_one(answer) for answer in revisor_answers]
        refined_answers = await atqdm.gather(
            *tasks, desc="mad-follower", total=len(tasks), dynamic_ncols=True
        )

        return follower_answers + list(refined_answers)


__all__ = ["MadConformist", "MadFollower", "parse_answer"]
