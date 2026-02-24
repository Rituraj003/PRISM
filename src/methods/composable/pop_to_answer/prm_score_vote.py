from __future__ import annotations

import asyncio
from collections import defaultdict
from typing import cast

from pydantic_ai import Agent
from pydantic_ai.settings import ModelSettings
from tqdm.asyncio import tqdm as atqdm

from answer_verification import compute_prm_score, normalize_answer
from prompts import PRISM_MATH, PRISM_MCQ, PRISM_TEXT

from ..stages import Answer, PopulationToAnswer, StageContext
from ..stepwise import coerce_answer_stepwise


class PrmScoreVote(PopulationToAnswer):
    """
    Reduce population to final answer using PRM-score-weighted voting.

    Each candidate is verifier-scored once, and normalized choices are ranked by
    accumulated PRM score.
    """

    async def __call__(self, context: StageContext, population: list[Answer]) -> Answer:
        if not population:
            return Answer(
                reasoning="No answers in population for PRM-score vote",
                choice=None,
            )

        question_type = context.question_type
        prompt_family = question_type.prompt_family
        if prompt_family == "mcq":
            prompts = PRISM_MCQ
        elif prompt_family == "math":
            prompts = PRISM_MATH
        else:
            prompts = PRISM_TEXT

        verifier_settings_dict = dict(context.model_settings)
        verifier_settings_dict["temperature"] = 0.0
        verifier_settings_dict["top_p"] = 1.0
        verifier_settings = cast(ModelSettings, verifier_settings_dict)

        verifier_agent: Agent[None, str] = Agent(
            context.model,
            system_prompt=prompts["prm"],
            model_settings=verifier_settings,
        )

        question = context.example.to_prompt()

        async def score_one(
            idx: int, answer: Answer
        ) -> tuple[int, str | None, str | None, float]:
            normalized = normalize_answer(answer.choice, question_type=question_type)
            if normalized is None:
                return idx, None, answer.choice, 0.0

            coerced = coerce_answer_stepwise(answer)
            proposed_answer_text = answer.choice if answer.choice is not None else "<none>"
            prompt = (
                f"Problem:\n{question}\n\n"
                f"Reasoning:\n{coerced.reasoning}\n\n"
                f"Proposed Answer:\n{proposed_answer_text}\n\n"
                "Follow the system instructions exactly. Output only <step> lines and the final "
                "<answer> line."
            )

            try:
                result = await asyncio.wait_for(
                    verifier_agent.run(prompt),
                    timeout=300.0,
                )
                context.record_usage(result.usage())
                feedback = result.output or ""
            except Exception:
                return idx, normalized, answer.choice, 0.0

            score = compute_prm_score(feedback)
            return idx, normalized, answer.choice, score

        tasks = [score_one(i, ans) for i, ans in enumerate(population)]
        scored = await atqdm.gather(
            *tasks,
            desc="prm-score-vote",
            total=len(tasks),
            dynamic_ncols=True,
        )

        weight_by_choice: dict[str, float] = defaultdict(float)
        max_score_by_choice: dict[str, float] = defaultdict(float)
        pos_count_by_choice: dict[str, int] = defaultdict(int)
        best_raw_by_choice: dict[str, tuple[float, str]] = {}
        per_answer_lines: list[str] = []

        for idx, normalized, raw_choice, score in scored:
            per_answer_lines.append(
                f"  #{idx + 1:02d} choice={raw_choice} norm={normalized} score={score:.3f}"
            )
            if normalized is None:
                continue

            weight_by_choice[normalized] += score
            if score > max_score_by_choice[normalized]:
                max_score_by_choice[normalized] = score
            if score > 0:
                pos_count_by_choice[normalized] += 1

            if raw_choice is not None:
                best = best_raw_by_choice.get(normalized)
                if best is None or score > best[0]:
                    best_raw_by_choice[normalized] = (score, raw_choice)

        if not weight_by_choice:
            return Answer(
                reasoning="No valid choices in population for PRM-score vote",
                choice=None,
            )

        def sort_key(item: tuple[str, float]) -> tuple[float, float, int, str]:
            choice, total_score = item
            return (
                total_score,
                max_score_by_choice.get(choice, 0.0),
                pos_count_by_choice.get(choice, 0),
                choice,
            )

        best_norm, best_weight = max(weight_by_choice.items(), key=sort_key)
        best_raw = best_raw_by_choice.get(best_norm, (best_weight, best_norm))[1]

        totals = ", ".join(
            f"{choice}: {weight_by_choice[choice]:.3f}"
            for choice in sorted(weight_by_choice.keys())
        )
        reasoning = (
            f"PRM-score-weighted vote across {len(population)} answers.\n"
            f"Per-answer PRM scores:\n" + "\n".join(per_answer_lines) + "\n"
            f"Total PRM score by normalized choice: {totals}\n"
            f"Selected: {best_raw} (norm={best_norm}, total_score={best_weight:.3f})"
        )
        return Answer(reasoning=reasoning, choice=best_raw)


__all__ = ["PrmScoreVote"]

