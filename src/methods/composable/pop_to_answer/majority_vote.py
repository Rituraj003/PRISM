from __future__ import annotations

from collections import Counter
from typing import Sequence

from answer_verification import normalize_answer
from shared import QuestionType

from ..stages import Answer, PopulationToAnswer, StageContext


class MajorityVote(PopulationToAnswer):
    """
    Reduce population to final answer using majority voting.

    Returns the most common choice among the population.
    If there's a tie, picks the first one in alphabetical order.
    """

    @staticmethod
    def vote_majority(
        choices: Sequence[str | None],
        question_type: QuestionType,
    ) -> str | None:
        """Find majority choice from a list of choices with modality-aware tie-breaking."""
        vals: list[str] = [c for c in choices if c is not None]
        if not vals:
            return None

        counts: Counter[str] = Counter(vals)
        top = counts.most_common()

        if len(top) == 1 or (len(top) > 1 and top[0][1] > top[1][1]):
            return top[0][0]

        max_count = top[0][1]
        tied_choices = [choice for choice, count in top if count == max_count]

        if question_type == QuestionType.MCQ and all(
            c in ("a", "b", "c", "d") for c in tied_choices
        ):
            for c in ("a", "b", "c", "d"):
                if c in tied_choices:
                    return c

        for choice in vals:
            if choice in tied_choices:
                return choice

        return None

    async def __call__(self, context: StageContext, population: list[Answer]) -> Answer:
        if not population:
            return Answer(
                reasoning="No answers in population for majority vote",
                choice=None,
            )

        question_type = context.question_type

        # Count choices (ignore None)
        valid_choices = [ans.choice for ans in population if ans.choice is not None]

        if not valid_choices:
            return Answer(
                reasoning="No valid choices in population",
                choice=None,
            )

        if question_type == QuestionType.MATH:
            normalized_pairs = []
            for ans in population:
                if ans.choice is None:
                    continue
                normalized = normalize_answer(ans.choice, question_type=question_type)
                if normalized is None:
                    continue
                normalized_pairs.append((normalized, ans.choice))

            if normalized_pairs:
                counter = Counter(norm for norm, _ in normalized_pairs)
                most_common_norm, count = counter.most_common(1)[0]
                most_common_choice = None
                for norm, raw in normalized_pairs:
                    if norm == most_common_norm:
                        most_common_choice = raw
                        break
                if most_common_choice is None:
                    most_common_choice = most_common_norm

                vote_summary = ", ".join(
                    f"{choice}: {cnt}" for choice, cnt in counter.most_common()
                )
                total_votes = len(normalized_pairs)

                reasoning = (
                    f"Majority vote across {total_votes} answers.\n"
                    f"Vote distribution (normalized): {vote_summary}\n"
                    f"Selected: {most_common_choice} ({count}/{total_votes} votes)"
                )

                return Answer(reasoning=reasoning, choice=most_common_choice)

        # Find most common
        counter = Counter(valid_choices)
        most_common_choice = self.vote_majority(
            valid_choices,
            question_type=question_type,
        )
        if most_common_choice is None:
            return Answer(
                reasoning="No valid choices in population",
                choice=None,
            )
        count = counter[most_common_choice]

        # Build reasoning showing the vote distribution
        vote_summary = ", ".join(
            f"{choice}: {cnt}" for choice, cnt in counter.most_common()
        )

        reasoning = (
            f"Majority vote across {len(population)} answers.\n"
            f"Vote distribution: {vote_summary}\n"
            f"Selected: {most_common_choice} ({count}/{len(population)} votes)"
        )

        return Answer(reasoning=reasoning, choice=most_common_choice)


__all__ = ["MajorityVote"]
