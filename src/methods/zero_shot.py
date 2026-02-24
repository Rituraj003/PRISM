from __future__ import annotations

import asyncio

from pydantic_ai import Agent

from answer_verification import (
    check_answer_correctness,
    normalize_answer,
    parse_answer_with_reasoning,
)
from methods.composable.stages import Answer
from prompts import ZERO_SHOT_MATH, ZERO_SHOT_MCQ, ZERO_SHOT_TEXT
from shared import (
    DepthEvent,
    EvaluationExample,
    EvaluationResponse,
    MethodResult,
    QuestionType,
    TokenUsage,
    extract_usage_tokens,
)

from . import MethodContext, MethodRunner, MethodSpec, register_method


def parse_answer(response: str, question_type: QuestionType) -> Answer:
    """Parse answer from response using unified extraction."""
    reasoning, choice_str = parse_answer_with_reasoning(
        response, question_type=question_type
    )
    return Answer(reasoning, choice_str)


def create_runner(context: MethodContext) -> MethodRunner:
    async def runner(
        example: EvaluationExample,
        question_index: int,
    ) -> MethodResult:
        # Detect the prompt family to use for this question
        question_type = example.question_type
        prompt_family = question_type.prompt_family
        if prompt_family == "text":
            prompts = ZERO_SHOT_TEXT
        elif prompt_family == "mcq":
            prompts = ZERO_SHOT_MCQ
        else:
            prompts = ZERO_SHOT_MATH

        agent: Agent[None, str] = Agent(
            context.model,
            system_prompt=prompts["solver"],
            model_settings=context.model_settings,
        )

        async def solve(
            prompt: str, question_type: QuestionType
        ) -> tuple[Answer, TokenUsage]:
            result = await agent.run(f"Problem:\n{prompt}")
            usage = extract_usage_tokens(result.usage())
            return parse_answer(result.output, question_type=question_type), usage

        responses: list[EvaluationResponse] = []
        depth_events: list[DepthEvent] = []

        # Run samples concurrently for better throughput
        prompt = example.to_prompt()

        async def run_one(sample_idx: int) -> tuple[int, Answer, TokenUsage]:
            try:
                ans, usage = await solve(prompt, question_type=question_type)
            except Exception as exc:  # pragma: no cover - defensive
                ans = Answer(
                    reasoning=f"Exception during zero-shot run: {exc}",
                    choice=None,
                )
                usage = TokenUsage()
            return sample_idx, ans, usage

        tasks = [run_one(i) for i in range(context.settings.samples)]
        results = await asyncio.gather(*tasks, return_exceptions=False)

        # Get correct answer for comparison
        correct_answer = example.correct_answer

        # Preserve deterministic ordering by sample_idx when logging
        for sample_idx, answer, usage in sorted(results, key=lambda x: x[0]):
            normalized = normalize_answer(answer.choice, question_type=question_type)
            is_correct = check_answer_correctness(
                answer.choice,
                correct_answer,
                question_type=question_type,
                example=example,
            )

            responses.append(
                EvaluationResponse(
                    method=context.method_name,
                    dataset=example.dataset,
                    question_id=example.question_id,
                    question_index=question_index,
                    chain_id=sample_idx,
                    step=0,
                    reasoning=answer.reasoning,
                    raw_answer=answer.choice,
                    normalized_answer=normalized,
                    predicted_label=normalized,
                    is_correct=is_correct,
                    total_input_tokens=usage.total_input_tokens,
                    total_output_tokens=usage.total_output_tokens,
                    metadata={"sample_index": sample_idx},
                )
            )
            depth_events.append(
                DepthEvent(depth=0, answer=normalized, chain_id=sample_idx)
            )

        return MethodResult(responses=responses, depth_events=depth_events)

    return runner


register_method(
    MethodSpec(
        name="zero-shot",
        create_runner=create_runner,
        aliases=("zero_shot", "zs"),
    )
)
