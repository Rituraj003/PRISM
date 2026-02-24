from __future__ import annotations

import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from tqdm import tqdm

from answer_verification import check_answer_correctness, normalize_answer
from shared import (
    DepthEvent,
    EvaluationExample,
    EvaluationResponse,
    MethodResult,
    TokenUsage,
    sanitized_settings_dump,
)

from .stages import (
    Answer,
    CreatePopulation,
    PopulationToAnswer,
    PopulationToPopulation,
    StageContext,
)

# Increase CSV field limit to handle large code blocks/traces in logs
try:
    csv.field_size_limit(sys.maxsize)
except OverflowError:
    # On some systems (e.g. Windows), sys.maxsize might exceed C long
    csv.field_size_limit(2147483647)

if TYPE_CHECKING:
    from shared import SharedSettings

    from .. import MethodContext, MethodRunner


@dataclass
class ComposableMethodConfig:
    """Configuration for a composable method."""

    name: str
    create_population: CreatePopulation
    pop_to_pop: PopulationToPopulation | None
    pop_to_answer: PopulationToAnswer
    aliases: tuple[str, ...] = ()


POPULATION_SIGNATURE_IGNORE_SETTINGS = {
    "method",
    "pop_to_answer",
    "run_desc",
    "output_csv",
    "depth_metrics_csv",
    # Run selection / orchestration controls (do not affect per-question outputs)
    "start_index",
    "question_ids",
    "pop_to_pop",
    "create_population",
    "model_url",
    "verifier_model_url",
}
SEED_SIGNATURE_IGNORE_SETTINGS = {
    # Run description (varies between sweeps but doesn't affect seed generation)
    "run_desc",
    # Run selection (doesn't affect seed content)
    "start_index",
    # Dataset selection (seed reuse should work across different dataset configs)
    "max_samples_per_dataset",
    # Downstream stage configuration (seed generation doesn't depend on these)
    "pop_to_pop",  # Seed is created before pop_to_pop stage
    "pop_to_answer",  # Seed is created before pop_to_answer stage
    "prism_temperature",
    "prism_ess_threshold",
    "prism_acceptance_noise",
    "follower_ratio",
}


def _build_settings_signature(
    settings: "SharedSettings",
    extra_ignore_keys: set[str] | None = None,
) -> dict[str, Any]:
    payload = json.loads(sanitized_settings_dump(settings))
    ignore_keys = set(POPULATION_SIGNATURE_IGNORE_SETTINGS)
    if extra_ignore_keys:
        ignore_keys.update(extra_ignore_keys)
    for key in ignore_keys:
        payload.pop(key, None)
    return payload


def _build_population_signature(
    example: EvaluationExample,
    question_index: int,
    create_stage_name: str,
    pop_to_pop_name: str | None,
    depth_iterations: int,
    settings_signature: dict[str, Any],
) -> dict[str, Any]:
    return {
        "dataset": example.dataset,
        "question_id": example.question_id,
        "question_index": question_index,
        "create_stage": create_stage_name,
        "pop_to_pop_stage": pop_to_pop_name,
        "depth_iterations": depth_iterations,
        "settings": settings_signature,
    }


def _normalize_signature(
    sig: dict[str, Any] | None,
    ignore_settings_keys: set[str] | None = None,
) -> dict[str, Any] | None:
    """Normalize a (seed/population) signature loaded from CSV for comparison.

    Older runs may include fields we now ignore (e.g., URL fields) within the
    nested settings dict. Strip those here so equality checks succeed across
    runs from different servers.
    """
    if not isinstance(sig, dict):
        return sig
    if "model_url" in sig:
        sig = dict(sig)
        sig.pop("model_url", None)
    if "verifier_model_url" in sig:
        sig = dict(sig)
        sig.pop("verifier_model_url", None)
    settings = sig.get("settings")
    if isinstance(settings, dict):
        ignore_keys = {"model_url", "verifier_model_url"}
        if ignore_settings_keys:
            ignore_keys.update(ignore_settings_keys)
        for k in ignore_keys:
            if k in settings:
                settings = dict(settings)
                settings.pop(k, None)
                sig = dict(sig)
                sig["settings"] = settings
    return sig


@dataclass
class DepthRow:
    dataset: str
    question_id: str
    question_index: int
    run_id: str
    step: int
    chain_id: int | None
    reasoning: str
    raw_answer: str | None
    population_signature: dict[str, Any] | None
    seed_signature: dict[str, Any] | None
    settings_dict: dict[str, Any] | None  # For reconstructing seed signature
    response_metadata: dict[str, Any]
    total_input_tokens: int
    total_output_tokens: int
    secondary_input_tokens: int
    secondary_output_tokens: int

    @property
    def is_final(self) -> bool:
        return bool(self.response_metadata.get("final"))

    @classmethod
    def from_csv_row(cls, row: dict[str, str]) -> DepthRow | None:
        dataset = row.get("dataset")
        question_id = row.get("question_id")
        if not dataset or not question_id:
            return None
        run_id = row.get("run_id", "")
        if not run_id:
            return None

        question_index = _parse_int_field(row.get("question_index"), default=0)
        if question_index is None:
            return None

        step = _parse_int_field(row.get("step"), default=0)
        if step is None:
            return None

        chain_id_raw = row.get("chain_id")
        chain_id: int | None
        if chain_id_raw is None or chain_id_raw == "":
            chain_id = None
        else:
            chain_id = _parse_int_field(chain_id_raw)
            if chain_id is None:
                return None

        metadata_json = row.get("metadata_json", "")
        metadata = {}
        if metadata_json:
            try:
                # Handle cases where metadata_json might be a primitive (like "0")
                parsed = json.loads(metadata_json)
                if isinstance(parsed, dict):
                    metadata = parsed
            except json.JSONDecodeError as e:
                # Skip corrupted JSON entries
                print(
                    f"[Warning] Skipping row with corrupted JSON at position {e.pos}: {e.msg}"
                )
                return None

        response_meta = metadata.get("response")
        if not isinstance(response_meta, dict):
            response_meta = {}

        population_signature = response_meta.get("population_signature")
        if not isinstance(population_signature, dict):
            population_signature = None
        else:
            population_signature = _normalize_signature(population_signature)

        seed_signature = response_meta.get("seed_signature")
        if not isinstance(seed_signature, dict):
            if population_signature is not None:
                derived_signature = dict(population_signature)
                derived_signature["pop_to_pop_stage"] = None
                derived_signature["depth_iterations"] = 0
                seed_signature = _normalize_signature(
                    derived_signature,
                    ignore_settings_keys=SEED_SIGNATURE_IGNORE_SETTINGS,
                )
            else:
                seed_signature = None
        else:
            seed_signature = _normalize_signature(
                seed_signature,
                ignore_settings_keys=SEED_SIGNATURE_IGNORE_SETTINGS,
            )

        reasoning = row.get("reasoning", "") or ""
        raw_answer = row.get("raw_answer") or None

        # Parse settings for reconstructing seed signature when needed
        settings_json = row.get("settings_json", "")
        settings_dict: dict[str, Any] | None = None
        if settings_json:
            try:
                settings_dict = json.loads(settings_json)
            except json.JSONDecodeError:
                pass

        total_input_tokens = _parse_token_field(row.get("total_input_tokens"))
        total_output_tokens = _parse_token_field(row.get("total_output_tokens"))
        secondary_input_tokens = _parse_token_field(row.get("secondary_input_tokens"))
        secondary_output_tokens = _parse_token_field(row.get("secondary_output_tokens"))

        return cls(
            dataset=dataset,
            question_id=question_id,
            question_index=question_index,
            run_id=run_id,
            step=step,
            chain_id=chain_id,
            reasoning=reasoning,
            raw_answer=raw_answer,
            population_signature=population_signature,
            seed_signature=seed_signature,
            settings_dict=settings_dict,
            response_metadata=response_meta,
            total_input_tokens=total_input_tokens,
            total_output_tokens=total_output_tokens,
            secondary_input_tokens=secondary_input_tokens,
            secondary_output_tokens=secondary_output_tokens,
        )

    def matches_problem(self, signature: dict[str, Any]) -> bool:
        return (
            self.dataset == signature["dataset"]
            and self.question_id == signature["question_id"]
            and self.question_index == signature["question_index"]
        )

    def matches_seed_signature(self, expected_signature: dict[str, Any]) -> bool:
        """Check if this row can be used as seed population.

        Simple matching on core parameters:
        - Same question (dataset + question_id)
        - Same create_population stage (e.g., SampleN)
        - Same core settings: model_name, width, depth, samples, temperature
        """
        # Must have settings to compare
        if self.settings_dict is None:
            return False

        exp_settings = expected_signature.get("settings", {})
        csv_settings = self.settings_dict

        # Check question matches
        if self.dataset != expected_signature.get("dataset"):
            return False
        if self.question_id != expected_signature.get("question_id"):
            return False

        # Check create_population stage matches
        csv_create = csv_settings.get("create_population", "")
        exp_create = expected_signature.get("create_stage", "")
        # Convert csv snake_case to PascalCase for comparison
        csv_create_class = _config_to_class_name(csv_create) if csv_create else ""
        if csv_create_class != exp_create:
            return False

        # Check core settings that affect seed population
        core_keys = ["model_name", "width", "samples", "temperature"]
        for key in core_keys:
            if csv_settings.get(key) != exp_settings.get(key):
                return False

        return True


def _config_to_class_name(config_name: str) -> str:
    """Convert config name to class name (sample_n -> SampleN, etc.)"""
    # Handle snake_case to PascalCase conversion
    parts = config_name.split("_")
    return "".join(word.capitalize() for word in parts)


def _parse_token_field(value: Any) -> int:
    if value is None:
        return 0
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if value.is_integer():
            return int(value)
        return 0
    raw = str(value).strip()
    if not raw:
        return 0
    sign = 1
    if raw[0] in "+-":
        sign = -1 if raw[0] == "-" else 1
        raw = raw[1:]
    if raw.isdigit():
        return sign * int(raw)
    return 0


def _parse_int_field(value: Any, default: int | None = None) -> int | None:
    if value is None:
        return default
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value) if value.is_integer() else None
    raw = str(value).strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        try:
            num = float(raw)
        except ValueError:
            return None
        return int(num) if num.is_integer() else None


def _load_cached_populations(
    csv_path: Path,
    population_signature: dict[str, Any],
) -> tuple[list[list[Answer]], tuple[int, int, int, int] | None] | None:
    if not csv_path.exists():
        return None

    def group_key(signature: dict[str, Any]) -> tuple[str, str, str | None, int, str]:
        settings = signature.get("settings")
        settings_payload = settings if isinstance(settings, dict) else {}
        return (
            str(signature.get("dataset", "")),
            str(signature.get("create_stage", "")),
            signature.get("pop_to_pop_stage"),
            int(signature.get("depth_iterations", 0)),
            json.dumps(settings_payload, sort_keys=True),
        )

    target_group_key = group_key(population_signature)

    runs: dict[str, dict[int, list[tuple[int, Answer]]]] = {}
    run_step_usage: dict[str, dict[int, tuple[int, int, int, int]]] = {}
    run_final_usage: dict[str, tuple[int, int, int, int]] = {}
    run_completed_questions: dict[str, set[str]] = {}

    with open(csv_path, newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            depth_row = DepthRow.from_csv_row(row)
            if depth_row is None:
                continue

            if depth_row.population_signature is None:
                continue
            if group_key(depth_row.population_signature) != target_group_key:
                continue

            if depth_row.is_final:
                run_completed_questions.setdefault(depth_row.run_id, set()).add(
                    depth_row.question_id
                )
                if depth_row.matches_problem(population_signature):
                    if depth_row.population_signature == population_signature:
                        run_final_usage[depth_row.run_id] = (
                            depth_row.total_input_tokens,
                            depth_row.total_output_tokens,
                            depth_row.secondary_input_tokens,
                            depth_row.secondary_output_tokens,
                        )
                continue
            if not depth_row.matches_problem(population_signature):
                continue
            if depth_row.population_signature != population_signature:
                continue
            if depth_row.chain_id is None:
                continue

            answer = Answer(
                reasoning=depth_row.reasoning,
                choice=depth_row.raw_answer,
            )
            run_steps = runs.setdefault(depth_row.run_id, {})
            run_steps.setdefault(depth_row.step, []).append(
                (depth_row.chain_id, answer)
            )
            run_usage = run_step_usage.setdefault(depth_row.run_id, {})
            run_usage[depth_row.step] = (
                depth_row.total_input_tokens,
                depth_row.total_output_tokens,
                depth_row.secondary_input_tokens,
                depth_row.secondary_output_tokens,
            )

    if not runs:
        return None

    expected_steps = population_signature["depth_iterations"] + 1
    expected_step_indices = set(range(expected_steps))
    # Prefer runs that completed more questions (skips partial runs).
    for run_id in sorted(
        runs.keys(),
        key=lambda rid: (len(run_completed_questions.get(rid, set())), rid),
        reverse=True,
    ):
        step_to_answers = runs[run_id]
        if set(step_to_answers.keys()) != expected_step_indices:
            continue

        populations: list[list[Answer]] = []
        valid = True
        for step in range(expected_steps):
            answers_with_chain = step_to_answers.get(step, [])
            if not answers_with_chain:
                valid = False
                break
            answers_with_chain.sort(key=lambda item: item[0])
            # Require contiguous chain ids starting at 0 to ensure consistent population size
            for expected_chain, (chain_id, _) in enumerate(answers_with_chain):
                if chain_id != expected_chain:
                    valid = False
                    break
            if not valid:
                break
            populations.append([answer for _, answer in answers_with_chain])

        if not valid:
            continue

        step_usage = run_step_usage.get(run_id, {})
        baseline_step = len(populations) - 1
        usage = step_usage.get(baseline_step)
        if usage is None:
            usage = run_final_usage.get(run_id)

        return populations, usage

    return None


def _load_seed_population(
    csv_path: Path,
    expected_signature: dict[str, Any],
) -> tuple[list[Answer], tuple[int, int, int, int] | None] | None:
    if not csv_path.exists():
        return None

    runs: dict[str, list[tuple[int, Answer]]] = {}
    run_usage: dict[str, tuple[int, int, int, int]] = {}

    try:
        with open(csv_path, newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                depth_row = DepthRow.from_csv_row(row)
                if depth_row is None:
                    continue

                if not depth_row.matches_problem(expected_signature):
                    continue

                if depth_row.step != 0:
                    continue

                if not depth_row.matches_seed_signature(expected_signature):
                    continue
                if depth_row.chain_id is None:
                    continue

                answer = Answer(
                    reasoning=depth_row.reasoning,
                    choice=depth_row.raw_answer,
                )
                # Note: Allow empty reasoning - seed population just needs the answers
                run_answers = runs.setdefault(depth_row.run_id, [])
                run_answers.append((depth_row.chain_id, answer))
                run_usage[depth_row.run_id] = (
                    depth_row.total_input_tokens,
                    depth_row.total_output_tokens,
                    depth_row.secondary_input_tokens,
                    depth_row.secondary_output_tokens,
                )
    except Exception:
        return None

    if not runs:
        return None

    for run_id in sorted(runs.keys(), reverse=True):
        answers_with_chain = runs[run_id]
        if not answers_with_chain:
            continue

        # Deduplicate by chain_id (keep first occurrence)
        seen_chains: dict[int, Answer] = {}
        for chain_id, answer in answers_with_chain:
            if chain_id not in seen_chains:
                seen_chains[chain_id] = answer

        # Sort by chain_id
        sorted_chains = sorted(seen_chains.items(), key=lambda x: x[0])

        # Validate that chain IDs are contiguous starting from 0
        valid = True
        for expected_chain, (chain_id, _) in enumerate(sorted_chains):
            if chain_id != expected_chain:
                valid = False
                break

        if not valid or not sorted_chains:
            continue

        usage = run_usage.get(run_id)
        return [answer for _, answer in sorted_chains], usage

    return None


def create_composable_runner(
    config: ComposableMethodConfig,
) -> Any:  # Returns MethodRunner
    """
    Create a method runner that composes the three stages.

    Flow:
    1. Create initial population
    2. Optionally transform population (can be repeated for depth > 0)
    3. Reduce population to final answer
    """

    async def runner(
        example: EvaluationExample,
        question_index: int,
        context: MethodContext,
    ) -> MethodResult:
        # Detect the question modality
        question_type = example.question_type
        prm_temp = context.settings.prm_temperature or 0.0
        stage_context = StageContext(
            settings=context.settings,
            model=context.model,
            model_settings=context.model_settings,
            example=example,
            question_index=question_index,
            question_type=question_type,
            prm_temperature=prm_temp,
        )

        responses: list[EvaluationResponse] = []
        depth_events: list[DepthEvent] = []

        create_stage_name = type(config.create_population).__name__
        pop_to_pop_name = (
            type(config.pop_to_pop).__name__ if config.pop_to_pop is not None else None
        )
        depth_iterations = (
            max(0, context.settings.depth) if config.pop_to_pop is not None else 0
        )
        settings_signature = _build_settings_signature(context.settings)
        seed_settings_signature = _build_settings_signature(
            context.settings,
            extra_ignore_keys=SEED_SIGNATURE_IGNORE_SETTINGS,
        )
        population_signature = _build_population_signature(
            example=example,
            question_index=question_index,
            create_stage_name=create_stage_name,
            pop_to_pop_name=pop_to_pop_name,
            depth_iterations=depth_iterations,
            settings_signature=settings_signature,
        )
        depth_csv_path = Path(context.settings.depth_metrics_csv)

        seed_signature = _build_population_signature(
            example=example,
            question_index=question_index,
            create_stage_name=create_stage_name,
            pop_to_pop_name=None,
            depth_iterations=0,
            settings_signature=seed_settings_signature,
        )

        cached_result = _load_cached_populations(depth_csv_path, population_signature)

        cache_mode = "none"

        if cached_result is not None:
            cached_populations, cached_usage = cached_result
            all_populations = cached_populations
            population = cached_populations[-1] if cached_populations else []
            cache_mode = "full"
            if cached_usage is not None:
                stage_context.token_usage.add(
                    TokenUsage(
                        total_input_tokens=cached_usage[0],
                        total_output_tokens=cached_usage[1],
                        secondary_input_tokens=cached_usage[2],
                        secondary_output_tokens=cached_usage[3],
                    )
                )
        else:
            seed_result = _load_seed_population(depth_csv_path, seed_signature)

            if seed_result is not None:
                seed_population, seed_usage = seed_result
                population = seed_population
                all_populations = [population]
                cache_mode = "seed"
                if seed_usage is not None:
                    stage_context.token_usage.add(
                        TokenUsage(
                            total_input_tokens=seed_usage[0],
                            total_output_tokens=seed_usage[1],
                            secondary_input_tokens=seed_usage[2],
                            secondary_output_tokens=seed_usage[3],
                        )
                    )

                if config.pop_to_pop is not None:
                    for _ in range(depth_iterations):
                        population = await config.pop_to_pop(stage_context, population)
                        all_populations.append(population)
            else:
                population = await config.create_population(stage_context)
                all_populations = [population]

                if config.pop_to_pop is not None:
                    for _ in range(depth_iterations):
                        population = await config.pop_to_pop(stage_context, population)
                        all_populations.append(population)

        cache_used = cache_mode != "none"
        if cache_mode == "full":
            tqdm.write(
                f"[composable] Using cached population (mode=full) for "
                f"{example.dataset}/{example.question_id}; steps={len(all_populations)}",
                file=sys.stderr,
            )
        elif cache_mode == "seed":
            tqdm.write(
                f"[composable] Reusing seed population for "
                f"{example.dataset}/{example.question_id}; regenerating downstream stages",
                file=sys.stderr,
            )
        else:
            tqdm.write(
                f"[composable] No cached population for "
                f"{example.dataset}/{example.question_id}; running all stages",
                file=sys.stderr,
            )

        # Stage 3: Reduce to final answer
        final_answer = await config.pop_to_answer(stage_context, population)

        # Take usage snapshot after all stages are complete
        usage_snapshot = stage_context.token_usage.clone()

        # Get correct answer for comparison
        correct_answer = example.correct_answer

        # Log responses for all intermediate populations to depth_events (with full details)
        depth_responses = []
        for step, pop in enumerate(all_populations):
            for chain_id, answer in enumerate(pop):
                normalized = normalize_answer(
                    answer.choice, question_type=question_type
                )
                is_correct = check_answer_correctness(
                    answer.choice,
                    correct_answer,
                    question_type=question_type,
                    example=example,
                )

                depth_responses.append(
                    EvaluationResponse(
                        method=context.method_name,
                        dataset=example.dataset,
                        question_id=example.question_id,
                        question_index=question_index,
                        chain_id=chain_id,
                        step=step,
                        reasoning=answer.reasoning,
                        raw_answer=answer.choice,
                        normalized_answer=normalized,
                        predicted_label=normalized,
                        is_correct=is_correct,
                        total_input_tokens=usage_snapshot.total_input_tokens,
                        total_output_tokens=usage_snapshot.total_output_tokens,
                        secondary_input_tokens=usage_snapshot.secondary_input_tokens,
                        secondary_output_tokens=usage_snapshot.secondary_output_tokens,
                        verifier_model_name=context.settings.verifier_model_name,
                        metadata={
                            "step": step,
                            "chain_id": chain_id,
                            "population_signature": population_signature,
                            "seed_signature": seed_signature,
                        },
                    )
                )
                depth_events.append(
                    DepthEvent(depth=step, answer=normalized, chain_id=chain_id)
                )

        final_normalized = normalize_answer(
            final_answer.choice, question_type=question_type
        )
        final_is_correct = check_answer_correctness(
            final_answer.choice,
            correct_answer,
            question_type=question_type,
            example=example,
        )

        # Add final answer to depth responses too
        depth_responses.append(
            EvaluationResponse(
                method=context.method_name,
                dataset=example.dataset,
                question_id=example.question_id,
                question_index=question_index,
                chain_id=None,
                step=len(all_populations),
                reasoning=final_answer.reasoning,
                raw_answer=final_answer.choice,
                normalized_answer=final_normalized,
                predicted_label=final_normalized,
                is_correct=final_is_correct,
                total_input_tokens=usage_snapshot.total_input_tokens,
                total_output_tokens=usage_snapshot.total_output_tokens,
                secondary_input_tokens=usage_snapshot.secondary_input_tokens,
                secondary_output_tokens=usage_snapshot.secondary_output_tokens,
                verifier_model_name=context.settings.verifier_model_name,
                metadata={
                    "final": True,
                    "population_signature": population_signature,
                    "seed_signature": seed_signature,
                },
            )
        )

        # Only the final answer goes to shared-results.csv
        responses = [
            EvaluationResponse(
                method=context.method_name,
                dataset=example.dataset,
                question_id=example.question_id,
                question_index=question_index,
                chain_id=None,
                step=len(all_populations),
                reasoning=final_answer.reasoning,
                raw_answer=final_answer.choice,
                normalized_answer=final_normalized,
                predicted_label=final_normalized,
                is_correct=final_is_correct,
                total_input_tokens=usage_snapshot.total_input_tokens,
                total_output_tokens=usage_snapshot.total_output_tokens,
                secondary_input_tokens=usage_snapshot.secondary_input_tokens,
                secondary_output_tokens=usage_snapshot.secondary_output_tokens,
                verifier_model_name=context.settings.verifier_model_name,
                metadata={
                    "final": True,
                    "population_signature": population_signature,
                    "seed_signature": seed_signature,
                },
            )
        ]

        return MethodResult(
            responses=responses,
            depth_events=depth_events,
            metadata={
                "depth_responses": depth_responses,
                "population_cache_used": cache_used,
                "population_cache_mode": cache_mode,
                "token_usage": {
                    "total_input_tokens": usage_snapshot.total_input_tokens,
                    "total_output_tokens": usage_snapshot.total_output_tokens,
                    "secondary_input_tokens": usage_snapshot.secondary_input_tokens,
                    "secondary_output_tokens": usage_snapshot.secondary_output_tokens,
                },
            },
        )

    # Wrap to match MethodRunner signature
    async def wrapped_runner(
        example: EvaluationExample,
        question_index: int,
    ) -> MethodResult:
        # Get context from closure (will be set by create_runner)
        return await runner(example, question_index, wrapped_runner._context)  # pyright: ignore[reportFunctionMemberAccess]

    return wrapped_runner


def register_composable_method(config: ComposableMethodConfig) -> None:
    """Register a composable method."""
    from .. import MethodSpec, register_method

    def create_runner(context: MethodContext) -> MethodRunner:
        runner = create_composable_runner(config)
        runner._context = context  # pyright: ignore[reportAttributeAccessIssue]
        return runner

    register_method(
        MethodSpec(
            name=config.name,
            create_runner=create_runner,
            aliases=config.aliases,
        )
    )


__all__ = [
    "ComposableMethodConfig",
    "create_composable_runner",
    "register_composable_method",
]
