from __future__ import annotations

import asyncio
from collections import Counter
from datetime import datetime, timezone
from typing import Any, Sequence, cast

from huggingface_hub import login as hf_login
from tqdm.asyncio import tqdm as atqdm

from data_sources import DatasetSpec, get_dataset, list_datasets
from methods import (
    MethodContext,
    get_create_population_stage,
    get_method,
    get_pop_to_answer_stage,
    get_pop_to_pop_stage,
    list_create_population_stages,
    list_methods,
    list_pop_to_answer_stages,
    list_pop_to_pop_stages,
)
from methods.composable import ComposableMethodConfig, register_composable_method
from shared import (
    DepthLogger,
    EvaluationExample,
    MethodResult,
    ResultLogger,
    SharedSettings,
    build_model,
)


async def run_method_on_example(
    runner,
    example: EvaluationExample,
    question_index: int,
) -> MethodResult:
    result = await runner(example, question_index)
    return result


def ensure_hf_login(token: str | None) -> None:
    if token:
        hf_login(token)


def resolve_datasets(settings: SharedSettings) -> list[DatasetSpec]:
    resolved: list[DatasetSpec] = []
    available = list_datasets()
    for name in settings.datasets:
        spec = get_dataset(name)
        if spec is None:
            raise SystemExit(
                f"Unknown dataset '{name}'. Available datasets: {', '.join(available)}"
            )
        resolved.append(spec)
    return resolved


def _instantiate_pop_to_pop_stage(
    pop_to_pop_class: type[Any], settings: SharedSettings
) -> Any:
    """
    Instantiate a PopulationToPopulation stage with appropriate hyperparameters.

    This function handles stage-specific hyperparameters by checking the class name
    and passing the appropriate arguments. This avoids type checking errors while
    maintaining runtime correctness.

    To add hyperparameters for a new stage:
    1. Add the hyperparameter fields to SharedSettings
    2. Add an elif branch here checking for the class name
    3. Set defaults and instantiate with those parameters
    """
    class_name = pop_to_pop_class.__name__

    if class_name == "Prism":
        # Set defaults for PRISM hyperparameters
        if settings.prism_temperature is None:
            settings.prism_temperature = 0.8
        if settings.prism_ess_threshold is None:
            settings.prism_ess_threshold = 0.5
        if settings.prism_acceptance_noise is None:
            settings.prism_acceptance_noise = 0.1

        # Cast to Any to avoid type checker errors with dynamic instantiation
        return cast(Any, pop_to_pop_class)(
            temperature=settings.prism_temperature,
            ess_threshold=settings.prism_ess_threshold,
            acceptance_noise=settings.prism_acceptance_noise,
        )

    elif class_name == "MadFollower":
        # Set default for follower_ratio
        if settings.follower_ratio is None:
            settings.follower_ratio = 0.3

        return cast(Any, pop_to_pop_class)(
            follower_ratio=settings.follower_ratio,
        )

    # Default: instantiate with no arguments
    return cast(Any, pop_to_pop_class)()


def build_composable_method_from_settings(settings: SharedSettings) -> None:
    """
    Build and register a composable method from stage settings.

    If create_population, pop_to_pop, and pop_to_answer are specified,
    creates a dynamic composable method named 'composable'.
    """
    if not settings.create_population:
        return  # Not using composable method

    # Get create population stage
    create_pop_class = get_create_population_stage(settings.create_population)
    if create_pop_class is None:
        available = ", ".join(list_create_population_stages())
        raise SystemExit(
            f"Unknown create_population stage '{settings.create_population}'. "
            f"Available: {available}"
        )

    # Get pop to answer stage
    if not settings.pop_to_answer:
        raise SystemExit(
            "pop_to_answer is required when using composable method. "
            f"Available: {', '.join(list_pop_to_answer_stages())}"
        )

    pop_to_answer_class = get_pop_to_answer_stage(settings.pop_to_answer)
    if pop_to_answer_class is None:
        available = ", ".join(list_pop_to_answer_stages())
        raise SystemExit(
            f"Unknown pop_to_answer stage '{settings.pop_to_answer}'. Available: {available}"
        )

    # Get optional pop to pop stage
    pop_to_pop_class = None
    if settings.pop_to_pop and settings.pop_to_pop.lower() != "none":
        pop_to_pop_class = get_pop_to_pop_stage(settings.pop_to_pop)
        if pop_to_pop_class is None:
            available = ", ".join(list_pop_to_pop_stages())
            raise SystemExit(
                f"Unknown pop_to_pop stage '{settings.pop_to_pop}'. Available: {available}"
            )

    # Build method name from stages
    method_name_parts = [settings.create_population]
    if pop_to_pop_class and settings.pop_to_pop:
        method_name_parts.append(settings.pop_to_pop)
    method_name_parts.append(settings.pop_to_answer)
    dynamic_method_name = "_".join(method_name_parts)

    pop_to_pop = None
    if pop_to_pop_class is not None:
        # Instantiate pop_to_pop stage with appropriate hyperparameters
        pop_to_pop = _instantiate_pop_to_pop_stage(pop_to_pop_class, settings)

    # Create and register the composable method
    config = ComposableMethodConfig(
        name=dynamic_method_name,
        create_population=create_pop_class(),
        pop_to_pop=pop_to_pop,
        pop_to_answer=pop_to_answer_class(),
        aliases=("composable",),
    )
    register_composable_method(config)

    # Update settings to use the new method
    settings.method = dynamic_method_name


def main() -> None:
    settings = SharedSettings()  # pyright: ignore[reportCallIssue]

    ensure_hf_login(settings.HF_TOKEN)

    build_composable_method_from_settings(settings)

    method_spec = get_method(settings.method)
    if method_spec is None:
        available_methods = ", ".join(list_methods())
        raise SystemExit(
            f"Unknown method '{settings.method}'. Available methods: {available_methods}"
        )

    dataset_specs = resolve_datasets(settings)
    if not dataset_specs:
        raise SystemExit(
            "No datasets selected. Use --datasets to specify at least one."
        )

    model, model_settings = build_model(settings)

    context = MethodContext(
        settings=settings,
        model=model,
        model_settings=model_settings,
        method_name=method_spec.name,
    )
    runner = method_spec.create_runner(context)

    result_logger = ResultLogger(settings.output_csv, settings)
    depth_logger = DepthLogger(settings.depth_metrics_csv, settings)
    run_id = datetime.now(tz=timezone.utc).isoformat()

    async def evaluate_all() -> tuple[int, int, Counter[str]]:
        total_examples = 0
        total_responses = 0
        dataset_counts: Counter[str] = Counter()

        for spec in dataset_specs:
            examples: Sequence[EvaluationExample] = spec.loader(settings)

            if settings.question_ids:
                wanted = set(settings.question_ids)
                examples = [ex for ex in examples if ex.question_id in wanted]
                if not examples:
                    print(
                        f"[main] No examples matched question_ids for {spec.name}: "
                        f"{', '.join(settings.question_ids)}"
                    )

            # Apply start_index to skip already processed examples
            if settings.start_index > 0:
                skipped = min(settings.start_index, len(examples))
                examples = examples[settings.start_index :]
                if skipped > 0:
                    print(
                        f"[main] Skipping first {skipped} examples from {spec.name} (start_index={settings.start_index})"
                    )

            dataset_counts[spec.name] += len(examples)
            # Async-friendly progress bar for examples within this dataset
            for example in atqdm(
                examples,
                desc=f"{spec.name}",
                unit="ex",
                leave=False,
            ):
                method_result = await run_method_on_example(
                    runner,
                    example,
                    example.question_index,
                )

                for response in method_result.responses:
                    result_logger.log_response(run_id, example, response)
                    total_responses += 1

                # Log detailed depth responses if available
                depth_responses = method_result.metadata.get("depth_responses", [])
                if depth_responses:
                    for depth_response in depth_responses:
                        depth_logger.log_response(run_id, example, depth_response)
                else:
                    depth_logger.log(
                        run_id,
                        settings,
                        example,
                        method_spec.name,
                        method_result.depth_events,
                    )

                total_examples += 1

        return total_examples, total_responses, dataset_counts

    total_examples, total_responses, dataset_counts = asyncio.run(evaluate_all())

    datasets_summary = ", ".join(
        f"{name}: {count}" for name, count in dataset_counts.items()
    )
    print(
        f"[main] Method={method_spec.name} Examples={total_examples} Responses={total_responses} "
        f"Datasets=[{datasets_summary}] Output={settings.output_csv}"
    )


if __name__ == "__main__":
    main()
