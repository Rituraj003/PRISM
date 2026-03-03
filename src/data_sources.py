from __future__ import annotations

import random
import re
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, Sequence, cast

from datasets import (
    Dataset,
    DatasetDict,
    IterableDataset,
    IterableDatasetDict,
    load_dataset,
)

from shared import (
    EvaluationExample,
    SharedSettings,
)


@dataclass
class DatasetSpec:
    name: str
    loader: Callable[[SharedSettings], Sequence[EvaluationExample]]
    aliases: tuple[str, ...] = ()


_DATASETS: Dict[str, DatasetSpec] = {}


def register_dataset(spec: DatasetSpec) -> DatasetSpec:
    keys = {spec.name.lower(), *(alias.lower() for alias in spec.aliases)}
    for key in keys:
        _DATASETS[key] = spec
    return spec


def get_dataset(name: str) -> DatasetSpec | None:
    return _DATASETS.get(name.lower())


def list_datasets() -> list[str]:
    return sorted({spec.name for spec in _DATASETS.values()})


# ----------------------------------------------------------------------------
# Multiple-choice (GPQA-style)
# ----------------------------------------------------------------------------


def format_question_block(question: str, choices: Sequence[str]) -> str:
    labels = ["a", "b", "c", "d"]
    lines = [question]
    for lbl, choice in zip(labels, choices):
        lines.append(f"\n{lbl}) {choice}")
    return "\n".join(lines)


def _strip_choice_prefix(choice: str) -> str:
    """Normalize choices like 'a) text' -> 'text' for consistent indexing."""
    if len(choice) > 2 and choice[1] == ")" and choice[0].lower() in "abcd":
        return choice.split(")", 1)[1].strip()
    return choice


def build_multiple_choice_examples(
    ds: DatasetDict,
    max_samples: int,
    seed: int,
    dataset_label: str,
) -> list[EvaluationExample]:
    train = ds["train"]
    questions = cast(list[str], train["Question"])
    correct_answers = cast(list[str], train["Correct Answer"])
    answers_1 = cast(list[str], train["Incorrect Answer 1"])
    answers_2 = cast(list[str], train["Incorrect Answer 2"])
    answers_3 = cast(list[str], train["Incorrect Answer 3"])

    rng = random.Random(seed)
    items: list[EvaluationExample] = []
    for idx, (q, corr, a1, a2, a3) in enumerate(
        zip(questions, correct_answers, answers_1, answers_2, answers_3)
    ):
        # Normalize all options before shuffling so correct-answer lookup is stable.
        normalized_choices = [_strip_choice_prefix(c) for c in [corr, a1, a2, a3]]
        normalized_correct = normalized_choices[0]
        choices = list(normalized_choices)
        rng.shuffle(choices)
        correct_label = "abcd"[choices.index(normalized_correct)]
        block = format_question_block(q, choices)
        metadata = {
            "choice_order": {lbl: text for lbl, text in zip("abcd", choices)},
            "original_correct_answer": corr,
            "normalized_correct_answer": normalized_correct,
        }
        items.append(
            EvaluationExample(
                dataset=dataset_label,
                question_id=str(idx),
                question_index=idx,
                question_block=block,
                choices=tuple(choices),
                correct_answer=correct_label,  # Store the letter (a-d) as the answer
                metadata=metadata,
            )
        )
        if len(items) >= max_samples:
            break
    return items


def load_gpqa_examples(settings: SharedSettings) -> Sequence[EvaluationExample]:
    hf_id, config = ("Idavidrein/gpqa", "gpqa_diamond")
    ds = load_dataset(hf_id, config)
    assert isinstance(ds, DatasetDict), "Expected dataset to be a DatasetDict"
    return build_multiple_choice_examples(
        ds,
        settings.max_samples_per_dataset,
        settings.seed,
        f"{hf_id}/{config}",
    )


register_dataset(
    DatasetSpec(
        name="gpqa",
        loader=load_gpqa_examples,
        aliases=("multiple_choice", "mc"),
    )
)


# ----------------------------------------------------------------------------
# Math competition datasets (free-form numeric/latex answers)
# ----------------------------------------------------------------------------


@dataclass
class BenchmarkProblem:
    problem: str
    answer: str
    dataset: str
    problem_idx: int | None = None
    problem_type: list[str] | None = None


MATH_DATASETS = {
    "hmmt": ("MathArena/hmmt_feb_2025", "HMMT"),
    "aime": ("MathArena/aime_2025", "AIME"),
}


def _iter_dataset_split(
    dataset: DatasetDict | Dataset | IterableDatasetDict | IterableDataset,
    split: str | None,
) -> Iterable[Any]:
    if isinstance(dataset, (DatasetDict, IterableDatasetDict)):
        if split and split in dataset:
            return dataset[split]
        fallback = next(iter(dataset.keys()))
        return dataset[fallback]
    elif isinstance(dataset, Dataset):
        return dataset
    else:  # IterableDataset
        return dataset


def _load_math_dataset(
    hf_id: str, label: str, limit: int | None
) -> list[BenchmarkProblem]:
    try:
        raw = load_dataset(hf_id)
    except Exception as exc:  # pragma: no cover - defensive
        print(f"[datasets] Error loading {hf_id}: {exc}")
        return []

    iterable = _iter_dataset_split(raw, "train")
    problems: list[BenchmarkProblem] = []
    for item in iterable:
        problems.append(
            BenchmarkProblem(
                problem=item.get("problem", ""),
                answer=str(item.get("answer", "")),
                dataset=label,
                problem_idx=item.get("problem_idx"),
                problem_type=item.get("problem_type", []),
            )
        )
        if limit is not None and len(problems) >= limit:
            break
    return problems


MATH_NORMALIZE_FRACTION = re.compile(r"\\frac\{([^}]+)\}\{([^}]+)\}")
MATH_NORMALIZE_SQRT = re.compile(r"\\sqrt\{([^}]+)\}")
MATH_REMOVE_LATEX_WS = re.compile(r"\\\s*")


def normalize_math_answer(ans: str | None) -> str:
    if not ans:
        return ""

    value = ans.strip().replace("$", "")
    value = MATH_NORMALIZE_FRACTION.sub(r"(\1)/(\2)", value)
    value = MATH_NORMALIZE_SQRT.sub(r"sqrt(\1)", value)
    value = value.replace(r"\left", "").replace(r"\right", "")
    value = MATH_REMOVE_LATEX_WS.sub("", value)
    return value


def load_math_examples(
    settings: SharedSettings, dataset_key: str
) -> Sequence[EvaluationExample]:
    hf_id, label = MATH_DATASETS[dataset_key]
    limit = settings.max_samples_per_dataset
    problems = _load_math_dataset(hf_id, label, limit)

    examples: list[EvaluationExample] = []
    for idx, problem in enumerate(problems):
        question_id = (
            f"{problem.dataset}-{problem.problem_idx}"
            if problem.problem_idx is not None
            else f"{problem.dataset}-{idx}"
        )
        metadata = {
            "dataset": problem.dataset,
            "problem_idx": problem.problem_idx,
            "problem_type": problem.problem_type,
            "normalizer": "normalize_math_answer",
        }
        examples.append(
            EvaluationExample(
                dataset=problem.dataset,
                question_id=question_id,
                question_index=idx,
                question_block=problem.problem,
                correct_answer=problem.answer,
                metadata=metadata,
            )
        )
    return examples


for dataset_key in MATH_DATASETS:
    register_dataset(
        DatasetSpec(
            name=dataset_key,
            loader=lambda settings, key=dataset_key: load_math_examples(settings, key),
            aliases=(f"math_{dataset_key}",),
        )
    )


# ----------------------------------------------------------------------------
# QA datasets (text answers)
# ----------------------------------------------------------------------------

QA_DATASETS = {
    "simpleqa": ("google/simpleqa-verified", "SimpleQA"),
}


def load_qa_dataset(
    settings: SharedSettings, dataset_key: str
) -> Sequence[EvaluationExample]:
    dataset_label = QA_DATASETS[dataset_key][0]
    print(dataset_label)
    ds = load_dataset(dataset_label)
    assert isinstance(ds, DatasetDict), "Expected dataset to be a DatasetDict"
    test = ds["eval"]

    problems = cast(list[str], test["problem"])
    answers = cast(list[str], test["answer"])
    topic_list = cast(list[str], test["topic"])
    answer_type_list = cast(list[str], test["answer_type"])

    items: list[EvaluationExample] = []
    for idx, (problem, answer, topic, answer_type) in enumerate(
        zip(problems, answers, topic_list, answer_type_list)
    ):
        metadata = {
            "topic": topic,
            "answer_type": answer_type,
            "evaluation_type": "text_answer",
        }
        items.append(
            EvaluationExample(
                dataset=dataset_label,
                question_id=str(idx),
                question_index=idx,
                question_block=problem,
                correct_answer=answer,  # Store the letter (a-d) as the answer
                metadata=metadata,
            )
        )
        max_samples = settings.max_samples_per_dataset
        if len(items) >= max_samples:
            break
    return items


for dataset_key in QA_DATASETS:
    register_dataset(
        DatasetSpec(
            name=dataset_key,
            loader=lambda settings, key=dataset_key: load_qa_dataset(settings, key),
            aliases=(f"{dataset_key}",),
        )
    )
