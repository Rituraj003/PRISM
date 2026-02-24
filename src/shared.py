from __future__ import annotations

import csv
import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Literal, Protocol, Sequence

from httpx import AsyncClient, HTTPStatusError
from pydantic import BaseModel, Field
from pydantic_ai.models import Model
from pydantic_ai.models.google import GoogleModel, GoogleModelSettings
from pydantic_ai.models.openai import OpenAIChatModel, OpenAIChatModelSettings
from pydantic_ai.providers.google import GoogleProvider
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.providers.together import TogetherProvider
from pydantic_ai.retries import AsyncTenacityTransport, RetryConfig, wait_retry_after
from pydantic_ai.settings import ModelSettings
from pydantic_ai.usage import RunUsage
from tenacity import retry_if_exception_type, stop_after_attempt, wait_exponential

from settings import SharedSettings

# Type alias for answers - can be MCQ letters (a-d) or math values (any string)
Choice = str
# Specific type for MCQ labels only
MCQLabel = Literal["a", "b", "c", "d"]


class QuestionType(str, Enum):
    MCQ = "mcq"
    MATH = "math"
    NUMBER = "number"
    DATE = "date"
    PERSON = "person"
    PLACE = "place"
    OTHER = "other"

    @classmethod
    def from_metadata(cls, value: str) -> "QuestionType":
        normalized = value.lower()
        for member in cls:
            if member.value == normalized:
                return member
        raise ValueError(f"Unknown QuestionType from metadata: {value}")

    @property
    def is_textual(self) -> bool:
        return self in {
            QuestionType.NUMBER,
            QuestionType.DATE,
            QuestionType.PERSON,
            QuestionType.PLACE,
            QuestionType.OTHER,
        }

    @property
    def prompt_family(self) -> Literal["mcq", "math", "text"]:
        if self == QuestionType.MCQ:
            return "mcq"
        if self == QuestionType.MATH:
            return "math"
        return "text"

class EvaluationExample(BaseModel):
    dataset: str
    question_id: str
    question_index: int = Field(default=0, ge=0)
    question_block: str
    correct_answer: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    # Optional: For multiple choice questions
    choices: tuple[str, ...] | None = None

    def to_prompt(self) -> str:
        if self.choices is not None:
            return build_user_prompt(self.question_block)
        return self.question_block

    @property
    def question_type(self) -> QuestionType:
        """Return the question/answer modality for downstream routing."""
        if self.choices is not None:
            return QuestionType.MCQ
        if self.metadata.get("evaluation_type") == "text_answer":
            raw_answer_type = self.metadata.get("answer_type")
            assert isinstance(raw_answer_type, str)
            return QuestionType.from_metadata(raw_answer_type)
        return QuestionType.MATH

    @property
    def label_map(self) -> dict[str, str] | None:
        """For MCQ, returns mapping of labels (a-d) to choice text."""
        if self.choices is None:
            return None
        return {label: text for label, text in zip("abcd", self.choices)}


class EvaluationResponse(BaseModel):
    method: str
    dataset: str
    question_id: str
    question_index: int
    chain_id: int | None = None
    step: int = 0
    reasoning: str | None = None
    raw_answer: str | None = None
    normalized_answer: str | None = None
    predicted_label: Choice | None = None
    is_correct: bool | None = None
    latency_seconds: float | None = None
    total_input_tokens: int
    total_output_tokens: int
    secondary_input_tokens: int = 0
    secondary_output_tokens: int = 0
    verifier_model_name: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


@dataclass
class DepthEvent:
    depth: int
    answer: str | None
    chain_id: int | None = None


@dataclass
class MethodResult:
    responses: list[EvaluationResponse]
    depth_events: list[DepthEvent] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class TokenUsage:
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    secondary_input_tokens: int = 0
    secondary_output_tokens: int = 0

    def add(self, other: "TokenUsage") -> None:
        self.total_input_tokens += other.total_input_tokens
        self.total_output_tokens += other.total_output_tokens
        self.secondary_input_tokens += other.secondary_input_tokens
        self.secondary_output_tokens += other.secondary_output_tokens

    def add_secondary(self, other: "TokenUsage") -> None:
        self.secondary_input_tokens += other.total_input_tokens
        self.secondary_output_tokens += other.total_output_tokens

    def clone(self) -> "TokenUsage":
        return TokenUsage(
            total_input_tokens=self.total_input_tokens,
            total_output_tokens=self.total_output_tokens,
            secondary_input_tokens=self.secondary_input_tokens,
            secondary_output_tokens=self.secondary_output_tokens,
        )

    def is_empty(self) -> bool:
        return (
            self.total_input_tokens == 0
            and self.total_output_tokens == 0
            and self.secondary_input_tokens == 0
            and self.secondary_output_tokens == 0
        )


def extract_usage_tokens(usage: RunUsage) -> TokenUsage:
    return TokenUsage(
        total_input_tokens=usage.input_tokens,
        total_output_tokens=usage.output_tokens,
    )


class DatasetLoader(Protocol):
    def __call__(self, settings: "SharedSettings") -> Sequence[EvaluationExample]: ...


def build_user_prompt(question_block: str) -> str:
    return f"<question>{question_block}</question>"


def build_model(
    settings: SharedSettings, verifier: bool = False
) -> tuple[Model, ModelSettings]:
    if verifier and settings.verifier_model_name:
        name = settings.verifier_model_name
        url = settings.verifier_model_url
    else:
        name = settings.model_name
        url = settings.model_url

    name_lower = name.lower()

    if "gemini" in name_lower:
        if not settings.GEMINI_API_KEY:
            raise ValueError(
                "GEMINI_API_KEY is required when using a Gemini model name."
            )

        def create_retrying_client():
            """Create a client with smart retry handling for multiple error types."""

            def should_retry_status(response):
                """Raise exceptions for retryable HTTP status codes."""
                if response.status_code in (429, 502, 503, 504):
                    response.raise_for_status()  # This will raise HTTPStatusError

            transport = AsyncTenacityTransport(
                config=RetryConfig(
                    # Retry on HTTP errors and connection issues
                    retry=retry_if_exception_type((HTTPStatusError, ConnectionError)),
                    # Smart waiting: respects Retry-After headers, falls back to exponential backoff
                    wait=wait_retry_after(
                        fallback_strategy=wait_exponential(multiplier=1, max=60),
                        max_wait=300,
                    ),
                    # Stop after 5 attempts
                    stop=stop_after_attempt(5),
                    # Re-raise the last exception if all retries fail
                    reraise=True,
                ),
                validate_response=should_retry_status,
            )
            return AsyncClient(transport=transport)

        client = create_retrying_client()
        provider = GoogleProvider(api_key=settings.GEMINI_API_KEY, http_client=client)
        model = GoogleModel(name, provider=provider)
        model_settings = GoogleModelSettings(
            google_thinking_config={"include_thoughts": True, "thinking_budget": -1},
            temperature=settings.temperature,
            top_p=settings.top_p,
        )
    else:
        if url:
            provider = OpenAIProvider(base_url=url)
        elif "o3" in name_lower and settings.OPENAI_API_KEY:
            provider = OpenAIProvider(api_key=settings.OPENAI_API_KEY)
        elif settings.TOGETHER_API_KEY:
            provider = TogetherProvider(api_key=settings.TOGETHER_API_KEY)
        else:
            raise ValueError(
                "TOGETHER_API_KEY is required when using a non-Gemini model name."
            )
        model = OpenAIChatModel(name, provider=provider)

        extra_body = None
        # Include provider-specific extra_body when using vllm/qwen/hf inference
        if url or "vllm" in name_lower or "qwen" in name_lower:
            extra_body = {"chat_template_kwargs": {"enable_thinking": True}}

        kwargs = {
            "temperature": settings.temperature,
            "top_p": settings.top_p,
            "openai_reasoning_effort": "medium",
            "timeout": 300,
        }
        if extra_body is not None:
            kwargs["extra_body"] = extra_body

        model_settings = OpenAIChatModelSettings(**kwargs)
    return model, model_settings


def sanitized_settings_dump(settings: SharedSettings) -> str:
    raw = settings.model_dump()
    secret_like = {"GEMINI_API_KEY", "TOGETHER_API_KEY", "HF_TOKEN"}

    def is_secret(key: str) -> bool:
        upper = key.upper()
        return (
            key in secret_like
            or "API_KEY" in upper
            or upper.endswith("_KEY")
            or "TOKEN" in upper
        )

    filtered: dict[str, Any] = {}
    for key, value in raw.items():
        if is_secret(key):
            continue
        if isinstance(value, Path):
            filtered[key] = str(value)
        else:
            filtered[key] = value
    return json.dumps(filtered, sort_keys=True)


class ResultLogger:
    header = [
        "timestamp",
        "run_id",
        "method",
        "dataset",
        "question_id",
        "question_index",
        "chain_id",
        "step",
        "temperature",
        "run_description",
        "question",
        "correct_answer",
        "correct_label",
        "raw_answer",
        "normalized_answer",
        "predicted_label",
        "is_correct",
        "reasoning",
        "latency_seconds",
        "total_input_tokens",
        "total_output_tokens",
        "secondary_input_tokens",
        "secondary_output_tokens",
        "verifier_model_name",
        "metadata_json",
        "settings_json",
    ]

    def __init__(self, csv_path: Path | str, settings: SharedSettings):
        self.csv_path = Path(csv_path)
        self.settings = settings
        self._settings_json = sanitized_settings_dump(settings)
        self._output_header = list(self.header)
        self._ensure_header()

    def _ensure_header(self) -> None:
        os.makedirs(self.csv_path.parent, exist_ok=True)
        if not self.csv_path.exists():
            with open(self.csv_path, "w", newline="") as handle:
                csv.writer(handle).writerow(self.header)
            self._output_header = list(self.header)
            return
        try:
            with open(self.csv_path, newline="") as handle:
                reader = csv.reader(handle)
                existing = next(reader, None)
            if existing:
                self._output_header = list(existing)
        except Exception:
            # Fall back to canonical header if file can't be read.
            self._output_header = list(self.header)

    def log_response(
        self,
        run_id: str,
        example: EvaluationExample,
        response: EvaluationResponse,
    ) -> None:
        correct_label = ""
        if example.question_type == QuestionType.MCQ:
            correct_label = example.correct_answer or ""

        metadata_combined = {
            "example": example.metadata,
            "response": response.metadata,
        }

        is_correct = ""
        if response.is_correct is not None:
            is_correct = "1" if response.is_correct else "0"

        row = [
            datetime.now(tz=timezone.utc).isoformat(),
            run_id,
            response.method,
            example.dataset,
            example.question_id,
            example.question_index,
            "" if response.chain_id is None else response.chain_id,
            response.step,
            self.settings.temperature,
            self.settings.run_desc,
            example.question_block,
            example.correct_answer or "",
            correct_label,
            response.raw_answer or "",
            response.normalized_answer or "",
            response.predicted_label or "",
            is_correct,
            response.reasoning or "",
            (
                ""
                if response.latency_seconds is None
                else f"{response.latency_seconds:.4f}"
            ),
            str(response.total_input_tokens),
            str(response.total_output_tokens),
            str(response.secondary_input_tokens),
            str(response.secondary_output_tokens),
            response.verifier_model_name or "",
            json.dumps(metadata_combined, sort_keys=True),
            self._settings_json,
        ]
        row_dict = dict(zip(self.header, row))
        out_row = [row_dict.get(col, "") for col in self._output_header]
        with open(self.csv_path, "a", newline="") as handle:
            csv.writer(handle).writerow(out_row)


class DepthLogger:
    header = [
        "timestamp",
        "run_id",
        "method",
        "dataset",
        "question_id",
        "question_index",
        "chain_id",
        "step",
        "temperature",
        "run_description",
        "question",
        "correct_answer",
        "correct_label",
        "raw_answer",
        "normalized_answer",
        "predicted_label",
        "is_correct",
        "reasoning",
        "latency_seconds",
        "total_input_tokens",
        "total_output_tokens",
        "secondary_input_tokens",
        "secondary_output_tokens",
        "verifier_model_name",
        "metadata_json",
        "settings_json",
    ]

    def __init__(self, csv_path: Path | str, settings: SharedSettings):
        self.csv_path = Path(csv_path)
        self.settings = settings
        self._settings_json = sanitized_settings_dump(settings)
        self._output_header = list(self.header)
        self._ensure_header()

    def _ensure_header(self) -> None:
        os.makedirs(self.csv_path.parent, exist_ok=True)
        if not self.csv_path.exists():
            with open(self.csv_path, "w", newline="") as handle:
                csv.writer(handle).writerow(self.header)
            self._output_header = list(self.header)
            return
        try:
            with open(self.csv_path, newline="") as handle:
                reader = csv.reader(handle)
                existing = next(reader, None)
            if existing:
                self._output_header = list(existing)
        except Exception:
            self._output_header = list(self.header)

    def log_response(
        self,
        run_id: str,
        example: EvaluationExample,
        response: EvaluationResponse,
    ) -> None:
        correct_label = ""
        if example.question_type == QuestionType.MCQ:
            correct_label = example.correct_answer or ""

        metadata_combined = {
            "example": example.metadata,
            "response": response.metadata,
        }

        is_correct = ""
        if response.is_correct is not None:
            is_correct = "1" if response.is_correct else "0"

        row = [
            datetime.now(tz=timezone.utc).isoformat(),
            run_id,
            response.method,
            example.dataset,
            example.question_id,
            example.question_index,
            "" if response.chain_id is None else response.chain_id,
            response.step,
            self.settings.temperature,
            self.settings.run_desc,
            example.question_block,
            example.correct_answer or "",
            correct_label,
            response.raw_answer or "",
            response.normalized_answer or "",
            response.predicted_label or "",
            is_correct,
            response.reasoning or "",
            (
                ""
                if response.latency_seconds is None
                else f"{response.latency_seconds:.4f}"
            ),
            str(response.total_input_tokens),
            str(response.total_output_tokens),
            str(response.secondary_input_tokens),
            str(response.secondary_output_tokens),
            response.verifier_model_name or "",
            json.dumps(metadata_combined, sort_keys=True),
            self._settings_json,
        ]
        row_dict = dict(zip(self.header, row))
        out_row = [row_dict.get(col, "") for col in self._output_header]
        with open(self.csv_path, "a", newline="") as handle:
            csv.writer(handle).writerow(out_row)

    def log(
        self,
        run_id: str,
        settings: SharedSettings,
        example: EvaluationExample,
        method: str,
        events: Sequence[DepthEvent],
    ) -> None:
        # Legacy method for backward compatibility - logs summary depth events
        if not events:
            return

        correct_answer = example.correct_answer or ""

        with open(self.csv_path, "a", newline="") as handle:
            writer = csv.writer(handle)
            for event in events:
                # Create a minimal row for legacy depth events
                row = [
                    datetime.now(tz=timezone.utc).isoformat(),
                    run_id,
                    method,
                    example.dataset,
                    example.question_id,
                    example.question_index,
                    "" if event.chain_id is None else event.chain_id,
                    event.depth,
                    settings.temperature,
                    settings.run_desc,
                    example.question_block,
                    correct_answer,
                    correct_answer if example.question_type == QuestionType.MCQ else "",
                    event.answer or "",
                    event.answer or "",
                    event.answer or "",
                    "",  # is_correct - would need to be calculated
                    "",  # reasoning
                    "",  # latency
                    "",  # total_input_tokens
                    "",  # total_output_tokens
                    "",  # secondary_input_tokens
                    "",  # secondary_output_tokens
                    "",  # verifier_model_name
                    # Include a minimal `response` object for legacy rows so
                    # downstream analysis and cache lookup can at least see
                    # seed/population metadata. We can't reconstruct the
                    # full create_stage name here, but including a
                    # seed_signature (with settings) prevents these rows
                    # from being treated as completely legacy/unknown.
                    json.dumps(
                        {
                            "legacy_depth_event": True,
                            "depth": event.depth,
                            "chain_id": event.chain_id,
                            "response": {
                                "seed_signature": {
                                    "create_stage": None,
                                    "dataset": example.dataset,
                                    "depth_iterations": 0,
                                    "pop_to_pop_stage": None,
                                    "question_id": example.question_id,
                                    "question_index": example.question_index,
                                    "settings": json.loads(self._settings_json),
                                }
                            },
                        }
                    ),
                    self._settings_json,
                ]
                writer.writerow(row)
