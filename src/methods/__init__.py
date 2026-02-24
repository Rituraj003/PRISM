from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from typing import Callable, Dict, Protocol

from pydantic_ai.models import Model
from pydantic_ai.settings import ModelSettings

from methods.composable.create_population import SampleN
from methods.composable.pop_to_answer import (
    LLMAggregate,
    MajorityVote,
    PrmScoreVote,
)
from methods.composable.pop_to_pop import (
    AgenticDebate,
    MadConformist,
    MadFollower,
    Prism,
    RecursiveAggregate,
    Refine,
)
from methods.composable.stages import (
    CreatePopulation,
    PopulationToAnswer,
    PopulationToPopulation,
)
from shared import EvaluationExample, MethodResult, SharedSettings


class MethodRunner(Protocol):
    async def __call__(
        self, example: EvaluationExample, question_index: int
    ) -> MethodResult: ...


@dataclass
class MethodContext:
    settings: SharedSettings
    model: Model
    model_settings: ModelSettings
    method_name: str


@dataclass
class MethodSpec:
    name: str
    create_runner: Callable[[MethodContext], MethodRunner]
    aliases: tuple[str, ...] = ()


# Registry for full methods
_METHODS: Dict[str, MethodSpec] = {}

# Registries for composable stages
_CREATE_POPULATION_STAGES: Dict[str, type[CreatePopulation]] = {}
_POP_TO_POP_STAGES: Dict[str, type[PopulationToPopulation]] = {}
_POP_TO_ANSWER_STAGES: Dict[str, type[PopulationToAnswer]] = {}


def register_method(spec: MethodSpec) -> MethodSpec:
    """Register a complete method (including outlier methods)."""
    keys = {spec.name.lower(), *(alias.lower() for alias in spec.aliases)}
    for key in keys:
        _METHODS[key] = spec
    return spec


def register_create_population(name: str, stage_class: type[CreatePopulation]) -> None:
    """Register a CreatePopulation stage implementation."""
    _CREATE_POPULATION_STAGES[name.lower()] = stage_class


def register_pop_to_pop(name: str, stage_class: type[PopulationToPopulation]) -> None:
    """Register a PopulationToPopulation stage implementation."""
    _POP_TO_POP_STAGES[name.lower()] = stage_class


def register_pop_to_answer(name: str, stage_class: type[PopulationToAnswer]) -> None:
    """Register a PopulationToAnswer stage implementation."""
    _POP_TO_ANSWER_STAGES[name.lower()] = stage_class


def get_method(name: str) -> MethodSpec | None:
    """Get a registered method by name."""
    return _METHODS.get(name.lower())


def get_create_population_stage(name: str) -> type[CreatePopulation] | None:
    """Get a CreatePopulation stage by name."""
    return _CREATE_POPULATION_STAGES.get(name.lower())


def get_pop_to_pop_stage(name: str) -> type[PopulationToPopulation] | None:
    """Get a PopulationToPopulation stage by name."""
    return _POP_TO_POP_STAGES.get(name.lower())


def get_pop_to_answer_stage(name: str) -> type[PopulationToAnswer] | None:
    """Get a PopulationToAnswer stage by name."""
    return _POP_TO_ANSWER_STAGES.get(name.lower())


def list_methods() -> list[str]:
    """List all registered method names."""
    return sorted({spec.name for spec in _METHODS.values()})


def list_create_population_stages() -> list[str]:
    """List all registered CreatePopulation stage names."""
    return sorted(_CREATE_POPULATION_STAGES.keys())


def list_pop_to_pop_stages() -> list[str]:
    """List all registered PopulationToPopulation stage names."""
    return sorted(_POP_TO_POP_STAGES.keys())


def list_pop_to_answer_stages() -> list[str]:
    """List all registered PopulationToAnswer stage names."""
    return sorted(_POP_TO_ANSWER_STAGES.keys())


# Register stage implementations
register_create_population("sample_n", SampleN)
register_pop_to_pop("refine", Refine)
register_pop_to_pop("agentic_debate", AgenticDebate)
register_pop_to_pop("recursive_aggregate", RecursiveAggregate)
register_pop_to_pop("mad_conformist", MadConformist)
register_pop_to_pop("mad_follower", MadFollower)
register_pop_to_pop("prism", Prism)
register_pop_to_answer("majority_vote", MajorityVote)
register_pop_to_answer("prm_score_vote", PrmScoreVote)
register_pop_to_answer("llm_aggregate", LLMAggregate)

# Eager import of built-in methods to populate registry
# Note: zero_shot is an "outlier" method that doesn't use composable stages
for module_name in (
    "methods.zero_shot",
):
    import_module(module_name)


__all__ = [
    "MethodContext",
    "MethodRunner",
    "MethodSpec",
    "register_method",
    "register_create_population",
    "register_pop_to_pop",
    "register_pop_to_answer",
    "get_method",
    "get_create_population_stage",
    "get_pop_to_pop_stage",
    "get_pop_to_answer_stage",
    "list_methods",
    "list_create_population_stages",
    "list_pop_to_pop_stages",
    "list_pop_to_answer_stages",
]
