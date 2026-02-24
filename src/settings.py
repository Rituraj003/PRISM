from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class SharedSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        cli_parse_args=True,
        extra="ignore",
    )

    # Run configuration
    method: str = Field(default="zero-shot", description="Evaluation method to run")
    run_desc: str = Field(
        default="",
        description="Optional free-form description of this run",
        alias="desc",
    )
    output_csv: Path = Field(
        default=Path("data/outputs/shared-results.csv"),
        description="Path of the shared CSV to append results to",
    )
    depth_metrics_csv: Path = Field(
        default=Path("data/outputs/depth_accuracy.csv"),
        description="Path for logging per-depth accuracy rows",
    )

    # Composable method stage selection
    create_population: str | None = Field(
        default=None,
        description="Stage 1: How to create initial population (e.g., 'sample_n')",
        alias="cp",
    )
    pop_to_pop: str | None = Field(
        default=None,
        description="Stage 2: How to transform population (e.g., 'refine', 'agentic_debate', 'recursive_aggregate', 'mad_conformist', 'mad_follower','none')",
        alias="p2p",
    )
    pop_to_answer: str | None = Field(
        default=None,
        description="Stage 3: How to reduce population to answer (e.g., 'majority_vote', 'llm_aggregate')",
        alias="p2a",
    )

    # Dataset controls
    datasets: list[str] = Field(
        default_factory=lambda: ["gpqa"],
        description="Datasets to evaluate (e.g. gpqa, hmmt, aime)",
    )
    max_samples_per_dataset: int = Field(
        default=120, description="Max problems to load per dataset", alias="n"
    )
    start_index: int = Field(
        default=0,
        description="Start index for resuming runs (skip first N examples)",
        alias="start",
        ge=0,
    )
    question_ids: list[str] = Field(
        default_factory=list,
        description="Optional list of question_ids to evaluate (exact match)",
    )
    seed: int = Field(default=69)

    # Sampling / orchestrator controls
    samples: int = Field(
        default=10,
        description="Number of zero-shot samples per question",
        ge=1,
    )
    width: int = Field(default=5, ge=1, alias="w")
    depth: int = Field(default=10, ge=0, alias="d")
    agg_pool_size: int = Field(
        default=4,
        ge=1,
        alias="agg",
        description="Aggregation pool size for methods that use it",
    )

    # API and auth
    GEMINI_API_KEY: str | None = Field(default=None, description="API key for Gemini")
    OPENAI_API_KEY: str | None = Field(
        default=None, description="API key for OpenAI Reasoning Models"
    )
    HF_TOKEN: str | None = Field(
        default=None, description="HF auth token (optional but recommended)"
    )
    TOGETHER_API_KEY: str | None = Field(
        default=None, description="API key for Together"
    )

    # Model selection
    model_name: str = Field(default="gemini-2.5-flash", alias="m")
    model_url: str | None = Field(
        default=None,
        description="Optional custom model URL if using local model server",
    )
    verifier_model_name: str | None = Field(
        default=None,
        description="Optional secondary model for verification tasks",
        alias="vm",
    )
    verifier_model_url: str | None = Field(
        default=None,
        description="Optional custom URL for the verifier model",
        alias="vm_url",
    )
    temperature: float = Field(
        default=0.5, description="Sampling temperature", alias="t"
    )
    prm_temperature: float = Field(
        default=0.0,
        description="PRM verification temperature",
        alias="prm_t",
        ge=0.0,
    )
    top_p: float = Field(default=0.9)

    # PRISM controls
    prism_temperature: float = Field(
        default=0.8,
        description="PRISM temperature for Boltzmann weights",
        alias="prism_t",
        ge=0.0,
    )
    prism_ess_threshold: float = Field(
        default=0.5,
        description="PRISM resampling threshold (ESS < threshold * N)",
        alias="prism_ess",
        ge=0.0,
        le=1.0,
    )
    prism_acceptance_noise: float = Field(
        default=0.1,
        description="PRISM proposal noise probability",
        alias="prism_noise",
        ge=0.0,
        le=1.0,
    )

    # MadFollower / PRISM shared settings
    follower_ratio: float = Field(
        default=0.3,
        description="Follower ratio for MadFollower; also used by prism for replication share",
        alias="follower_ratio",
    )
