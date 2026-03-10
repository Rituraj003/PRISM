# PRISM

PRISM is a research codebase for evaluating LLM reasoning methods across multiple benchmark types:

- multiple-choice QA (`gpqa`)
- math competition problems (`hmmt`, `aime`)

The repo supports both single-shot baselines and composable multi-stage methods such as recursive aggregation, debate-style refinement, and PRISM.

## Paper

This repository corresponds to the paper:

- [PRISM: Pushing the Frontier of Deep Think via Process Reward Model-Guided Inference](https://arxiv.org/abs/2603.02479)

## What This Repo Does

At a high level, the benchmark runner:

1. loads examples from one or more datasets
2. runs a method on each example
3. checks correctness when possible
4. appends results to CSV files in `data/outputs/`

There are two main output files:

- `data/outputs/shared-results.csv`: final answers only
- `data/outputs/depth_accuracy.csv`: per-step / per-chain depth traces plus final rows

## Requirements

- Python `>=3.12` in `pyproject.toml`
- `uv` recommended for dependency management
- access to at least one model backend:
  - Gemini via `GEMINI_API_KEY`
  - Together via `TOGETHER_API_KEY`
  - OpenAI-compatible endpoint via `--model_url`
- `HF_TOKEN` recommended for dataset access, especially `livecodebench_v6`

## Setup

### 1. Create the environment

```bash
uv sync
```

If you want dev tools too:

```bash
uv sync --group dev
```

### 2. Configure environment variables

Create a `.env` file in the repo root if you do not want to pass secrets through the shell:

```env
GEMINI_API_KEY =...   (if using Gemini)
OPENAI_API_KEY=...    (if using OpenAI)
TOGETHER_API_KEY=...  (if using Together)
HF_TOKEN=...     (for Hugging Face datasets)
```

Only set the variables you actually need.

## Quick Start

Show all CLI options:

```bash
uv run python src/main.py --help
```

Run a small zero-shot evaluation:

```bash
uv run python src/main.py \
  --datasets gpqa \
  --method zero-shot \
  -n 10 \
  --samples 1 \
  -m openai/gpt-oss-20b \
  --model_url http://localhost:8089/v1
```

Run a composable method:

```bash
uv run python src/main.py \
  --datasets gpqa \
  --cp sample_n \
  --p2p recursive_aggregate \
  --p2a majority_vote \
  --samples 10 \
  -w 10 \
  -d 5 \
  -t 0.8 \
  -m openai/gpt-oss-20b \
  --model_url http://localhost:8089/v1
```

Run PRISM:

```bash
uv run python src/main.py \
  --datasets gpqa \
  --cp sample_n \
  --p2p prism \
  --p2a prm_score_vote \
  --samples 10 \
  -w 10 \
  -d 5 \
  -t 0.8 \
  --prism_t 0.8 \
  --prism_ess 0.5 \
  --prism_noise 0.1 \
  -m openai/gpt-oss-20b \
  --model_url http://localhost:8089/v1
```

## Core CLI Arguments

Common arguments:

- `--datasets`: dataset names to run
- `-n`, `--max_samples_per_dataset`: cap examples per dataset
- `--start`: skip the first `N` examples in each selected dataset
- `--question_ids`: run only specific question IDs
- `-m`: model name
- `--model_url`: OpenAI-compatible endpoint for local or remote servers
- `-t`: sampling temperature
- `--output_csv`: final-result CSV path
- `--depth_metrics_csv`: depth-trace CSV path

Composable method arguments:

- `--cp`: create-population stage
- `--p2p`: population-to-population stage
- `--p2a`: population-to-answer stage
- `--samples`: number of seed samples
- `-w`: width
- `-d`: depth
- `--agg`: aggregation pool size

PRISM-specific arguments:

- `--prism_t`
- `--prism_ess`
- `--prism_noise`
- `--follower_ratio`

## Available Datasets

Registered datasets in the current code:

- `gpqa`
- `hmmt`
- `aime`


## Available Methods

Built-in full methods:

- `zero-shot`

Composable stages:

- create-population:
  - `sample_n`
- population-to-population:
  - `refine`
  - `agentic_debate`
  - `recursive_aggregate`
  - `mad_conformist`
  - `mad_follower`
  - `prism`
- population-to-answer:
  - `majority_vote`
  - `prm_score_vote`
  - `llm_aggregate`

When you use `--cp/--p2p/--p2a`, the runner builds a dynamic method name such as:

```text
sample_n_recursive_aggregate_majority_vote
```

## Running on Slurm

The repo includes helper scripts in `script/`:

- `script/bench.sh`: run the benchmark on a CPU partition against an existing model endpoint
- `script/run_bench.sh`: start a vLLM server on a GPU node, then run the benchmark against it
- `script/serve_vllm.sh`: standalone vLLM server job
- `script/sweep_methods_parallel.sh`: submit one Slurm job per method combination

### Submit a benchmark job against an existing endpoint

```bash
sbatch script/bench.sh src/main.py \
  -m openai/gpt-oss-20b \
  --model_url http://tc-gpu001:8089/v1 \
  --datasets gpqa \
  --cp sample_n \
  --p2p recursive_aggregate \
  --p2a majority_vote \
  --samples 10 \
  -w 10 \
  -d 5
```

### Submit a job that launches vLLM first

```bash
sbatch script/run_bench.sh python src/main.py \
  --datasets gpqa \
  --method zero-shot \
  -n 10 \
  --samples 1 \
  -m openai/gpt-oss-20b
```

Cluster logs are written to `job-outputs/`.

## Outputs and Logging

`shared-results.csv` includes one final row per example. Typical columns:

- `run_id`
- `method`
- `dataset`
- `question_id`
- `question_index`
- `step`
- `raw_answer`
- `normalized_answer`
- `predicted_label`
- `is_correct`
- token usage fields

`depth_accuracy.csv` includes intermediate responses too:

- per-chain seed rows
- per-depth transformed populations
- final answer rows

This is the file to use for depth curves and chain-level analysis.

## Development

Run tests:

```bash
uv run pytest
```

Run linting:

```bash
uv run ruff check .
```

Run type checking:

```bash
uv run pyright
```

Current tests cover:

- answer normalization / verification
- composable cache behavior

## Project Layout

```text
src/main.py                    CLI entrypoint
src/settings.py                CLI / env configuration
src/data_sources.py            dataset registry and loaders
src/shared.py                  model setup, schemas, CSV logging
src/methods/                   methods and composable stages
script/bench.sh                Slurm benchmark wrapper
script/run_bench.sh            Slurm vLLM + benchmark wrapper
script/serve_vllm.sh           Slurm vLLM server launcher
script/sweep_methods_parallel.sh Slurm sweep helper
data/outputs/                  result CSVs
job-outputs/                   Slurm stdout / stderr logs
tests/                         test suite
```

## Notes and Gotchas

- Non-Gemini models require either `--model_url`, `TOGETHER_API_KEY`, or a supported OpenAI setup.
- For OpenAI-compatible local servers, `--model_url` is usually the simplest path.
- Output CSVs are append-only by default.
- Large runs can produce very large CSV files because code prompts, traces, and reasoning are stored inline.

## License

This project is licensed under the MIT License. See [LICENSE](/home/rituraj/PRISM/LICENSE).
