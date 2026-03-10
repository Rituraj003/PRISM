#!/bin/bash
######################## start of slurm options #########################################
#SBATCH --job-name=gpqa          # Job name
#SBATCH --account=llms-ar                  # Account name
#SBATCH --time=36:00:00                    # Time limit (HH:MM:SS)
#SBATCH --partition=a100_normal_q          # GPU partition for vLLM
#SBATCH --output=job-outputs/%j-gpqa.out  # Standard output file (%j = job ID)
#SBATCH --error=job-outputs/%j-gpqa.err   # Standard error file
#SBATCH --mem=48G                          # Memory per node (increased for both)
#SBATCH --gres=gpu:1                      # GPU for vLLM
########################## end of slurm options #########################################

set -e  # Exit on error

# Configuration
# Use job-specific port to avoid conflicts when multiple jobs run on same node
# Base port 8300 + (last 2 digits of job ID)
if [ -n "$SLURM_JOB_ID" ]; then
    PORT_OFFSET=$((SLURM_JOB_ID % 100))
    VLLM_PORT=$((8300 + PORT_OFFSET))
else
    VLLM_PORT=8332  # Default for non-SLURM runs
fi
VLLM_HOST="localhost"
MODEL_NAME="${MODEL_NAME:-openai/gpt-oss-20b}"  # Default model, override with env var

echo "========================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Started at: $(date)"
echo "========================================"

# Load modules for vLLM
module load CUDA/12.4.0
module load Miniconda3/24.7.1-0

export CUDA_HOME=$CUDA_HOME
export HF_HOME=~/.cache/huggingface
export TRANSFORMERS_CACHE=/projects/llms-lab/transformers

# Initialize conda
eval "$(conda shell.bash hook)"
conda activate grpo >/dev/null 2>&1

echo "Starting vLLM server on port $VLLM_PORT..."
echo "Model: $MODEL_NAME"

# Start vLLM server in background
# Set VLLM_LOG_FILE env var to override log location
VLLM_LOG="${VLLM_LOG_FILE:-job-outputs/vllm_server_${VLLM_PORT}.log}"
echo "vLLM logs: $VLLM_LOG"

python -m vllm.entrypoints.openai.api_server \
    --port $VLLM_PORT \
    --model "$MODEL_NAME" \
    --trust-remote-code \
    > "$VLLM_LOG" 2>&1 &
VLLM_PID=$!

# Function to cleanup vLLM server on exit
cleanup() {
    echo "Cleaning up..."
    if kill -0 $VLLM_PID 2>/dev/null; then
        echo "Stopping vLLM server (PID: $VLLM_PID)"
        kill $VLLM_PID
        wait $VLLM_PID 2>/dev/null || true
    fi
    echo "Finished at: $(date)"
}
trap cleanup EXIT

# Wait for vLLM server to be ready
echo "Waiting for vLLM server to be ready..."
MAX_WAIT=1200  # 20 minutes max wait
WAITED=0
while ! curl -s "http://${VLLM_HOST}:${VLLM_PORT}/health" > /dev/null 2>&1; do
    if ! kill -0 $VLLM_PID 2>/dev/null; then
        echo "ERROR: vLLM server died unexpectedly"
        exit 1
    fi
    if [ $WAITED -ge $MAX_WAIT ]; then
        echo "ERROR: vLLM server failed to start within ${MAX_WAIT}s"
        exit 1
    fi
    sleep 5
    WAITED=$((WAITED + 5))
    echo "  Waiting... (${WAITED}s)"
done

echo "vLLM server is ready!"
echo "========================================"

# Set model URL for benchmark
export MODEL_URL="http://${VLLM_HOST}:${VLLM_PORT}/v1"

# Run benchmark with remaining arguments + model_url
echo "Running benchmark with args: $* --model_url $MODEL_URL"
echo "Model URL: $MODEL_URL"
echo "========================================"

cd "$SLURM_SUBMIT_DIR"
uv run "$@" --model_url "$MODEL_URL"

echo "========================================"
echo "Benchmark completed successfully!"

# ========================================
# USAGE EXAMPLES:
# ========================================
# Basic (uses default model):
#   sbatch script/run_bench.sh python src/main.py --method zero-shot --datasets gpqa
#
# With composable method:
#   sbatch script/run_bench.sh python src/main.py \
#       --cp sample_n --p2p prm_mcmc --p2a majority_vote \
#       --datasets aime --samples 10 -d 5 \
#       --model_url http://localhost:8332/v1
#
# Override model:
#   MODEL_NAME="meta-llama/Llama-3-70B" sbatch script/run_bench.sh python src/main.py ...
#
# Mute vLLM logs:
#   VLLM_LOG_FILE="/dev/null" sbatch script/run_bench.sh python src/main.py ...
#
# Custom log location:
#   VLLM_LOG_FILE="my_custom.log" sbatch script/run_bench.sh python src/main.py ...
#
# Key arguments (see src/settings.py for full list):
#   --method          : Evaluation method (zero-shot, etc.)
#   --cp              : Create population stage (sample_n, etc.)
#   --p2p             : Pop-to-pop stage (prm_mcmc, refine, etc.)
#   --p2a             : Pop-to-answer stage (majority_vote, prm_score_vote, llm_aggregate)
#   --datasets        : Datasets to evaluate (gpqa, aime, hmmt)
#   --samples         : Number of samples per question
#   -d                : Depth iterations (e.g., -d 5)
#   -w                : Width parameter (e.g., -w 10)
#   --output_csv      : Output file path (NOTE: underscore!)
#   --model_url       : Model API endpoint (NOTE: underscore!)
#   -t                : Temperature (e.g., -t 0.8)
#   --prm_mcmc_t      : PRM-MCMC temperature (default 0.2)
#   --prm_mcmc_ess    : PRM-MCMC ESS threshold (default 0.5)
#   --prm_mcmc_noise  : PRM-MCMC noise probability (default 0.1)
# ==========================================
