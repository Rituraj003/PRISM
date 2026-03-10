#!/bin/bash
######################## start of slurm options #########################################
#SBATCH --job-name=vllm       # Job name
#SBATCH --account=llms-ar                  # Account name
#SBATCH --time=30:00:00                     # Time limit (HH:MM:SS)
#SBATCH --partition=a100_normal_q                # Partition name
#SBATCH --output=job-outputs/%j-vllm.out          # Standard output file (%j = job ID)
#SBATCH --error=job-outputs/%j-vllm.err            # Standard error file
#SBATCH --mem=48G                           # Memory per node
#SBATCH --gres=gpu:1
########################## end of slurm options #########################################

set -e

VLLM_PORT="${VLLM_PORT:-8089}"
MODEL_NAME="${MODEL_NAME:-openai/gpt-oss-20b}"

echo "========================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Started at: $(date)"
echo "Port: $VLLM_PORT"
echo "Model: $MODEL_NAME"
echo "========================================"

module reset
module load CUDA/12.4.0
module load Miniconda3/24.7.1-0

export CUDA_HOME=$CUDA_HOME
export HF_HOME=/projects/llms-lab/huggingface
export TRANSFORMERS_CACHE=/projects/llms-lab/transformers

eval "$(conda shell.bash hook)"
conda activate grpo >/dev/null 2>&1

cd "$SLURM_SUBMIT_DIR"
python -m vllm.entrypoints.openai.api_server \
    --port "$VLLM_PORT" \
    --model "$MODEL_NAME" \
    --trust-remote-code \
    "$@"
