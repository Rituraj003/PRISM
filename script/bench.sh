#!/bin/bash
######################## start of slurm options #########################################
#SBATCH --job-name=parallel       # Job name
#SBATCH --account=llms-ar                  # Account name
#SBATCH --time=36:00:00                     # Time limit (HH:MM:SS)
#SBATCH --partition=normal_q                # Partition name
#SBATCH --output=job-outputs/%j-parallel.out          # Standard output file (%j = job ID)
#SBATCH --error=job-outputs/%j-parallel.err            # Standard error file
#SBATCH --mem=16G                           # Memory per node
########################## end of slurm options #########################################

module reset # Resets module system (recommended)
module load Python/3.11.3-GCCcore-12.3.0
source "$SLURM_SUBMIT_DIR"/.venv/bin/activate
which python

cd "$SLURM_SUBMIT_DIR"

# Use uv if pip is missing from the venv
if ! python -m pip --version > /dev/null 2>&1; then
    echo "pip not found in venv, using uv for installation..."
    uv pip install -e .
else
    python -m pip install --upgrade pip
    python -m pip install -e .
fi

export TRANSFORMERS_CACHE=/projects/llms-lab/transformers
export HF_HOME=/projects/llms-lab/huggingface
echo "Running with args: $*"
# run whatever user passes into cli
python "$@"
