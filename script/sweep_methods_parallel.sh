#!/bin/bash
# Parallel grid search for p2p and p2a method combinations
# Submits one SLURM job per configuration

set -e

# Fixed configuration
DATASET="hmmt"
SAMPLES=10
WIDTH=10
DEPTH=5
BASE_DESC="hmmt"
MODEL_NAME="openai/gpt-oss-20b"  # Model served by vLLM
TEMP=0.8

# Output CSVs
SHARED_RESULTS_CSV="data/outputs/shared-results_v2.csv"
SHARED_DEPTH_CSV="data/outputs/depth_accuracy_v2.csv"

# Grid: p2p:p2a (one per line for readability)
# Valid p2p: refine, prm_refine, agentic_debate, recursive_aggregate, mad_conformist, mad_follower, prm_mcmc
# Valid p2a: majority_vote, prm_score_vote, llm_aggregate
GRID=(
    # "refine:majority_vote"
    # "prm_mcmc:majority_vote"
    # "prm_refine:majority_vote"
    # "agentic_debate:majority_vote"
    "recursive_aggregate:majority_vote"
    # "mad_conformist:majority_vote"
    # "mad_follower:majority_vote"
    # "refine:llm_aggregate"
    # "agentic_debate:llm_aggregate"
    # "recursive_aggregate:llm_aggregate"
    # "mad_conformist:llm_aggregate"
    # "mad_follower:llm_aggregate"
    # "prm_refine:llm_aggregate"
    # "prm_mcmc:llm_aggregate"
)



echo "========================================"
echo "Parallel Methods Grid Search (p2p:p2a)"
echo "========================================"
echo "Output directory: $OUT_DIR"
echo "Grid size: ${#GRID[@]} configurations"
echo "Dataset: $DATASET"
echo "Samples: $SAMPLES, Width: $WIDTH, Depth: $DEPTH"
echo "Temperature: $TEMP"
echo "Model: $MODEL_NAME"
echo "Results CSV: $SHARED_RESULTS_CSV"
echo "Depth CSV: $SHARED_DEPTH_CSV"
echo "========================================"
echo

# Submit jobs
JOB_IDS=()
for config in "${GRID[@]}"; do
    # Parse config
    IFS=':' read -r P2P P2A <<< "$config"
    
    # Format tag
    TAG="${P2P}_${P2A}"
    DESC="${BASE_DESC}-${TAG}"
    
    echo "Submitting: $TAG (p2p=$P2P, p2a=$P2A)"
    
    # All jobs write to shared files
    JOB_ID=$(sbatch --parsable script/run_bench.sh python src/main.py \
        --cp sample_n \
        --p2p "$P2P" \
        --p2a "$P2A" \
        --datasets "$DATASET" \
        --samples "$SAMPLES" \
        -w "$WIDTH" \
        -d "$DEPTH" \
        -m "$MODEL_NAME" \
        -t "$TEMP" \
        --output_csv "$SHARED_RESULTS_CSV" \
        --depth_metrics_csv "$SHARED_DEPTH_CSV" \
        --desc "$DESC")
    
    JOB_IDS+=("$JOB_ID")
    echo "  → Job ID: $JOB_ID"
    echo
done

echo "========================================"
echo "Submitted ${#JOB_IDS[@]} jobs:"
echo "${JOB_IDS[@]}"
echo "========================================"
echo
echo "Monitor with: squeue -u $USER"
echo "Cancel all with: scancel ${JOB_IDS[@]}"
echo
echo "========================================"

# Save job info
INFO_FILE="./sweep_info.txt"
{
    echo "Sweep started: $(date)"
    echo "User: $USER"
    echo "Job IDs: ${JOB_IDS[@]}"
    echo "Grid size: ${#GRID[@]}"
    echo "Dataset: $DATASET"
    echo "Samples: $SAMPLES, Width: $WIDTH, Depth: $DEPTH"
    echo "Temperature: $TEMP"
    echo "Results CSV: $SHARED_RESULTS_CSV"
    echo "Depth CSV: $SHARED_DEPTH_CSV"
    echo
    echo "Grid configurations:"
    for i in "${!GRID[@]}"; do
        echo "  ${GRID[$i]} → Job ${JOB_IDS[$i]}"
    done
} > "$INFO_FILE"

echo "Sweep info saved to: $INFO_FILE"
