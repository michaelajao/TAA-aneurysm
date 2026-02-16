#!/bin/bash
# TAA-PINN Training Launch Script
#
# Usage:
#   bash scripts/run_training.sh                         # launch all 6 on GPU 1
#   GPU=0 bash scripts/run_training.sh                   # override GPU
#   GEOMS="AS5 AD5" bash scripts/run_training.sh         # subset of geometries

set -e
cd "$(dirname "$0")/.."
PROJECT_DIR=$(pwd)

GPU="${GPU:-1}"
GEOMS="${GEOMS:-AS5 AD5 PD5 AS6 AD6 PD6}"
CONDA_ENV="${CONDA_ENV:-deep_tf}"

echo "========================================================"
echo "  TAA-PINN Adaptive Physics Training (GPU $GPU)"
echo "  Project: $PROJECT_DIR"
echo "========================================================"
echo ""
echo "Geometries: $GEOMS"
echo ""

CONDA_BASE=$(conda info --base)
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

mkdir -p logs

RUN_TAG="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="logs/gpu${GPU}_${RUN_TAG}.log"

# Build the training commands
CMD=""
for geom in $GEOMS; do
    CMD+="echo '=== ${geom} starting ===' && "
    CMD+="python -u -m src.training.trainer --config configs/${geom}_config.yaml 2>&1 && "
    CMD+="echo '=== ${geom} done ===' && "
done
# Remove trailing ' && '
CMD="${CMD% && }"

echo "Launching on GPU $GPU ..."
nohup bash -c "
    source \"$CONDA_BASE/etc/profile.d/conda.sh\"
    conda activate $CONDA_ENV
    cd \"$PROJECT_DIR\"
    export CUDA_VISIBLE_DEVICES=$GPU
    $CMD
" > "$LOG_FILE" 2>&1 &
TRAIN_PID=$!
echo "  PID: $TRAIN_PID"

echo ""
echo "GPU $GPU launched! Order: $GEOMS"
echo "  Monitor: tail -f $LOG_FILE"
echo "  Check:   nvidia-smi"
echo ""
echo "PID: $TRAIN_PID"
