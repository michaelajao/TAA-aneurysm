#!/bin/bash
# Adaptive Physics Training Launch Script
# Improvements: physics annealing, periodic loss renorm, collocation resampling,
#               validation split, restored lambda_physics=1.0, consistent coord centering
#
# Usage:
#   bash hpc/run_adaptive_physics_training.sh              # launch all 6 on GPU 1
#   GPU=0 bash hpc/run_adaptive_physics_training.sh        # override GPU
#   GEOMS="AS5 AD5" bash hpc/run_adaptive_physics_training.sh  # subset of geometries

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
echo "Key improvements from previous versions:"
echo "  - lambda_physics restored to 1.0 (prevents divergence)"
echo "  - Physics loss annealing: smooth ramp over 2000 epochs"
echo "  - Periodic loss renormalization every 500 epochs"
echo "  - Dynamic collocation resampling every 1000 epochs"
echo "  - 20% validation split for generalization monitoring"
echo "  - Consistent coordinate centering (full-data mean)"
echo "  - best_model saved immediately on improvement"
echo "  - 15K epochs (optimized from previous 10K/20K trials)"
echo ""
echo "Geometries: $GEOMS"
echo ""

CONDA_BASE=$(conda info --base)
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

mkdir -p hpc/logs

RUN_TAG="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="hpc/logs/gpu${GPU}_adaptive_physics_${RUN_TAG}.log"

# Build the training commands
CMD=""
for geom in $GEOMS; do
    CMD+="echo '=== ${geom} adaptive_physics starting ===' && "
    CMD+="python -u training/train_single_geometry.py --config configs/${geom}_config.yaml 2>&1 && "
    CMD+="echo '=== ${geom} adaptive_physics done ===' && "
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
V3_PID=$!
echo "  PID: $V3_PID"

echo ""
echo "GPU $GPU launched! Order: $GEOMS"
echo "  Monitor: tail -f $LOG_FILE"
echo "  Check:   nvidia-smi"
echo ""
echo "PID: $V3_PID"
