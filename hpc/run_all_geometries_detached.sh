#!/bin/bash
set -euo pipefail

# Sequential non-Slurm launcher for all geometries.
# Run from the project root (e.g. ~/Project/TAA-aneurysm).
#
# Typical detached usage:
#   nohup bash hpc/run_all_geometries_detached.sh > hpc/logs/nohup_all.log 2>&1 &
#
# Resume one interrupted geometry while running all:
#   RESUME_GEOM=AS5 RESUME_CKPT=experiments/AS5_baseline/checkpoint_epoch_1000.pt \
#   nohup bash hpc/run_all_geometries_detached.sh > hpc/logs/nohup_all.log 2>&1 &

CONDA_ENV="${CONDA_ENV:-deep_tf}"
RESUME_GEOM="${RESUME_GEOM:-}"
RESUME_CKPT="${RESUME_CKPT:-}"
GEOMS_LIST="${GEOMS_LIST:-AS5 AD5 PD5 AS6 AD6 PD6}"

read -r -a GEOMS <<< "${GEOMS_LIST}"

mkdir -p hpc/logs

if [[ -f "${HOME}/miniconda3/etc/profile.d/conda.sh" ]]; then
  source "${HOME}/miniconda3/etc/profile.d/conda.sh"
elif [[ -f "${HOME}/anaconda3/etc/profile.d/conda.sh" ]]; then
  source "${HOME}/anaconda3/etc/profile.d/conda.sh"
else
  echo "Could not find conda.sh. Set up conda init on HPC first."
  exit 1
fi

conda activate "${CONDA_ENV}"

RUN_TAG="$(date +%Y%m%d_%H%M%S)"
MASTER_LOG="hpc/logs/run_all_${RUN_TAG}.log"

echo "============================================================" | tee -a "${MASTER_LOG}"
echo "TAA non-Slurm sequential run started: $(date)" | tee -a "${MASTER_LOG}"
echo "Conda env: ${CONDA_ENV}" | tee -a "${MASTER_LOG}"
echo "Geometries: ${GEOMS_LIST}" | tee -a "${MASTER_LOG}"
echo "Host: $(hostname)" | tee -a "${MASTER_LOG}"
echo "============================================================" | tee -a "${MASTER_LOG}"

for geom in "${GEOMS[@]}"; do
  GEOM_LOG="hpc/logs/${geom}_${RUN_TAG}.log"
  CMD=(python training/train_single_geometry.py --config "configs/${geom}_config.yaml")

  if [[ -n "${RESUME_GEOM}" && -n "${RESUME_CKPT}" && "${geom}" == "${RESUME_GEOM}" ]]; then
    CMD+=(--resume "${RESUME_CKPT}")
  fi

  echo "[$(date)] Starting ${geom}" | tee -a "${MASTER_LOG}"
  echo "[$(date)] Command: ${CMD[*]}" | tee -a "${MASTER_LOG}"

  if "${CMD[@]}" 2>&1 | tee -a "${GEOM_LOG}" >> "${MASTER_LOG}"; then
    echo "[$(date)] Completed ${geom}" | tee -a "${MASTER_LOG}"
  else
    echo "[$(date)] FAILED ${geom}. Check ${GEOM_LOG}" | tee -a "${MASTER_LOG}"
    exit 1
  fi

  echo "------------------------------------------------------------" | tee -a "${MASTER_LOG}"
done

echo "All geometries finished successfully: $(date)" | tee -a "${MASTER_LOG}"
