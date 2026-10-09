#!/bin/bash -l

#SBATCH -J mar_inference
#SBATCH -N 1
#SBATCH -p plgrid-gpu-gh200
#SBATCH -A plgsmartudl-gpu-gh200
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:1
#SBATCH --gpu-bind=closest
#SBATCH --output="/net/storage/pr3/plgrid/plggsmartudlteam/CircNet/slurm_logs/inference_%j.out"
#SBATCH --error="/net/storage/pr3/plgrid/plggsmartudlteam/CircNet/slurm_logs/inference_%j.err"

export CUDA_LAUNCH_BLOCKING=1
export APPTAINER_CACHEDIR=/net/storage/pr3/plgrid/plggsmartudlteam/apptainer_cache
export PYTHONPATH="$PYTHONPATH:/"

TEAM_DIR=/net/storage/pr3/plgrid/plggsmartudlteam

# Nazwa run-name z treningu — wskazuje na katalog z best_model.pth
RUN_NAME="mar_run_16838558"

mkdir -p ${TEAM_DIR}/CircNet/slurm_logs

echo "[INFO] Job started: $(date)"
echo "[INFO] Running on node: $(hostname)"
echo "[INFO] SLURM Job ID: $SLURM_JOB_ID"

echo "[INFO] Starting inference"
apptainer exec --nv \
  --bind ${TEAM_DIR}/data:/app/data,\
${TEAM_DIR}/CircNet:/app/CircNet \
  ${TEAM_DIR}/sif_images/sif_image.sif \
  python3 /app/CircNet/inference.py \
    --model-path      /app/CircNet/${RUN_NAME}/best_model.pth \
    --data-root       /app/data/RPI/body \
    --test-folders    body4 \
    --output-dir      /app/CircNet/${RUN_NAME}/corrected \
    --polar-size      512x1024 \
    --radius-features 64 \
    --device          cuda \
  2>&1 | tee ${TEAM_DIR}/CircNet/slurm_logs/inference_${SLURM_JOB_ID}.log
echo "[INFO] Finished inference"

echo "[INFO] Job finished: $(date)"
