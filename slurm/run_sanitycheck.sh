#!/bin/bash -l

#SBATCH -J mar_sanitycheck
#SBATCH -N 1
#SBATCH -p plgrid-gpu-gh200
#SBATCH -A plgsmartudl-gpu-gh200
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16GB
#SBATCH --time=00:15:00
#SBATCH --gres=gpu:1
#SBATCH --output="/net/storage/pr3/plgrid/plggsmartudlteam/CircNet/slurm_logs/sanitycheck_%j.out"
#SBATCH --error="/net/storage/pr3/plgrid/plggsmartudlteam/CircNet/slurm_logs/sanitycheck_%j.err"

export APPTAINER_CACHEDIR=/net/storage/pr3/plgrid/plggsmartudlteam/apptainer_cache

TEAM_DIR=/net/storage/pr3/plgrid/plggsmartudlteam
RUN_NAME="mar_run_16838558"

mkdir -p ${TEAM_DIR}/CircNet/slurm_logs

echo "[INFO] Job started: $(date)"
echo "[INFO] Running on node: $(hostname)"
echo "[INFO] SLURM Job ID: $SLURM_JOB_ID"

apptainer exec \
  --bind ${TEAM_DIR}:/mnt \
  ${TEAM_DIR}/sif_images/sif_image.sif \
  python3 /mnt/CircNet/sanity_check.py \
    --corrected-dir /mnt/CircNet/${RUN_NAME}/corrected \
    --baseline-dir  /mnt/data/RPI/body/body4/Baseline \
    --output-dir    /mnt/CircNet/${RUN_NAME}/sanitycheck \
    --n-samples     16 \
    --window        soft_tissue

echo "[INFO] Job finished: $(date)"
