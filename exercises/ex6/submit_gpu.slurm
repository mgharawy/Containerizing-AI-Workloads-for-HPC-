#!/bin/bash
#SBATCH --job-name=singularity_single_gpu
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --mem=16G
#SBATCH --time=00:15:00

# Exit script immediately on non-zero command failure
set -eo pipefail

echo "=========================================================="
echo "Job Execution Started: $(date)"
echo "Node Allocated: $(hostname)"
echo "Assigned GPU Index: ${CUDA_VISIBLE_DEVICES}"
echo "=========================================================="

# 1. Load Singularity / Apptainer module (site-specific; adjust if needed)
module load singularity 2>/dev/null || module load apptainer 2>/dev/null || true

# 2. Define Image and Data Paths
SIF_IMAGE="/ibex/user/$USER/slurm_gpu_lab/pytorch_24.08.sif"
WORK_DIR="/ibex/user/$USER/slurm_gpu_lab"

# Ensure image exists before execution
if [ ! -f "${SIF_IMAGE}" ]; then
    echo "[ERROR] SIF container image not found at ${SIF_IMAGE}"
    exit 1
fi

# 3. Configure temporary cache locations
export SINGULARITY_CACHEDIR="/ibex/user/$USER/.singularity/cache"
export SINGULARITY_TMPDIR="/ibex/user/$USER/.singularity/tmp"

# 4. Launch Containerized Workload
# Flags explained:
#   --nv            : Injects host NVIDIA driver libraries and CUDA devices
#   --cleanenv      : Prevents host user environment contamination inside the container
#   --bind          : Mounts host directories into container filesystem
#   --pwd           : Sets the starting directory inside the container
srun singularity exec \
    --nv \
    --cleanenv \
    --bind /dev/shm:/dev/shm \
    --bind ${WORK_DIR}:${WORK_DIR} \
    --pwd ${WORK_DIR} \
    ${SIF_IMAGE} \
    python3 train_benchmark.py

echo "=========================================================="
echo "Job Execution Finished: $(date)"
echo "=========================================================="
