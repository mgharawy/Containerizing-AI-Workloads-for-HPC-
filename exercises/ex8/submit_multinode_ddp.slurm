#!/bin/bash
#SBATCH --job-name=singularity-ddp
#SBATCH --nodes=2                   # Number of physical nodes
#SBATCH --ntasks-per-node=2         # 1 Slurm task per GPU
#SBATCH --gpus-per-node=2           # Number of GPUs requested per node
#SBATCH --cpus-per-task=4           # CPU cores assigned per GPU task
#SBATCH --mem=32G                   # System memory per node
#SBATCH --time=00:15:00             # Time limit (15 minutes)
#SBATCH --output=ddp_%j.log
#SBATCH --error=ddp_%j.err

# Exit immediately if a command fails
set -euo pipefail

# 1. Identify Host Network Topology & Rendezvous Target
# Identify the primary node hostname for torch.distributed TCP rendezvous
NODES=($(scontrol show hostnames "$SLURM_JOB_NODELIST"))
MASTER_ADDR=${NODES[0]}
MASTER_PORT=29500

echo "=== Distributed Cluster Layout ==="
echo "Allocated Nodes : ${NODES[*]}"
echo "Primary Master  : ${MASTER_ADDR}:${MASTER_PORT}"
echo "Total Tasks     : ${SLURM_NTASKS}"
echo "=================================="

# 2. Configure NCCL Optimization & Tuning Variables
export NCCL_DEBUG=INFO                 # Print NCCL initialization diagnostics
export NCCL_DEBUG_SUBSYS=INIT,COLL,ENV # Output init, collective, and environment variables
export NCCL_IB_DISABLE=0               # Enforce InfiniBand utilization (0 = enabled)
export NCCL_NET_GDR_LEVEL=5            # Enable GPUDirect RDMA across NVLink/InfiniBand

# Identify high-speed network device interfaces (adjust to match cluster, e.g., ib0, mlx5_0, ens3f0)
# export NCCL_SOCKET_IFNAME=ib0

# 3. Define Container Image Location
CONTAINER_IMAGE="/ibex/user/$USER/pytorch_24.08.sif"

if [ ! -f "$CONTAINER_IMAGE" ]; then
    echo "[ERROR] Container image not found at $CONTAINER_IMAGE" >&2
    exit 1
fi

# 4. Launch Containerized Distributed Tasks with srun
# Pass Slurm task environment into each container as PyTorch DDP variables
srun --export=ALL bash -c "
    export RANK=\$SLURM_PROCID
    export WORLD_SIZE=\$SLURM_NTASKS
    export LOCAL_RANK=\$SLURM_LOCALID
    export MASTER_ADDR=$MASTER_ADDR
    export MASTER_PORT=$MASTER_PORT

    singularity exec \
        --nv \
        --cleanenv \
        --bind /dev/shm:/dev/shm \
        --bind /dev/infiniband:/dev/infiniband \
        --bind /ibex/user/$USER:/ibex/user/$USER \
        \"$CONTAINER_IMAGE\" \
        python3 /ibex/user/$USER/multinode_lab/ddp_benchmark.py
"
