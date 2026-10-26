#!/bin/bash
#SBATCH --job-name=singularity_mpi
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=2
#SBATCH --mem=8GB
#SBATCH --time=00:10:00
#SBATCH --output=mpi_job_%j.log
#SBATCH --error=mpi_job_%j.err

echo "=========================================================="
echo " Starting Singularity MPI Multi-Node Execution"
echo " Slurm Job ID     : $SLURM_JOB_ID"
echo " Allocated Nodes  : $SLURM_JOB_NODELIST"
echo " Total Tasks      : $SLURM_NTASKS (2 nodes x 2 tasks/node)"
echo "=========================================================="

# 1. Purge conflicting host environment modules
module purge 2>/dev/null || true

# 2. Set Singularity and Network Environment Variables
# Prevent OpenMPI from trying to use ssh to launch remote tasks
export OMPI_MCA_plm=rsh
export OMPI_MCA_plm_rsh_agent=false

# Specify PMI interface (adjust to 'pmi2' or 'pmix' depending on cluster config)
MPI_TYPE="pmi2"

# Ensure host-level temporary directory is used
export SINGULARITY_TMPDIR=/ibex/user/$USER/tmp
mkdir -p "$SINGULARITY_TMPDIR"

WORK_DIR="/ibex/user/$USER/mpi_singularity_lab"
CONTAINER_IMG="${WORK_DIR}/mpi_stack.sif"

echo -e "\n--- Test 1: Running C MPI Ring Benchmark ---"
srun --mpi=${MPI_TYPE} \
     singularity exec \
     --bind /dev/shm \
     --bind /dev/infiniband \
     --bind ${WORK_DIR}:${WORK_DIR} \
     --pwd ${WORK_DIR} \
     "$CONTAINER_IMG" ./mpi_ring

echo -e "\n--- Test 2: Running Python AllReduce Benchmark ---"
srun --mpi=${MPI_TYPE} \
     singularity exec \
     --bind /dev/shm \
     --bind /dev/infiniband \
     --bind ${WORK_DIR}:${WORK_DIR} \
     --pwd ${WORK_DIR} \
     "$CONTAINER_IMG" python3 mpi_allreduce.py

echo -e "\nExecution finished with exit code: $?"
