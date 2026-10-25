# Exercise 7: MPI Compatibility and Inter-Container Communication

**Prerequisites:**

* Completed Exercises 5 and 6.

---

## Overview & Objectives

In large-scale AI/ML (such as distributed foundation model training with Megatron-LM, DeepSpeed, or PyTorch Distributed) and traditional HPC scientific applications, processes across multiple nodes must communicate with low latency and high bandwidth via the **Message Passing Interface (MPI)**.

Running MPI inside containers introduces a unique challenge: the containerized application must communicate across the host's specialized high-speed interconnect fabric (such as InfiniBand or RoCE) while being coordinated by the host's workload manager (Slurm).

In this exercise, you will:

1. Understand the two primary container MPI architectures: the **Hybrid Model** and the **Bind Model**.
2. Learn the compatibility rules governing **MPICH** (ABI compatibility) and **OpenMPI** (PMIx/PMI2 process management).
3. Bind host network interfaces, InfiniBand devices, and driver libraries into Singularity.
4. Author and compile an MPI communication test program.
5. Launch and verify multi-process, inter-container MPI communication using Slurm `srun`.
6. Troubleshoot common MPI runtime errors, fabric mismatches, and PMI protocol failures.

---

## 1. Architectural Foundations: Host MPI vs. Container MPI

When orchestrating containerized MPI jobs across multiple compute nodes, two architectural patterns exist:

```
                      HYBRID MODEL (Recommended)
Host Node 1                                           Host Node 2
┌──────────────────────────────┐                      ┌──────────────────────────────┐
│  Slurm Job Step (srun)       │                      │  Slurm Job Step (srun)       │
│  Host MPI Launcher / PMIx    │                      │  Host MPI Launcher / PMIx    │
│              │               │                      │              │               │
│              ▼               │                      │              ▼               │
│   singularity exec           │                      │   singularity exec           │
│   ┌──────────────────────┐   │                      │   ┌──────────────────────┐   │
│   │ Container MPI Lib    │   │                      │   │ Container MPI Lib    │   │
│   │ Application Rank 0   │   │                      │   │ Application Rank 1   │   │
│   └──────────┬───────────┘   │                      │   └──────────┬───────────┘   │
└──────────────┼───────────────┘                      └──────────────┼───────────────┘
               │                                                     │
               └─────────── High-Speed Fabric (InfiniBand/UCX) ──────┘
```

### 1.1 The Hybrid Model
* **How it works:** The host environment supplies the job launcher (`srun` or `mpirun`) and the Process Management Interface (PMIx or PMI2). The container contains its own MPI runtime libraries and compiled application.
* **The Handshake:** When `srun` spawns `singularity`, the host PMIx server handshakes with the container's MPI client library over a UNIX socket or environment file descriptors.
* **Requirements:**
  * The MPI implementation inside the container must be wire-compatible and release-compatible with the host's MPI or PMI standard.
  * The container must have access to the host's network interconnect devices (e.g., `/dev/infiniband`).

### 1.2 The Bind Model
* **How it works:** The container contains *no* MPI libraries or contains only dummy stubs. At runtime, the host's MPI installation directories, Open Fabrics Enterprise Distribution (OFED) user-space libraries, and configuration files are dynamically mounted into the container using `--bind`.
* **Pros:** Guarantees absolute version match with cluster fabric drivers.
* **Cons:** Fragile; breaks container portability across different clusters with differing library paths, `glibc` versions, or driver architectures.

---

## 2. Choosing Your MPI Flavor: MPICH vs. OpenMPI

| Feature | OpenMPI Ecosystem | MPICH Ecosystem (MPICH, Intel MPI, MVAPICH, Cray MPT) |
| :--- | :--- | :--- |
| **Interoperability Standard** | Requires matching major versions (e.g., v4.x host $\leftrightarrow$ v4.x container) | Follows the **MPICH ABI Compatibility Initiative** |
| **Process Management** | Primarily **PMIx** (Process Management Interface Exascale) | Primarily **PMI2** or **PMIx** |
| **Transport Layer** | UCX (Unified Communication X) or native OFI | libfabric (OFI) or UCX |
| **Best Used For** | PyTorch Distributed, OpenMPI cluster deployments | Clusters standardized on Intel MPI, MVAPICH2, or Cray EX |

### The MPICH ABI Compatibility Advantage
Under the MPICH ABI Compatibility Initiative, binaries compiled against MPICH can dynamically link against and execute with Intel MPI, MVAPICH2, or Cray MPT without recompilation, provided the ABI major version is compatible.

### The OpenMPI PMIx Compatibility Rule
OpenMPI relies heavily on PMIx for wire-level process wire-up. For OpenMPI in Singularity:
* The host Slurm must support `--mpi=pmix` or `--mpi=pmi2`.
* The PMIx client version inside the container must fall within the backward-compatibility window of the host's PMIx server.

---

## 3. Step 1: Create the MPI Communication Test Suite

Create a dedicated workspace on high-speed scratch storage:

```bash
mkdir -p /ibex/user/$USER/mpi_singularity_lab && cd /ibex/user/$USER/mpi_singularity_lab
```

### 3.1 C-Based Ring Communication Benchmark (`mpi_ring.c`)
This program passes a message sequentially through all ranks across all containers, verifying inter-node connectivity and latency:

```c
cat << 'EOF' > mpi_ring.c
#define _GNU_SOURCE
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int world_rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    char hostname[256];
    gethostname(hostname, sizeof(hostname));

    printf("[Rank %d of %d] Initialized on host: %s\n", world_rank, world_size, hostname);
    MPI_Barrier(MPI_COMM_WORLD);

    int token;
    double start_time = MPI_Wtime();

    if (world_rank == 0) {
        token = 42;
        printf("\n>>> [Rank 0] Originating token %d to Rank 1...\n", token);
        MPI_Send(&token, 1, MPI_INT, (world_rank + 1) % world_size, 0, MPI_COMM_WORLD);
        MPI_Recv(&token, 1, MPI_INT, world_size - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        printf(">>> [Rank 0] Successfully received token %d back from Rank %d!\n", token, world_size - 1);
    } else {
        MPI_Recv(&token, 1, MPI_INT, world_rank - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        printf("    [Rank %d on %s] Received token %d, forwarding to Rank %d...\n", 
               world_rank, hostname, token, (world_rank + 1) % world_size);
        MPI_Send(&token, 1, MPI_INT, (world_rank + 1) % world_size, 0, MPI_COMM_WORLD);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double end_time = MPI_Wtime();

    if (world_rank == 0) {
        printf("\nRing pass completed in %f milliseconds across %d ranks.\n\n", 
               (end_time - start_time) * 1000.0, world_size);
    }

    MPI_Finalize();
    return 0;
}
EOF
```

### 3.2 Python AI/ML AllReduce Test (`mpi_allreduce.py`)
This script uses `mpi4py` to emulate the gradient-reduction patterns utilized by distributed deep learning engines:

```python
cat << 'EOF' > mpi_allreduce.py
import os
import socket
import numpy as np
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
host = socket.gethostname()

print(f"Python MPI Rank {rank:02d}/{size:02d} online on {host}")
comm.Barrier()

# Create a synthetic gradient tensor (1 million floats = 4 MB per rank)
tensor_size = 1_000_000
send_data = np.ones(tensor_size, dtype=np.float32) * (rank + 1)
recv_data = np.zeros(tensor_size, dtype=np.float32)

t_start = MPI.Wtime()
comm.Allreduce(send_data, recv_data, op=MPI.SUM)
t_end = MPI.Wtime()

# Expected sum formula: n * (n + 1) / 2
expected_value = (size * (size + 1)) / 2.0
is_correct = np.allclose(recv_data[0], expected_value)

comm.Barrier()
if rank == 0:
    elapsed_ms = (t_end - t_start) * 1000.0
    mb_transferred = (tensor_size * 4) / (1024 * 1024)
    print(f"\n[AllReduce Validation]")
    print(f"Ranks Participating : {size}")
    print(f"Tensor Payload Size : {mb_transferred:.2f} MB")
    print(f"AllReduce Duration  : {elapsed_ms:.2f} ms")
    print(f"Result Verification : {'PASSED' if is_correct else 'FAILED'}")
    print(f"Value at Index [0]  : {recv_data[0]} (Expected: {expected_value})\n")
EOF
```

---

## 4. Step 2: Build an MPI-Enabled Container

We will write a Singularity definition file that compiles OpenMPI with UCX support and configures `mpi4py`.

```singularity
cat << 'EOF' > mpi_stack.def
Bootstrap: docker
From: ubuntu:22.04

%labels
    Author WorkshopPresenter
    Description OpenMPI and mpi4py Singularity Stack

%environment
    export LC_ALL=C.UTF-8
    export LANG=C.UTF-8
    export PATH=/opt/openmpi/bin:$PATH
    export LD_LIBRARY_PATH=/opt/openmpi/lib:$LD_LIBRARY_PATH

%post
    export DEBIAN_FRONTEND=noninteractive
    apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        gfortran \
        wget \
        curl \
        ca-certificates \
        python3 \
        python3-pip \
        python3-dev \
        libibverbs-dev \
        libnl-3-200 \
        libnl-route-3-200 \
        libnuma1 \
        git \
        && rm -rf /var/lib/apt/lists/*

    # 1. Install OpenMPI from source (matches typical cluster OpenMPI 4.1.x)
    cd /tmp
    wget https://download.open-mpi.org/release/open-mpi/v4.1/openmpi-4.1.6.tar.gz
    tar -xzf openmpi-4.1.6.tar.gz
    cd openmpi-4.1.6
    ./configure --prefix=/opt/openmpi \
                --enable-mpi-cxx \
                --with-pmi \
                --with-pmix=internal \
                --without-verbs
    make -j$(nproc)
    make install
    cd / && rm -rf /tmp/openmpi*

    # 2. Configure Environment for Python bindings
    export PATH=/opt/openmpi/bin:$PATH
    export LD_LIBRARY_PATH=/opt/openmpi/lib:$LD_LIBRARY_PATH

    # 3. Install Python Dependencies
    python3 -m pip install --no-cache-dir --upgrade pip setuptools wheel
    python3 -m pip install --no-cache-dir numpy
    MPICC=/opt/openmpi/bin/mpicc python3 -m pip install --no-cache-dir mpi4py

%runscript
    exec "$@"
EOF
```

### Build the Image
```bash
singularity build --fakeroot mpi_stack.sif mpi_stack.def
```

### Compile the C Benchmark Inside the Container
Using `singularity exec`, compile the benchmark using the container's internal `mpicc`:

```bash
singularity exec mpi_stack.sif mpicc -O3 mpi_ring.c -o mpi_ring
```

---

## 5. Step 3: Interconnect Binding & Host Integration

To achieve near-native latency, the container must bind directly to the host's InfiniBand or RoCE hardware devices and system shared memory.

### 5.1 Required Host Device Binds
* `/dev/infiniband`: Contains user-space verbs devices (`uverbs0`, `rdma_cm`).
* `/dev/shm`: Shared memory directory used for zero-copy intra-node communication between ranks on the same physical server.

Add these binds to your Singularity execution flags:
```bash
--bind /dev/infiniband --bind /dev/shm
```

### 5.2 Host Library Binding (If Using Host MPI Bind Model)
If your cluster uses vendor-proprietary drivers (e.g., Cray MPT or Mellanox OFED) not bundled into the base container, bind the host runtime libraries dynamically:

```bash
# Example binding host OpenFabrics libraries
export SINGULARITY_BINDPATH="/usr/lib64/libibverbs.so.1,/usr/lib64/librdmacm.so.1,/usr/lib64/libmlx5.so.1"
```

---

## 6. Step 4: Slurm Multi-Node MPI Job Submission

In modern HPC environments, **Slurm acts as the process manager**. You should **not** invoke `mpirun` inside the container. Instead, `srun` launches multiple instances of `singularity exec` across your allocation.

```
       srun (Host Slurm)
         ├──> Host Node 1 ──> singularity exec mpi_stack.sif ./mpi_ring (Rank 0)
         ├──> Host Node 1 ──> singularity exec mpi_stack.sif ./mpi_ring (Rank 1)
         ├──> Host Node 2 ──> singularity exec mpi_stack.sif ./mpi_ring (Rank 2)
         └──> Host Node 2 ──> singularity exec mpi_stack.sif ./mpi_ring (Rank 3)
```

### 6.1 Create the Slurm Batch Script (`submit_mpi.slurm`)

```bash
cat << 'EOF' > submit_mpi.slurm
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

CONTAINER_IMG="/ibex/user/$USER/mpi_singularity_lab/mpi_stack.sif"

echo -e "\n--- Test 1: Running C MPI Ring Benchmark ---"
srun --mpi=${MPI_TYPE} \
     singularity exec \
     --bind /dev/shm \
     --bind /dev/infiniband \
     "$CONTAINER_IMG" /app/mpi_ring

echo -e "\n--- Test 2: Running Python AllReduce Benchmark ---"
srun --mpi=${MPI_TYPE} \
     singularity exec \
     --bind /dev/shm \
     --bind /dev/infiniband \
     "$CONTAINER_IMG" python3 mpi_allreduce.py

echo -e "\nExecution finished with exit code: $?"
EOF
```

### 6.2 Submit and Monitor the Job

Submit the job to the cluster scheduler:

```bash
sbatch submit_mpi.slurm
```

Check queue status:

```bash
squeue -u $USER
```

---

## 7. Step 5: Reviewing Logs and Verifying Inter-Node Wire-Up

Once the job terminates, inspect the standard output log:

```bash
cat mpi_job_*.log
```

### Expected Output:
```text
==========================================================
 Starting Singularity MPI Multi-Node Execution
 Slurm Job ID     : 981245
 Allocated Nodes  : node042,node043
 Total Tasks      : 4 (2 nodes x 2 tasks/node)
==========================================================

--- Test 1: Running C MPI Ring Benchmark ---
[Rank 0 of 4] Initialized on host: node042
[Rank 1 of 4] Initialized on host: node042
[Rank 2 of 4] Initialized on host: node043
[Rank 3 of 4] Initialized on host: node043

>>> [Rank 0] Originating token 42 to Rank 1...
    [Rank 1 on node042] Received token 42, forwarding to Rank 2...
    [Rank 2 on node043] Received token 42, forwarding to Rank 3...
    [Rank 3 on node043] Received token 42, forwarding to Rank 0...
>>> [Rank 0] Successfully received token 42 back from Rank 3!

Ring pass completed in 0.482100 milliseconds across 4 ranks.

--- Test 2: Running Python AllReduce Benchmark ---
Python MPI Rank 00/04 online on node042
Python MPI Rank 01/04 online on node042
Python MPI Rank 02/04 online on node043
Python MPI Rank 03/04 online on node043

[AllReduce Validation]
Ranks Participating : 4
Tensor Payload Size : 3.81 MB
AllReduce Duration  : 2.15 ms
Result Verification : PASSED
Value at Index [0]  : 10.0 (Expected: 10.0)

Execution finished with exit code: 0
```

---

## 8. Troubleshooting Matrix: Common Container MPI Failures

| Symptom / Error Message | Root Cause | Remediation |
| :--- | :--- | :--- |
| `PMIx has detected a temporary server error` or `PMIx server mismatch` | The host PMIx version does not match the container client version. | Switch Slurm launcher to standard PMI2 by setting `srun --mpi=pmi2` instead of `pmix`. |
| `No OpenFabrics device found` / `Failed to open /dev/infiniband/uverbs0` | Interconnect hardware devices are unexposed to container namespaces. | Pass `--bind /dev/infiniband` to your `singularity exec` command. |
| `All ranks report rank 0 of 1` | `srun` executed the container multiple times as independent serial jobs rather than a coordinated MPI step. | Ensure `--mpi=pmi2` or `--mpi=pmix` is passed to `srun`, and verify that MPI inside the image was compiled with `--with-pmi`. |
| `UCX ERROR: Transport error: Cannot allocate memory` | Intra-node shared memory pool (`/dev/shm`) exhausted. | Explicitly bind the host shared memory mount: `--bind /dev/shm`. |
| `mpirun: command not found inside container` | Attempted to use the host MPI model without embedding the launcher binaries. | Use the **Hybrid Model**: invoke `srun` on the host, passing `singularity exec <image> <app_binary>`. |

