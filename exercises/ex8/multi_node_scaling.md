# Exercise 8: Multi-GPU and Multi-Node Scaling with Singularity and Slurm

**Prerequisites:**

* An NGC PyTorch SIF container (`pytorch_24.08.sif`) or the custom PyTorch container built in prior exercises.
* High-speed scratch filesystem access (`/ibex/user/$USER`).
* Completion of Exercise 6 and  7.

---

## Overview & Objectives

Training large deep learning models requires distributing computation across multiple accelerators and physical servers. In modern AI/ML infrastructure:

* **Intra-node communication** (between GPUs on the same server) operates over high-bandwidth proprietary interconnects (e.g., NVIDIA NVLink/NVSwitch at up to $900\text{ GB/s}$).
* **Inter-node communication** (between separate cluster servers) runs across Remote Direct Memory Access (RDMA) networks using **InfiniBand (IB)** or **RoCE (RDMA over Converged Ethernet)**.
* **Collective Operations:** PyTorch delegates distributed tensor aggregation (e.g., `AllReduce`, `AllGather`) to the **NVIDIA Collective Communications Library (NCCL)**.

In this exercise, you will:

1. Author a distributed PyTorch script using the `torch.distributed` API and NCCL backend.
2. Build a multi-node Slurm batch submission script coordinating Singularity containers across multiple nodes and multiple GPUs per node.
3. Configure NCCL environment variables to ensure containerized processes access physical NVLink and InfiniBand hardware paths without virtualization overhead.
4. Launch the job across multiple nodes and monitor compute, memory, and network throughput in real time.
5. Troubleshoot distributed synchronization barriers, rendezvous failures, and NCCL network deadlocks.

---

## 1. Architectural Model: Orchestrating Multi-Node Containers

When running multi-node containerized deep learning jobs, the standard orchestration design pairs **Slurm** for process placement with **`torchrun`** or direct **`srun` step orchestration**:

```
[Slurm Controller]
       │ Launches step via srun across allocated nodes
       ├───────────────────────────────────────────────┐
       ▼                                               ▼
Compute Node 1 (Master / Rank 0)             Compute Node 2 (Worker / Rank 1)
┌───────────────────────────────────────┐   ┌───────────────────────────────────────┐
│ Slurm Task (Rank 0)                   │   │ Slurm Task (Rank 1)                   │
│   └─ singularity exec --nv            │   │   └─ singularity exec --nv            │
│       └─ Python DDP (Local Rank 0)    │   │       └─ Python DDP (Local Rank 0)    │
│       └─ Python DDP (Local Rank 1)    │   │       └─ Python DDP (Local Rank 1)    │
│       └─ ... (per GPU)                │   │       └─ ... (per GPU)                │
└───────────────────┬───────────────────┘   └───────────────────┬───────────────────┘
                    │                                           │
                    └─── [InfiniBand / RoCE Fabric (NCCL)] ─────┘
```

### Communication Planes:
1. **Rendezvous Plane (TCP):** Used during startup to coordinate IP addresses, ranks, and world sizes via a designated master node (`MASTER_ADDR:MASTER_PORT`).
2. **Data Plane (NCCL):** High-throughput tensor exchanges bypass host CPU memory using GPUDirect RDMA.

---

## 2. Step 1: Create the Distributed AI/ML Workload

Create a dedicated workspace directory on high-speed scratch storage:

```bash
mkdir -p /ibex/user/$USER/multinode_lab && cd /ibex/user/$USER/multinode_lab
```

### Create `ddp_benchmark.py`

This script sets up a synthetic Distributed Data Parallel (DDP) workload, performs inter-GPU tensor reductions, and measures synchronization latency:

```python
cat << 'EOF' > ddp_benchmark.py
import os
import sys
import time
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP

def setup_distributed():
    """Initializes the distributed process group using NCCL."""
    # Read environment variables set by the launcher/orchestrator
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    torch.cuda.set_device(local_rank)
    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        world_size=world_size,
        rank=rank
    )
    return rank, world_size, local_rank

def cleanup():
    dist.destroy_process_group()

def run_workload():
    rank, world_size, local_rank = setup_distributed()
    device = torch.device(f"cuda:{local_rank}")

    if rank == 0:
        print("=" * 60)
        print("    PyTorch DDP Multi-Node Distributed Benchmark")
        print("=" * 60)
        print(f"Total Distributed World Size : {world_size} ranks")
        print(f"Communication Backend        : {dist.get_backend()}")
        print("=" * 60)

    # Barrier synchronization to ensure clean diagnostic printing
    dist.barrier()
    time.sleep(rank * 0.05)
    print(f"[Node: {os.uname().nodename}] Rank {rank:02d}/{world_size:02d} bound to Local GPU {local_rank} ({torch.cuda.get_device_name(device)})")
    dist.barrier()

    # Define a benchmark neural network
    model = nn.Sequential(
        nn.Linear(16384, 8192),
        nn.ReLU(),
        nn.Linear(8192, 4096),
        nn.ReLU(),
        nn.Linear(4096, 10)
    ).to(device)

    # Wrap model with DistributedDataParallel
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    batch_size = 128
    num_iterations = 50
    warmup_steps = 10

    if rank == 0:
        print(f"\nStarting training loop: {num_iterations} iterations (Batch size per GPU: {batch_size})...")

    # Warmup loop
    for _ in range(warmup_steps):
        inputs = torch.randn(batch_size, 16384, device=device)
        targets = torch.randint(0, 10, (batch_size,), device=device)
        optimizer.zero_grad()
        loss = criterion(model(inputs), targets)
        loss.backward()
        optimizer.step()

    torch.cuda.synchronize()
    dist.barrier()

    start_time = time.time()
    for step in range(1, num_iterations + 1):
        step_start = time.time()
        inputs = torch.randn(batch_size, 16384, device=device)
        targets = torch.randint(0, 10, (batch_size,), device=device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()  # Triggers asynchronous all-reduce of gradients via NCCL
        optimizer.step()

        torch.cuda.synchronize()

        if rank == 0 and step % 10 == 0:
            step_latency_ms = (time.time() - step_start) * 1000
            print(f"Step [{step:02d}/{num_iterations:02d}] - Loss: {loss.item():.4f} - Step Latency: {step_latency_ms:.2f} ms")

    dist.barrier()
    total_time = time.time() - start_time
    total_samples = batch_size * world_size * num_iterations
    throughput = total_samples / total_time

    if rank == 0:
        print("\n" + "=" * 60)
        print("                  Benchmark Results")
        print("=" * 60)
        print(f"Total Processing Time    : {total_time:.2f} seconds")
        print(f"Aggregate System Throughput : {throughput:.2f} samples/sec")
        print(f"Per-GPU Throughput          : {(throughput / world_size):.2f} samples/sec/GPU")
        print("=" * 60 + "\n")

    cleanup()

if __name__ == "__main__":
    run_workload()
EOF
```

---

## 3. Step 2: Author the Slurm Multi-Node Batch Script

Create `submit_multinode_ddp.slurm`. This script requests **2 compute nodes** with **2 GPUs each** (a total world size of 4), setting up environment rendezvous before launching Singularity via `srun`.

```bash
cat << 'EOF' > submit_multinode_ddp.slurm
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
EOF
```

### Explaining Critical Slurm & Singularity Parameters

| Parameter | Purpose |
| :--- | :--- |
| `srun --export=ALL` | Propagates job allocation environment variables to all sub-processes across nodes. |
| `export RANK=$SLURM_PROCID` | Maps Slurm's global task index ($0 \dots \text{world\_size}-1$) to PyTorch's `RANK`. |
| `export LOCAL_RANK=$SLURM_LOCALID` | Maps Slurm's per-node task index ($0 \dots \text{gpus\_per\_node}-1$) to `LOCAL_RANK` for GPU device assignment. |
| `--bind /dev/infiniband` | Gives the containerized user-space verbs driver direct access to host InfiniBand adapters. |
| `--bind /dev/shm:/dev/shm` | Shares the host POSIX shared memory pool to prevent inter-process IPC buffer exhaustion. |
| `NCCL_NET_GDR_LEVEL=5` | Permits GPUDirect RDMA over PCIe and NVLink switches, skipping intermediate host RAM copies. |

---

## 4. Step 3: Launch and Verify the Job

Submit the multi-node job using `sbatch`:

```bash
sbatch submit_multinode_ddp.slurm
```

### Monitor the Queue
Track job transition from `PD` (Pending) to `R` (Running):

```bash
squeue -u $USER
```

---

## 5. Step 4: Monitor Resource Utilization

To verify multi-node efficiency, monitor resource utilization across GPU, CPU, and network planes during execution.

### 5.1 Remote GPU Profiling via `srun`
While the job is running, run `nvidia-smi` across both nodes using an interactive step:

```bash
# Query active GPU utilization across all nodes in the active job allocation
srun --jobid=<JOB_ID> --nodes=2 --ntasks-per-node=1 nvidia-smi --query-gpu=timestamp,node,index,utilization.gpu,utilization.memory,memory.used,memory.total --format=csv
```

### 5.2 InfiniBand Fabric Activity
Verify that collective operations flow through the high-speed fabric instead of standard Ethernet:

```bash
# Check InfiniBand port counters on the host compute nodes
srun --jobid=<JOB_ID> perfquery
```

---

## 6. Step 5: Review Logs & Diagnostics

Once the job terminates, inspect the generated logs:

```bash
cat ddp_<JOB_ID>.log
```

### Expected Output Anatomy

#### 1. NCCL Hardware Detection & Channel Synthesis
Because `NCCL_DEBUG=INFO` was set, the log will first show NCCL scanning system topology, finding NVLink paths, and locating InfiniBand Host Channel Adapters (HCAs):

```text
node01:12345:12345 [0] NCCL INFO Channel 00/08 : 0 1 2 3
node01:12345:12345 [0] NCCL INFO Using network IB [0:mlx5_0:1/RoCE]; O_DIRECT=1
node01:12345:12345 [0] NCCL INFO Trees [0] 1/-1/-1->0->-1 [1] -1/-1/-1->0->1
node01:12345:12345 [0] NCCL INFO Ring 00 : 0 -> 1 -> 2 -> 3 -> 0
node01:12345:12345 [0] NCCL INFO Comm init complete ...
```

#### 2. Process Output and Benchmark Results
```text
============================================================
    PyTorch DDP Multi-Node Distributed Benchmark
============================================================
Total Distributed World Size : 4 ranks
Communication Backend        : nccl
============================================================
[Node: node01] Rank 00/04 bound to Local GPU 0 (NVIDIA A100-SXM4-80GB)
[Node: node01] Rank 01/04 bound to Local GPU 1 (NVIDIA A100-SXM4-80GB)
[Node: node02] Rank 02/04 bound to Local GPU 0 (NVIDIA A100-SXM4-80GB)
[Node: node02] Rank 03/04 bound to Local GPU 1 (NVIDIA A100-SXM4-80GB)

Starting training loop: 50 iterations (Batch size per GPU: 128)...
Step [10/50] - Loss: 2.3026 - Step Latency: 12.35 ms
Step [20/50] - Loss: 2.2984 - Step Latency: 12.21 ms
Step [30/50] - Loss: 2.2910 - Step Latency: 12.18 ms
Step [40/50] - Loss: 2.2855 - Step Latency: 12.24 ms
Step [50/50] - Loss: 2.2798 - Step Latency: 12.19 ms

============================================================
                  Benchmark Results
============================================================
Total Processing Time       : 0.61 seconds
Aggregate System Throughput : 10491.80 samples/sec
Per-GPU Throughput          : 2622.95 samples/sec/GPU
============================================================
```

---

## 7. Troubleshooting Distributed Multi-Node Jobs

| Issue / Error Output | Root Cause | Remediation |
| :--- | :--- | :--- |
| **`Connection refused` on rendezvous address** | Target master port (e.g., `29500`) blocked by local host firewall or incorrect node hostname. | Verify `MASTER_ADDR` matches `scontrol show hostnames`. Ensure port range is accessible within the cluster internal fabric. |
| **`NCCL WARN Call to connect returned Connection refused`** | NCCL attempted to connect over the wrong network interface (e.g., standard Docker bridge `docker0` instead of `ib0`). | Set `export NCCL_SOCKET_IFNAME=ib0` (or your cluster's high-speed subnet name) inside the Slurm job script. |
| **Job freezes at `dist.init_process_group` without error** | Process deadlock due to rank mismatch or dropped TCP packets between nodes. | Enable `export NCCL_DEBUG=INFO` and inspect `ddp_<JOB_ID>.err` to locate the specific rank that failed to check in. |
| **`NCCL WARN Network setup failed : GdrCopy support not available`** | GPUDirect driver missing or `/dev/infiniband` was not bound to the container. | Ensure `--bind /dev/infiniband:/dev/infiniband` is present in the `singularity exec` line. |
| **Sub-linear scaling / high step latency** | Communication bottleneck; NCCL fell back to host sockets rather than InfiniBand/NVLink. | Verify that `NCCL_IB_DISABLE=0` is set and host InfiniBand modules (`ib_uverbs`) are active. |

---

## 8. Hands-On Verification Challenges


1. **Evaluate Scaling Efficiency:**
   Modify the script to run on a single node with 2 GPUs, then compare aggregate throughput with the 2-node (4 GPU) run:
   $$\text{Scaling Efficiency} = \frac{\text{Throughput}_{4\text{ GPUs}}}{2 \times \text{Throughput}_{2\text{ GPUs}}} \times 100\%$$
   *Is the scaling efficiency greater than 90%? If not, inspect your NCCL interconnect diagnostics.*

2. **Check Shared Memory Allocation:**
   Remove `--bind /dev/shm:/dev/shm` from the Singularity command line, increase the model batch size to `1024`, and observe whether PyTorch's dataloader or NCCL shared-memory transport returns an IPC bus error.