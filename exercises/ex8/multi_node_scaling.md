# Exercise 8: Multi-GPU and Multi-Node Scaling with Singularity and Slurm

**Prerequisites:**

* An NGC PyTorch SIF container (`pytorch_24.08.sif`) or the custom PyTorch container built in prior exercises.
* High-speed scratch filesystem access (`/ibex/user/$USER`).
* Completion of Exercises 6 and 7.

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

Copy the exercise files into a workspace directory on high-speed scratch storage (run from this directory, `exercises/ex8`):

```bash
mkdir -p /ibex/user/$USER/multinode_lab
cp ddp_benchmark.py submit_multinode_ddp.slurm /ibex/user/$USER/multinode_lab/
cd /ibex/user/$USER/multinode_lab
```

### The DDP Benchmark ([`ddp_benchmark.py`](ddp_benchmark.py))

This script sets up a synthetic Distributed Data Parallel (DDP) workload, performs inter-GPU tensor reductions, and measures synchronization latency. It:
* Reads `RANK`, `WORLD_SIZE`, and `LOCAL_RANK` from the environment and initializes an NCCL process group.
* Wraps a three-layer MLP in `DistributedDataParallel`.
* Runs 10 warmup steps, then 50 timed steps, and has rank 0 report step latency, aggregate throughput, and per-GPU throughput.

---

## 3. Step 2: Author the Slurm Multi-Node Batch Script

[`submit_multinode_ddp.slurm`](submit_multinode_ddp.slurm) requests **2 compute nodes** with **2 GPUs each** (a total world size of 4), sets up environment rendezvous, and launches Singularity via `srun`. It:
1. Uses the first allocated node as `MASTER_ADDR` (port `29500`) for the TCP rendezvous.
2. Sets NCCL debugging and tuning variables (`NCCL_DEBUG`, `NCCL_IB_DISABLE=0`, `NCCL_NET_GDR_LEVEL=5`). Uncomment `NCCL_SOCKET_IFNAME` and set it to match your cluster's high-speed interface.
3. Checks that the SIF image exists at `CONTAINER_IMAGE`.
4. Maps each Slurm task's `SLURM_PROCID`, `SLURM_NTASKS`, and `SLURM_LOCALID` to PyTorch's `RANK`, `WORLD_SIZE`, and `LOCAL_RANK`, then runs `ddp_benchmark.py` inside the container.

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