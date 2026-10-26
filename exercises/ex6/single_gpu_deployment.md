# Exercise 6: Single-GPU Deployment with Slurm and Singularity

**Prerequisites:**
* A built or pulled Singularity image from previous exercises (e.g., `pytorch_24.08.sif` or `custom-ml-stack.sif`).
* High-speed scratch directory access (`/ibex/user/$USER`).

---

## Overview & Objectives

In production HPC environments, compute nodes are managed by job schedulers such as **Slurm**. Rather than launching containers interactively on login nodes, researchers submit batch jobs that request dedicated hardware resources (CPUs, system memory, and GPUs).

Running Singularity within Slurm jobs ensures reproducibility, eliminates software dependency conflicts across multi-tenant clusters, and enforces isolated hardware access.

In this exercise, you will:
1. Prepare an AI/ML training and benchmark workload script in PyTorch.
2. Author a production-ready Slurm batch submission script (`.slurm`) wrapping the Singularity container.
3. Manage host-to-container volume bindings and shared memory (`/dev/shm`) configuration.
4. Submit the job, track its execution in the Slurm queue, and verify GPU allocation.
5. Review output logs and diagnose common containerized GPU failure modes.

---

## 1. Prepare the AI/ML Test Workload

Copy the exercise files into a workspace directory on your scratch filesystem (run from this directory, `exercises/ex6`):

```bash
mkdir -p /ibex/user/$USER/slurm_gpu_lab
cp train_benchmark.py submit_gpu.slurm /ibex/user/$USER/slurm_gpu_lab/
cd /ibex/user/$USER/slurm_gpu_lab
```

### The Training Benchmark Script ([`train_benchmark.py`](train_benchmark.py))

This script performs synthetic tensor operations and a short training loop, logging GPU device properties, memory allocation, and training throughput. It exits with an error if CUDA is not available.

---

## 2. Writing the Slurm Batch Script

Slurm coordinates compute tasks using resource directives (`#SBATCH`) interpreted before the script executes.

### 2.1 Key Slurm Directives for GPUs

* `--gpus=1` (or `--gres=gpu:1`): Requests one dedicated GPU device. Slurm will assign an available card and export `CUDA_VISIBLE_DEVICES`.
* `--cpus-per-task=4`: Allocates CPU cores for background data ingestion and pre-processing.
* `--mem=16G`: Allocates system host RAM.
* `--time=00:15:00`: Sets a hard time limit ($15$ minutes) to prevent run-away jobs.

### 2.2 Shared Memory (`/dev/shm`) and Bind Mounts

PyTorch `DataLoader` workers use shared memory (`/dev/shm`) for inter-process communication (IPC). In containers, default `/dev/shm` mounts are often limited or missing, leading to `RuntimeError: DataLoader worker ... was killed by signal: Bus error`.

To prevent this:
* Pass `--bind /dev/shm` to grant the container access to host shared memory.
* Pass `--bind /ibex/user/$USER:/ibex/user/$USER` so the container can read data and write checkpoint files directly to high-performance storage.

### 2.3 The Batch Script ([`submit_gpu.slurm`](submit_gpu.slurm))

The script requests 1 node, 1 task, 4 CPUs, 1 GPU, 16 GB of RAM, and 15 minutes, then:
1. Loads the Singularity (or Apptainer) module.
2. Checks that the SIF image exists at `SIF_IMAGE`.
3. Points the Singularity cache and temp directories to scratch.
4. Launches `train_benchmark.py` with `srun singularity exec`, using `--nv` (host NVIDIA drivers), `--cleanenv` (no host environment leakage), `--bind` for `/dev/shm` and the work directory, and `--pwd` to set the starting directory.

> **Note:** If your SIF image is located in a different directory (such as where you pulled it in Exercise 1), adjust the `SIF_IMAGE` variable in `submit_gpu.slurm` accordingly.

---

## 3. Submitting the Job and Verifying GPU Access

### 3.1 Submit the Job with `sbatch`

Submit your batch script to the Slurm queue:

```bash
sbatch submit_gpu.slurm
```

*Expected output:*
```text
Submitted batch job 1849204
```

### 3.2 Inspect Job Status in the Queue

Query the state of your job using `squeue`:

```bash
# Check status for your username
squeue -u $USER
```

*Status Legend:*
* `PD` (Pending): The job is waiting for compute resources or GPU allocation.
* `R` (Running): The job is executing on the assigned node.
* `CG` (Completing): The job has finished and resources are being deallocated.

To view detailed real-time properties of the allocated job:

```bash
scontrol show job <JOB_ID>
```

---

## 4. Reviewing Output Logs & Diagnostics

Once the job changes from `R` to completed, two log files will be generated in your working directory:
* `slurm-<JOB_ID>.out`: Standard output (`stdout`).
* `slurm-<JOB_ID>.err`: Standard error (`stderr`).

### 4.1 Inspecting Standard Output

Read the generated output log:

```bash
cat slurm-<JOB_ID>.out
```

*Expected Successful Output:*

```text
==========================================================
Job Execution Started: Sun Oct 25 09:12:01 UTC 2026
Node Allocated: gpu-node-042
Assigned GPU Index: 0
==========================================================

==================== Environment & Hardware Diagnostics ====================
Host Hostname           : gpu-node-042
Slurm Job ID            : 1849204
Python Binary           : /opt/conda/bin/python3
PyTorch Version         : 2.4.0+cu124
CUDA Available          : True
CUDA_VISIBLE_DEVICES    : 0
Active GPU Device       : [0] NVIDIA A100-SXM4-80GB
Compute Capability      : 8.0
Total Dedicated Memory  : 80.00 GB

==================== Running Synthetic Neural Network Benchmark ====================
Batch Size: 256 | Iterations: 100
Step [025/100] - Loss: 2.3014 | VRAM Allocated: 112.5 MB | VRAM Reserved: 128.0 MB
Step [050/100] - Loss: 2.2891 | VRAM Allocated: 112.5 MB | VRAM Reserved: 128.0 MB
Step [075/100] - Loss: 2.2743 | VRAM Allocated: 112.5 MB | VRAM Reserved: 128.0 MB
Step [100/100] - Loss: 2.2512 | VRAM Allocated: 112.5 MB | VRAM Reserved: 128.0 MB

==================== Benchmark Summary ====================
Total Elapsed Time      : 1.84 s
Training Throughput     : 13913.04 samples/s
Peak VRAM Usage         : 120.3 MB
Job execution completed successfully.

==========================================================
Job Execution Finished: Sun Oct 25 09:12:06 UTC 2026
==========================================================
```

### 4.2 Inspecting Standard Error

Verify that no unhandled exceptions were reported:

```bash
cat slurm-<JOB_ID>.err
```

If the run succeeded cleanly, this file will be empty or will only contain standard third-party informational messages.

---

## 5. Troubleshooting & Common Failure Modes

### Failure Mode 1: Missing `--nv` Flag
* **Symptom:** In `slurm-<JOB_ID>.out`:
  ```text
  CUDA Available: False
  [ERROR] CUDA is not available to PyTorch. Exiting.
  ```
* **Root Cause:** Without `--nv`, Singularity creates an isolated namespace where `/dev/nvidia*` device nodes and host driver dynamic libraries (`libcuda.so`, `libnvidia-ml.so`) are masked.
* **Resolution:** Ensure `singularity exec` or `singularity run` includes the `--nv` flag.

---

### Failure Mode 2: CUDA Driver Compatibility Mismatch
* **Symptom:**
  ```text
  RuntimeError: The NVIDIA driver on your system is too old (found version 11080)
  and does not support CUDA 12.4. Please update your driver.
  ```
* **Root Cause:** The host kernel module (`nvidia.ko`) is running an older driver version than required by the CUDA toolkit bundled in the container.
* **Compatibility Rule:**
  $$\text{Container CUDA Toolkit Version} \le \text{Host NVIDIA Driver Maximum Supported CUDA Version}$$
* **Resolution:** Check host driver version with `nvidia-smi` on the node. Choose a container image built with a compatible CUDA version (e.g., fallback from CUDA 12.4 to CUDA 12.1 or 11.8).

---

### Failure Mode 3: Host vs Container Path Discrepancy
* **Symptom:**
  ```text
  python3: can't open file 'train_benchmark.py': [Errno 2] No such file or directory
  ```
* **Root Cause:** By default, Singularity mounts `$HOME`, `/tmp`, and current working directory, but on some clusters, high-speed scratch spaces (e.g., `/ibex/user/$USER`) are **not** mounted automatically.
* **Resolution:** Explicitly mount your directories using the `--bind` flag:
  ```bash
  --bind /ibex/user/$USER:/ibex/user/$USER --pwd /ibex/user/$USER/slurm_gpu_lab
  ```

---

### Failure Mode 4: DataLoader Bus Error (Shared Memory Exhaustion)
* **Symptom:**
  ```text
  RuntimeError: DataLoader worker (pid 41235) was killed by signal: Bus error.
  It is possible that the system has lost a memory mapped file or you are out of shared memory.
  ```
* **Root Cause:** PyTorch inter-process communication relies on `/dev/shm`. Containers default to minimal shared memory allocations unless mapped to the host.
* **Resolution:** Add `--bind /dev/shm:/dev/shm` to your `singularity exec` command.

---

### Failure Mode 5: Environment Variable Leakage
* **Symptom:** Unexplained `ImportError` loading libraries from `/home/<user>/.local/lib/pythonX.Y/site-packages` instead of the container's environment.
* **Root Cause:** Singularity inherits user shell environment variables by default, which can cause Python to prioritize local user-installed wheels over containerized wheels.
* **Resolution:** Add the `--cleanenv` (or `-e`) flag to prevent host environment variable propagation:
  ```bash
  singularity exec --nv --cleanenv ...
  ```

---

## 6. Summary: Slurm + Singularity Best Practice Matrix

| Parameter / Flag | Recommended Setting | Purpose |
| :--- | :--- | :--- |
| **GPU Request** | `#SBATCH --gpus=1` | Instructs Slurm to bind 1 GPU and set `CUDA_VISIBLE_DEVICES` |
| **Driver Binding** | `--nv` | Injects host NVIDIA user-space libraries and GPU character devices |
| **Environment** | `--cleanenv` | Prevents host `$PYTHONPATH` and variables from conflicting with container |
| **Shared Memory** | `--bind /dev/shm:/dev/shm` | Prevents PyTorch DataLoader worker crashes |
| **Storage Binding** | `--bind /ibex/user/$USER:/ibex/user/$USER` | Grants read/write access to parallel high-performance scratch |
| **Execution Tool** | `srun singularity exec ...` | Ensures task tracking, accounting, and cgroup enforcement |

---

## Hands-On Verification Challenges

1. **GPU Isolation Verification:**
   Modify `submit_gpu.slurm` to query `nvidia-smi` inside the container:
   ```bash
   srun singularity exec --nv ${SIF_IMAGE} nvidia-smi
   ```
   *Question: How many GPUs does `nvidia-smi` report inside the container when requesting `--gpus=1` on an 8-GPU node?*

2. **Benchmarking CPU vs GPU Execution:**
   Temporarily remove the `--nv` flag from your execution command in `submit_gpu.slurm` and resubmit.
   *Question: At what line does `train_benchmark.py` fail, and what error is recorded in `slurm-<JOB_ID>.err`?*

3. **Tracking Live Output:**
   Submit a job and monitor the output stream in real-time as it executes on the compute node:
   ```bash
   sbatch submit_gpu.slurm
   tail -f slurm-<JOB_ID>.out
   ```