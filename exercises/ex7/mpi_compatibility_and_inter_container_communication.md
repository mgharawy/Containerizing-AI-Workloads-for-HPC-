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
OpenMPI relies heavily on PMIx for process wire-up. For OpenMPI in Singularity:
* The host Slurm must support `--mpi=pmix` or `--mpi=pmi2`.
* The PMIx client version inside the container must fall within the backward-compatibility window of the host's PMIx server.

---

## 3. Step 1: Create the MPI Communication Test Suite

Copy the exercise files into a workspace on high-speed scratch storage (run from this directory, `exercises/ex7`):

```bash
mkdir -p /ibex/user/$USER/mpi_singularity_lab
cp mpi_ring.c mpi_allreduce.py mpi_stack.def submit_mpi.slurm /ibex/user/$USER/mpi_singularity_lab/
cd /ibex/user/$USER/mpi_singularity_lab
```

### 3.1 C-Based Ring Communication Benchmark ([`mpi_ring.c`](mpi_ring.c))
This program passes a token sequentially through all ranks across all containers and times the full ring, verifying inter-node connectivity and latency.

### 3.2 Python AI/ML AllReduce Test ([`mpi_allreduce.py`](mpi_allreduce.py))
This script uses `mpi4py` to emulate the gradient-reduction patterns used by distributed deep learning engines: each rank contributes a 4 MB tensor to an `Allreduce`, and rank 0 verifies the sum and reports the duration.

---

## 4. Step 2: Build an MPI-Enabled Container

The definition file [`mpi_stack.def`](mpi_stack.def) starts from `ubuntu:22.04` and:
* Installs compilers, Python, and InfiniBand verbs development libraries.
* Builds OpenMPI 4.1.6 from source into `/opt/openmpi` with PMI support and an internal PMIx (matching typical cluster OpenMPI 4.1.x installations).
* Installs `numpy` and builds `mpi4py` against the container's `mpicc`.

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

### 6.1 The Slurm Batch Script ([`submit_mpi.slurm`](submit_mpi.slurm))

The script requests 2 nodes with 2 tasks each (4 ranks total), then:
1. Purges conflicting host modules and stops OpenMPI from launching remote tasks over ssh.
2. Sets the PMI interface (`MPI_TYPE="pmi2"`; change to `pmix` if your cluster requires it).
3. Runs the C ring benchmark (the `mpi_ring` binary compiled in Step 2) and the Python AllReduce benchmark with `srun --mpi=${MPI_TYPE} singularity exec`, binding `/dev/shm`, `/dev/infiniband`, and the work directory, and starting each rank in the work directory (`--pwd`).

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
