# Exercise 3: Image Optimization for AI/ML Containers

**Estimated Duration:** 35 minutes
**Prerequisites:**
* Docker installed on a local development workstation or build node.
* Basic familiarity with `Dockerfile` syntax and container builds (Exercise 2).
* Access to Singularity on an HPC cluster or development system.

---

## Overview & Motivation

AI/ML container images routinely exceed 15–20 GB. In high-performance computing (HPC) and distributed training environments, oversized images introduce significant bottlenecks:

1. **Network Saturation:** Pushing and pulling multi-gigabyte layers across cluster nodes degrades interconnect bandwidth and strains registry rate limits.
2. **Filesystem Strain:** Unpacking multi-layered images containing millions of small files overwhelms parallel filesystems (e.g., Lustre, GPFS) by depleting inodes and creating metadata server contention.
3. **Storage Quota Depletion:** Singularity cache directories (`SINGULARITY_CACHEDIR`) and target directories quickly exceed allocated disk quotas.
4. **Security & Attack Surface:** Retaining development utilities (`gcc`, `g++`, header files, intermediate build dependencies) in production containers increases security exposure.

In this exercise, you will build both a "naive" AI/ML container and an "optimized" multi-stage container, analyze layer construction, and compare resulting image footprints in Docker and Singularity (`.sif`).

---

## 1. Core Principles of Container Optimization

### The Layer Mechanism and Whiteout Files
In Docker/OCI images, each `RUN`, `COPY`, and `ADD` directive commits a new read-only filesystem layer.

```
Layer 3: RUN rm -rf /tmp/build-cache   <-- Marks files as hidden (whiteout)
Layer 2: RUN pip install ...           <-- Contains /tmp/build-cache (1.8 GB)
Layer 1: FROM nvidia/cuda:...          <-- Base image
```

* **The Whiteout Trap:** Deleting a file in a subsequent `RUN` statement does **not** reduce image size. It merely records a "whiteout" record in the upper layer, while the data remains permanently stored in the underlying layer.
* **Consolidation Rule:** File creation, compilation, package installation, and cleanup must take place within the **same** `RUN` step to prevent intermediate data from persisting in the layer history.

### Multi-Stage Builds
Multi-stage builds decouple the **build environment** from the **runtime environment**:

* **Stage 1 (Builder):** Uses a heavy `devel` base image equipped with compilers (`gcc`, `nvcc`), header files (`python3-dev`, CUDA headers), and build utilities to compile Python wheels or binaries into a dedicated path (e.g., a virtual environment `/opt/venv`).
* **Stage 2 (Runtime):** Uses a minimal `runtime` or `base` image. It copies only the finished virtual environment `/opt/venv` and application source code, leaving behind all build toolchains and package caches.

---

## 2. Step 1: Build the Unoptimized ("Naive") Image

Run all Docker commands in this exercise from this directory (`exercises/ex3`), which contains the build files.

### 2.1 The Naive Dockerfile ([`Dockerfile.naive`](Dockerfile.naive))

Inspect this intentionally unoptimized Dockerfile, which exhibits common anti-patterns:
* Uses a heavyweight CUDA `devel` image rather than a `runtime` image.
* Executes commands in separate `RUN` layers.
* Retains `apt` package indexes in `/var/lib/apt/lists/`.
* Retains `pip` cache files in `~/.cache/pip`.
* Installs development compilers permanently into the production container.

### 2.2 Build the Naive Image

```bash
docker build -t ml-image:naive -f Dockerfile.naive .
```

---

## 3. Step 2: Build the Optimized Multi-Stage Image

Now design an optimized build that achieves the same runtime capability with minimal footprint.

### 3.1 The `.dockerignore` File

The provided [`.dockerignore`](.dockerignore) keeps git metadata, Python bytecode caches, notebook checkpoints, and previously built `.sif`/`.tar.gz` artifacts out of the Docker build context.

### 3.2 The Optimized Multi-Stage Dockerfile ([`Dockerfile.optimized`](Dockerfile.optimized))

This configuration uses:
1. **Multi-Stage Architecture:** Builder stage compiles and installs into `/opt/venv`; runtime stage imports only the isolated `/opt/venv`.
2. **Base Slicing:** The runtime stage uses `nvidia/cuda:12.4.1-runtime-ubuntu22.04` instead of `devel`.
3. **Layer Chaining:** Groups `apt-get update`, installation, and cleanup into a single `RUN` command.
4. **No Pip Cache:** Enforces `--no-cache-dir`.

### 3.3 Build the Optimized Image

```bash
docker build -t ml-image:optimized -f Dockerfile.optimized .
```

---

## 4. Step 3: Layer Analysis & Cache Strategies

### 4.1 Inspecting Layer Breakdown with `docker history`

Examine where storage is consumed across both images:

```bash
# Inspect layer breakdown for the naive image
docker history ml-image:naive

# Inspect layer breakdown for the optimized image
docker history ml-image:optimized
```

*Observations to make:*
* In `ml-image:naive`, each `RUN pip3 install` layer contributes gigabytes of uncompressed wheel cache overhead.
* In `ml-image:optimized`, the builder stage layers are completely discarded; only the final `COPY --from=builder /opt/venv /opt/venv` layer enters the runtime image.

### 4.2 Cache Optimization Strategies

Docker caches layers by evaluating if the instruction and the referenced files have changed since the previous build:

1. **Order of Volatility (Low to High):**
   Place instructions that change rarely at the top of the file, and instructions that change frequently toward the bottom:
   ```
   [Infrequent Changes]  1. OS package installation (`apt-get`)
          │              2. Core framework installation (`torch`, `cuda`)
          │              3. Auxiliary Python dependencies (`requirements.txt`)
          ▼              4. Application code (`COPY . /app`)
   [Frequent Changes]    5. Entrypoint instructions (`CMD` / `ENTRYPOINT`)
   ```

2. **Split Dependency Installation from Application Source:**
   Do not write:
   ```dockerfile
   # BAD: Re-triggers pip install on EVERY code change
   COPY . /app
   RUN pip install -r /app/requirements.txt
   ```
   Instead, isolate dependency specifications:
   ```dockerfile
   # GOOD: Caches dependencies until requirements.txt itself changes
   COPY requirements.txt /tmp/requirements.txt
   RUN pip install --no-cache-dir -r /tmp/requirements.txt
   COPY . /app
   ```

---

## 5. Step 4: Compare Image Sizes (Docker vs. Singularity SIF)

Singularity compresses the root filesystem into a read-only SquashFS format. Compare how optimization impacts both the uncompressed OCI layers in Docker and the compressed `.sif` on disk.

### 5.1 Compare Docker Images

Query Docker to display the uncompressed image sizes:

```bash
docker images | grep -E "ml-image|REPOSITORY"
```

*Representative Output:*
```text
REPOSITORY   TAG         IMAGE ID       CREATED         SIZE
ml-image     naive       9c8a77d12f4e   10 minutes ago  12.8 GB
ml-image     optimized   3a1b5c28d09f   2 minutes ago   5.6 GB
```

### 5.2 Convert Both Images to Singularity SIF

Convert both images locally using the Docker daemon transport (`docker-daemon://`):

```bash
# Convert naive image
singularity build ml-naive.sif docker-daemon://ml-image:naive

# Convert optimized image
singularity build ml-optimized.sif docker-daemon://ml-image:optimized
```

### 5.3 Compare SIF Files on Disk

Inspect the final file sizes of both Singularity containers:

```bash
ls -lh ml-naive.sif ml-optimized.sif
```

*Representative Output:*
```text
-rwxr-xr-x 1 user user 6.1G Oct 16 14:30 ml-naive.sif
-rwxr-xr-x 1 user user 2.7G Oct 16 14:35 ml-optimized.sif
```

---

## 6. Optimization Summary & Comparative Analysis

| Metric | Naive Image (`ml-image:naive`) | Optimized Image (`ml-image:optimized`) | Difference |
| :--- | :--- | :--- | :--- |
| **Base Image** | `cuda:12.4.1-devel-ubuntu22.04` | `cuda:12.4.1-runtime-ubuntu22.04` | Sliced base runtime |
| **Build Architecture** | Single-stage | Multi-stage (`builder` $\rightarrow$ `runtime`) | Isolated toolchains |
| **Compilers in Image** | Present (`gcc`, `g++`, `nvcc`) | Excluded | Reduced attack surface |
| **Docker Image Size** | ~12.8 GB | ~5.6 GB | **~56% reduction** |
| **Singularity SIF Size**| ~6.1 GB | ~2.7 GB | **~55% reduction** |
| **Network Transfer Time (1Gbps)** | ~50 seconds | ~22 seconds | Over 2x faster pull |
| **HPC Inode Strain** | High (temporary build caches) | Minimal (clean virtual environment) | Lower metadata contention |

---

## 7. Hands-on Verification Questions

1. **Compiler Verification:**
   Verify whether a C compiler exists inside the optimized Singularity container:
   ```bash
   singularity exec ml-optimized.sif which gcc
   ```
   *Expected result:* Exit status `1` (binary not found).

2. **Python Environment Verification:**
   Confirm that PyTorch and CUDA runtime functionality are intact inside the optimized container:
   ```bash
   singularity exec --nv ml-optimized.sif python3 -c "import torch; print('CUDA Functional:', torch.cuda.is_available())"
   ```

3. **Layer Audit Task:**
   Run `singularity sif list ml-optimized.sif` vs `singularity sif list ml-naive.sif`. Does the partition table structure change, or only the SquashFS payload size?
