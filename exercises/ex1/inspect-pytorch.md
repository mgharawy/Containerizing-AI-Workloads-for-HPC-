# Exercise 1: Pull and Inspect an NGC PyTorch Container

## Overview & Objectives
In this exercise, you will learn how Singularity handles container images built for high-performance computing (HPC) and deep learning workloads. By pulling and inspecting an NVIDIA GPU Cloud (NGC) PyTorch container, you will:

- Configure appropriate cache and scratch space for multi-gigabyte container downloads.
- Pull an OCI/Docker container image and convert it into a Singularity Image Format (`.sif`) file.
- Understand the differences between multi-layered OCI images and monolithic, flattened SIF images.
- Inspect SIF headers, internal metadata, definition records, and runtime environment definitions.
- Explore the container’s base operating system, pre-compiled CUDA toolchains, and deep learning framework stack.

---

## 1. Storage & Environment Setup

NVIDIA deep learning images are comprehensive distributions that include the CUDA runtime, cuDNN, NCCL, TensorRT, and Python deep learning frameworks. These images frequently exceed 15–20 GB when uncompressed.

Shared HPC systems often enforce strict quotas on `$HOME` (e.g., 20–50 GB). To prevent quota exhaustion and I/O bottlenecks, redirect Singularity’s cache and temporary directories to your high-speed scratch space prior to pulling.

```bash
# Define scratch-backed directories for Singularity cache and temporary build space
export SINGULARITY_TMPDIR=/ibex/user/$USER/tmpdir
export SINGULARITY_CACHEDIR=/ibex/user/$USER/cachedir

# Ensure target directories exist
mkdir -p "$SINGULARITY_CACHEDIR" "$SINGULARITY_TMPDIR"
```

---

## 2. Step 1: Pulling the NGC Container Image

Singularity can directly ingest images from OCI-compliant registries (such as NVIDIA NGC, Docker Hub, or GitHub Container Registry) using the `docker://` transport URI.

Execute the following command to download the NGC PyTorch container:

```bash
singularity pull pytorch_24.08.sif docker://nvcr.io/nvidia/pytorch:24.08-py3
```

### The Conversion Lifecycle
During this process, Singularity executes several stages automatically:
1. **Registry Manifest Query:** Queries `nvcr.io` for image layer manifests and configuration digests.
2. **Layer Retrieval:** Downloads the individual compressed layer archives (`.tar.gz`) to `$SINGULARITY_CACHEDIR/docker`.
3. **Layer Unpacking & Flattening:** Sequentially unpacks layers inside `$SINGULARITY_TMPDIR`, handling additions, permission changes, and whiteout deletions.
4. **SquashFS Creation:** Compresses the flattened filesystem into an immutable, read-only SquashFS partition.
5. **SIF Assembly:** Encapsulates the SquashFS partition, metadata, runscripts, and environment descriptors into the output binary `pytorch_24.08.sif`.

---

## 3. Step 2: Understanding Base Images & Layer Flattening

### Comparison: OCI Runtimes vs. Singularity SIF

| Feature | Docker / Podman (OCI) | Singularity (SIF) |
| :--- | :--- | :--- |
| **Storage Architecture** | Union mount of stacked layer directories via OverlayFS | Monolithic, flattened SquashFS filesystem inside a SIF envelope |
| **Filesystem Footprint** | Hundreds of thousands of small files spread across layers | Single, self-contained binary file on disk |
| **HPC Shared FS Impact** | High metadata overhead; severe performance degradation on Lustre/GPFS | Sequential I/O; negligible metadata and inode footprint |
| **Runtime Mutability** | Read-write copy-on-write (CoW) layer on top | Read-only / immutable by default |

```
Docker Registry (Stacked Layers)              Singularity Conversion
┌─────────────────────────┐
│ Layer 4: PyTorch Stack  │ ──┐
├─────────────────────────┤   │
│ Layer 3: CUDA & cuDNN   │ ──┼──> [Unpack & Flatten] ──> [SquashFS] ──> pytorch_24.08.sif
├─────────────────────────┤   │                                         (Single Immutable File)
│ Layer 2: System Packages│ ──┤
├─────────────────────────┤   │
│ Layer 1: Ubuntu Base    │ ──┘
└─────────────────────────┘
```

---

## 4. Step 3: Inspecting SIF Partitions & Image Metadata

Singularity images are structured files with dedicated descriptor headers and data partitions. You can inspect container properties without executing arbitrary code inside the container.

### 4.1 Inspect Low-Level SIF Partitions
Use `singularity sif list` to view the internal partition structure:

```bash
singularity sif list pytorch_24.08.sif
```

*Expected layout:*
```text
Container id: 6c738e4a-5f33-4b67-8fa0-1a2f6ef82914
Created on:   2024-08-28 14:22:05 +0000 UTC
Bottom Golden: false
--------------------------------------------------
ID   |GROUP|LINK |TYPE      |FLAGS    |DESCR
--------------------------------------------------
1    |1    |NONE |DEFFILE   |0        |Def.file
2    |1    |NONE |ENV       |0        |Environment
3    |1    |NONE |LABELS    |0        |JSON labels
4    |1    |NONE |FS        |0        |Squashfs filesystem (amd64)
```

### 4.2 Query Metadata and Build Attributes
Singularity stores labels and runtime instructions directly in the SIF headers:

```bash
# View general container info
singularity inspect pytorch_24.08.sif

# View labels specified during the build (JSON format)
singularity inspect --labels pytorch_24.08.sif

# View default environment variables defined in the image
singularity inspect --environment pytorch_24.08.sif

# View the entrypoint runscript
singularity inspect --runscript pytorch_24.08.sif
```

---

## 5. Step 4: Examining Internal System Libraries & Frameworks

You can inspect software installed in the container using `singularity exec`, which mounts the SquashFS partition and runs a specified command.

### 5.1 Base Operating System
Identify the Linux distribution used by the base image:

```bash
singularity exec pytorch_24.08.sif cat /etc/os-release
```
*Note: NGC images typically use Ubuntu LTS releases (e.g., Ubuntu 22.04 LTS), guaranteeing compatibility with common scientific libraries and system packages.*

### 5.2 CUDA Driver & Toolchain Inspection
Examine the embedded CUDA compiler and deep learning acceleration libraries:

```bash
# Check the embedded NVCC toolkit version
singularity exec pytorch_24.08.sif nvcc --version

# Search for installed CUDA, cuDNN, and NCCL packages
singularity exec pytorch_24.08.sif dpkg -l | grep -E "cuda|cudnn|nccl"
```

### 5.3 PyTorch Stack Verification
Query PyTorch's build parameters directly through Python:

```bash
singularity exec pytorch_24.08.sif python3 -c "
import torch
print('=== PyTorch Build Configuration ===')
print('PyTorch Version :', torch.__version__)
print('Compiled CUDA   :', torch.version.cuda)
print('cuDNN Version   :', torch.backends.cudnn.version())
print('NCCL Version    :', torch.cuda.nccl.version() if hasattr(torch.cuda, 'nccl') else 'N/A')
print('Supported SMs   :', torch.cuda.get_arch_list())
"
```

---

## 6. Hands-On Verification Tasks

Work through these questions to reinforce the core concepts:

1. **Locating Toolkit Paths:**
   Inspect `/usr/local` to identify the CUDA directories provided in the container:
   ```bash
   singularity exec pytorch_24.08.sif ls -l /usr/local | grep cuda
   ```
2. **Environment Precedence:**
   Determine which Python binary is active by default:
   ```bash
   singularity exec pytorch_24.08.sif which python3
   ```
   *Question: Does the image use `/usr/bin/python3` or an isolated environment (such as `/opt/conda` or `/usr/local/bin`)?*
3. **Filesystem Immutability Test:**
   Attempt to create a file inside the root directory:
   ```bash
   singularity exec pytorch_24.08.sif touch /test_file
   ```
   *Question: What error is returned, and why is this behavior beneficial on shared multi-user clusters?*