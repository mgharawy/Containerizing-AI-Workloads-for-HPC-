# Exercise 5: Convert to Singularity Format & Author SIF Definitions

**Estimated Duration:** 35 minutes
**Prerequisites:**
* Completed Exercises 2 and 3 (local Docker images built and tagged).
* Access to a Linux workstation or HPC cluster with Singularity (or Apptainer) installed.
* Standard user privileges (with unprivileged user namespaces or fakeroot configured for native builds).

---

## Overview & Objectives

While OCI/Docker container images are the industry standard for cloud environments, high-performance computing (HPC) clusters rely predominantly on the **Singularity Image Format (SIF)**. SIF solves fundamental multi-tenancy, security, and parallel I/O challenges inherent to shared supercomputing environments.

In this exercise, you will:
1. Examine the internal binary architecture of a `.sif` file (headers, descriptor tables, partitions).
2. Master all three Docker-to-Singularity conversion pathways (Registry, Docker Daemon, and Tar Archive).
3. Author a production-grade native Singularity definition file (`Singularity.def`) for AI/ML workloads.
4. Implement native multi-stage builds in Singularity and inspect SIF data partitions using low-level inspection utilities.

---

## 1. Deep Dive: The Singularity Image Format (SIF)

An OCI/Docker image is an archive containing a JSON manifest and multiple compressed filesystem tarballs representing diff layers. At runtime, these layers are merged using an `OverlayFS` union mount.

A **SIF file** is fundamentally different: it is an immutable, single-file binary envelope containing encapsulated metadata, cryptographic signatures, and a read-only **SquashFS** filesystem partition.

### High-Level SIF Layout

```text
+-----------------------------------------------------------------------+
| GLOBAL HEADER                                                         |
| Magic Bytes ("SIF_MAGIC"), UUID, Architecture, Timestamp, Descriptors |
+-----------------------------------------------------------------------+
| DESCRIPTOR TABLE                                                      |
| Metadata array defining each partition (Type, Offset, Length, Flags)  |
+-----------------------------------------------------------------------+
| DATA OBJECT 1: Definition File (DEFFILE)                              |
| Plaintext recipe used to build the container                          |
+-----------------------------------------------------------------------+
| DATA OBJECT 2: Environment Descriptors (ENV)                          |
| Shell configuration and runtime environment variables                 |
+-----------------------------------------------------------------------+
| DATA OBJECT 3: Labels (LABELS)                                        |
| JSON string describing author, versions, licenses                     |
+-----------------------------------------------------------------------+
| DATA OBJECT 4: System Partition (SquashFS / FS)                       |
| Complete, flattened, read-only rootfs mountable via loop device       |
+-----------------------------------------------------------------------+
| DATA OBJECT 5: Cryptographic Signature (SIGNATURE - Optional)         |
| PGP or x509 cryptographic validation signature                        |
+-----------------------------------------------------------------------+
```

### Why SIF Outperforms OCI on Shared Filesystems

1. **Inode and Metadata Preservation:** An uncompressed PyTorch container contains over 250,000 files and directories. Running 1,000 Docker containers across a cluster creates extreme metadata lock contention on parallel filesystems like Lustre or GPFS. A SIF image exists as **one physical file**, eliminating metadata strain.
2. **Deterministic Immutability:** SIF images cannot be modified after build without invalidating internal offsets and cryptographic hashes.
3. **Kernel Loop Mounts:** SIF mounts its internal SquashFS partition directly into a loop device in unprivileged user namespaces—requiring zero root daemons.

---

## 2. Docker to Singularity Conversion Pathways

Depending on whether you work on a workstation with Docker installed, a remote build cluster, or an air-gapped environment, there are three distinct conversion pathways:

```text
[Registry: Docker Hub / GHCR] ──> singularity pull ─────────────┐
                                                                 │
[Local Docker Daemon]         ──> singularity build (daemon) ───┼──> [ .SIF Container ]
                                                                 │
[Docker Tar Export]           ──> singularity build (archive) ──┘
```

### Pathway A: Direct from Remote OCI Registry (`docker://`)
Use this when your image is pushed to a remote registry (Docker Hub, Quay, GHCR, or NGC):

```bash
# Pull and convert directly from Docker Hub
singularity pull custom-ml.sif docker://username/custom-ml-stack:1.0
```

*Pros:* Simplest workflow; automated on HPC login nodes.
*Cons:* Requires high-bandwidth internet connectivity on the cluster.

---

### Pathway B: Direct from Local Docker Daemon (`docker-daemon://`)
Use this on a local development workstation where you just built an image with `docker build`, avoiding the need to push intermediate images to an external registry:

```bash
# Convert an image residing directly in the local Docker engine
singularity build custom-ml-local.sif docker-daemon://ml-image:optimized
```

*Under the Hood:* Singularity communicates with `/var/run/docker.sock`, streams the image layers directly from Docker's local image storage, unpacks them in `$SINGULARITY_TMPDIR`, and compresses them into SquashFS.

---

### Pathway C: From an Air-Gapped Docker Archive (`docker-archive://`)
HPC clusters often feature air-gapped compute networks without direct access to external container registries or Docker daemons.

#### Step 1: Export Image on Local Machine
```bash
# Save the Docker image as a flattened tar archive
docker save ml-image:optimized -o ml-image-optimized.tar
```

#### Step 2: Transfer Archive to Cluster and Convert
```bash
# Transfer archive to HPC scratch storage
scp ml-image-optimized.tar user@hpc-cluster:$SCRATCH/

# Convert the tarball into SIF on the HPC cluster
singularity build ml-image.sif docker-archive://$SCRATCH/ml-image-optimized.tar
```

---

## 3. Authoring Native Singularity Definition Files (`.def`)

While converting Docker images is standard practice, authoring native Singularity definition files (`.def`) is essential when compiling MPI-linked software against host InfiniBand stacks, setting specialized driver interfaces, or maintaining build recipes natively in HPC source repositories.

### Anatomical Structure of a Definition File

A Singularity definition file consists of a **Header** defining the base system, followed by functional **Sections** prefixed with `%`.

```text
┌────────────────────────────────────────────────────────┐
│ Header: Bootstrap & From                              │
├────────────────────────────────────────────────────────┤
│ %setup       -> Host actions before chroot build       │
│ %files       -> Copy files from host into container    │
│ %environment -> Runtime environment variables         │
│ %post        -> Install packages & compile tools (root)│
│ %runscript   -> Default execution entrypoint          │
│ %test        -> Post-build verification check          │
│ %labels      -> Metadata key-value pairs               │
│ %help        -> Help documentation                     │
└────────────────────────────────────────────────────────┘
```

### A Production AI/ML Definition File

Run the commands in Sections 3–6 from this directory (`exercises/ex5`), which contains the definition files and test script.

Open [`pytorch_lab.def`](pytorch_lab.def) and walk through each section:
* **Header:** Bootstraps from `nvidia/cuda:12.4.1-runtime-ubuntu22.04` on Docker Hub.
* **`%labels`:** Author, version, and description metadata.
* **`%environment`:** Locale, `PATH` to the `/opt/venv` virtualenv, unbuffered Python output, and `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.
* **`%files`:** Copies `test_workload.py` from the host into `/app`.
* **`%post`:** Installs Python, creates `/opt/venv`, installs CUDA-enabled PyTorch plus `transformers`, `datasets`, `accelerate`, and `scikit-learn`, then strips test suites and bytecode to shrink the SquashFS image.
* **`%runscript`:** Prints a banner and runs `/app/test_workload.py`, forwarding any arguments.
* **`%test`:** Fails the build if PyTorch cannot be imported.
* **`%help`:** Usage text shown by `singularity run-help`.

### The Accompanying Script ([`test_workload.py`](test_workload.py))

The test script referenced in the `%files` section. It prints the Python executable, PyTorch version, and CUDA availability, then runs a matrix multiplication on the GPU (or on the CPU as a fallback).

---

## 4. Multi-Stage Builds in Native Singularity Definitions

Just like Docker, modern Singularity supports multi-stage builds directly within definition files using the `Stage:` keyword. This completely separates build tools from the final runtime image.

Inspect the example multi-stage definition in [`pytorch_multistage.def`](pytorch_multistage.def):
* **Stage `build`:** Starts from the CUDA `devel` image, installs compilers and Python headers, and installs PyTorch into `/opt/venv`.
* **Stage `final`:** Starts from the slim CUDA `runtime` image and copies **only** `/opt/venv` from the build stage via `%files from build`.

---

## 5. Building the SIF Container

### Understanding Privileges: Root vs. Fakeroot

Because the `%post` section typically installs system packages via `apt-get`, root access is required during the build process:

1. **Option 1: Using `--fakeroot` (Recommended for clusters without native root):**
   Uses Linux user namespaces and UID/GID mapping to simulate root privileges:
   ```bash
   singularity build --fakeroot pytorch_lab.sif pytorch_lab.def
   ```

2. **Option 2: Using `sudo` (Local development workstations):**
   ```bash
   sudo singularity build pytorch_lab.sif pytorch_lab.def
   ```

3. **Option 3: Remote Build Service (If user namespaces are disabled):**
   ```bash
   singularity build --remote pytorch_lab.sif pytorch_lab.def
   ```

---

## 6. SIF Structural Inspection & Header Extraction

Now that you have built a native SIF container, inspect its low-level binary layout using the `sif` utility suite built into Singularity.

### 6.1 List Internal Data Partitions
```bash
singularity sif list pytorch_lab.sif
```

*Sample Output:*
```text
Container id: a49b29cb-875f-4131-b847-b8dbdfc4da73
Created on:   2026-10-18 09:30:15 +0000 UTC
Bottom Golden: false
--------------------------------------------------
ID   |GROUP|LINK |TYPE      |FLAGS    |DESCR
--------------------------------------------------
1    |1    |NONE |DEFFILE   |0        |Def.file
2    |1    |NONE |ENV       |0        |Environment
3    |1    |NONE |LABELS    |0        |JSON labels
4    |1    |NONE |FS        |0        |Squashfs filesystem (amd64)
5    |1    |NONE |RUNSCRIPT |0        |Runscript
6    |1    |NONE |TEST      |0        |Test script
```

### 6.2 Dump the Original Definition Recipe Directly from SIF
Because SIF preserves the build recipe in Partition ID `1` (`DEFFILE`), you can always retrieve the exact source instructions from any compiled SIF container:

```bash
# Dump the definition file stored inside the binary
singularity sif dump 1 pytorch_lab.sif
```

*Or via the high-level inspect interface:*
```bash
singularity inspect --deffile pytorch_lab.sif
```

### 6.3 Test Runtime Execution with GPU Bindings
Execute your freshly constructed native container using the host GPU:

```bash
singularity run --nv pytorch_lab.sif
```

---

## Summary: Dockerfile vs. Singularity Definition

| Concept | Dockerfile | Singularity Definition (`.def`) |
| :--- | :--- | :--- |
| **Base Image** | `FROM <image>` | `Bootstrap: docker` / `From: <image>` |
| **Package Installation** | Multiple `RUN` directives | Single `%post` block (runs sequentially in chroot) |
| **File Injection** | `COPY src dest` | `%files` section (`src dest`) |
| **Environment Settings**| `ENV KEY=VALUE` | `%environment` section (`export KEY=VALUE`) |
| **Default Entrypoint**  | `CMD [...]` or `ENTRYPOINT [...]` | `%runscript` block |
| **Build-time Tests**    | Run manually or as extra stages | Dedicated `%test` section (fails build on non-zero exit) |
| **Multi-stage**         | `FROM ... AS name` / `COPY --from` | `Stage: <name>` / `%files from <name>` |
| **Output Artifact**     | Layer tarballs stored in `/var/lib/docker` | Standalone single-file binary (`.sif`) |

---

## Hands-On Verification Challenges

1. **Extracting Embedded Documentation:**
   Run the `%help` text embedded inside your SIF file:
   ```bash
   singularity run-help pytorch_lab.sif
   ```
2. **SIF Modification Attempt:**
   Try appending content to an immutable SIF image using standard Linux file redirection:
   ```bash
   singularity exec pytorch_lab.sif touch /opt/newfile
   ```
   *Note what happens when attempting write operations on the SquashFS mount.*
3. **Partition Verification:**
   Determine which compression algorithm was used for the SquashFS partition:
   ```bash
   singularity sif info 4 pytorch_lab.sif
   ```