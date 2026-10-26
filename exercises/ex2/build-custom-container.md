# Exercise 2: Build a Custom AI/ML Container with Docker

**Prerequisites:**
* A local development machine or VM with Docker installed.
* A [Docker Hub](https://hub.docker.com/) account.

---

## Overview & Architecture

In production AI/ML workflows across HPC clusters, development and runtime environments are typically separated:

1. **Development & Build (Local Workstation / Dev Machine):** Write your recipe (`Dockerfile`), assemble application dependencies, and build an OCI container image using **Docker**.
2. **Distribution (Docker Hub):** Tag and push the container image to your central Docker Hub repository.
3. **Deployment & Execution (HPC Cluster):** Use **Singularity** to pull and convert the image into a flattened, immutable `.sif` file optimized for parallel filesystems and bare-metal GPU passthrough.

```
┌────────────────────────┐      ┌───────────────────────────┐      ┌─────────────────────────┐
│ Local / Dev Node       │      │ Docker Hub Registry       │      │ HPC Cluster             │
│                        │      │ (docker.io)               │      │                         │
│ 1. Write Dockerfile    │ push │                           │ pull │ 4. Pull to .sif         │
│ 2. Build with Docker   ├─────>│ docker.io/<user>/<image>  ├─────>│ 5. Execute via          │
│ 3. Test locally        │      │                           │      │    Singularity (--nv)   │
└────────────────────────┘      └───────────────────────────┘      └─────────────────────────┘
```

---

## Step 1: Review the Verification Script and Dockerfile

This exercise directory contains the two build inputs. Run all commands in Steps 1–3 from this directory (`exercises/ex2`) on your development machine.

### 1.1 The Verification Script ([`verify_ml.py`](verify_ml.py))

Reports the Python, PyTorch, and Transformers versions, checks whether CUDA is available, and runs a small matrix multiplication on the GPU (or on the CPU as a fallback).

### 1.2 The [`Dockerfile`](Dockerfile)

Based on NVIDIA's official CUDA runtime image on Ubuntu 22.04 (`nvidia/cuda:12.4.1-runtime-ubuntu22.04`), it:
* Installs Python and core OS utilities, then purges the apt cache.
* Installs PyTorch built for CUDA 12.4, plus `transformers`, `accelerate`, `datasets`, `scikit-learn`, and `tqdm`.
* Copies `verify_ml.py` into `/app` and sets it as the default `CMD`.

---

## Step 2: Build and Test the Image Locally with Docker

Set an environment variable for your Docker Hub username:

```bash
export DOCKER_USER="<your-dockerhub-username>"
export IMAGE_TAG="${DOCKER_USER}/custom-ml-stack:1.0"
```

### 2.1 Build the Image

Build the container image and tag it for your Docker Hub namespace:

```bash
docker build -t ${IMAGE_TAG} -f Dockerfile .
```

### 2.2 Test the Container Locally

Run the container locally to verify Python and package imports:

```bash
# Run in CPU mode
docker run --rm ${IMAGE_TAG}
```

If your local machine has an NVIDIA GPU and the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) configured, verify GPU access:

```bash
# Run with GPU passthrough enabled
docker run --rm --gpus all ${IMAGE_TAG}
```

---

## Step 3: Push the Image to Docker Hub

### 3.1 Create a Docker Hub Personal Access Token (PAT)

Using a Personal Access Token instead of your primary account password is recommended:

1. Log in to [Docker Hub](https://hub.docker.com/).
2. Click your username in the top-right corner $\rightarrow$ **Account Settings**.
3. Navigate to **Security** $\rightarrow$ **Personal access tokens** $\rightarrow$ **New Access Token**.
4. Set the **Access token description** (e.g., `hpc-workshop-token`).
5. Set permissions to **Read & Write** (or Read, Write, Delete).
6. Click **Generate** and copy the resulting token string.

### 3.2 Authenticate Docker with Docker Hub

Store the access token in an environment variable and authenticate:

```bash
export DOCKER_PAT="<your-docker-access-token>"

# Log in using standard input to avoid plain-text history exposure
echo "$DOCKER_PAT" | docker login -u "${DOCKER_USER}" --password-stdin
```

*Expected output:* `Login Succeeded`

### 3.3 Push the Image

Push your tagged image to your Docker Hub repository:

```bash
docker push ${IMAGE_TAG}
```

Once uploaded, you can view your repository at `https://hub.docker.com/r/<your-dockerhub-username>/custom-ml-stack`.

---

## Step 4: Pull and Convert the Image Using Singularity

Now, switch to your **HPC cluster** terminal or compute environment where Singularity is installed.

### 4.1 Configure Cache and Scratch Paths

Prevent quota issues on `$HOME` by pointing temporary extraction and cache folders to fast cluster scratch space:

```bash
export SINGULARITY_TMPDIR=/ibex/user/$USER/tmpdir
export SINGULARITY_CACHEDIR=/ibex/user/$USER/cachedir

mkdir -p "$SINGULARITY_CACHEDIR" "$SINGULARITY_TMPDIR"
```

### 4.2 Authenticate Singularity (For Private Repositories)

If your Docker Hub repository is set to **Private**, authenticate Singularity before pulling:

```bash
export SINGULARITY_DOCKER_USERNAME="${DOCKER_USER}"
export SINGULARITY_DOCKER_PASSWORD="${DOCKER_PAT}"
```

*Alternatively, authenticate via the CLI command:*

```bash
singularity registry login --username "${DOCKER_USER}" docker://docker.io
```

*(Note: If your Docker Hub repository is **Public**, authentication is not required, though logging in helps prevent Docker Hub anonymous rate limits).*

### 4.3 Pull and Convert the Image to SIF

Download the image using the `docker://` protocol prefix:

```bash
singularity pull custom-ml-stack.sif docker://${DOCKER_USER}/custom-ml-stack:1.0
```

Singularity will:
1. Connect to Docker Hub (`docker.io`).
2. Download all layer blobs to `$SINGULARITY_CACHEDIR`.
3. Unpack and flatten layers in `$SINGULARITY_TMPDIR`.
4. Compress the final filesystem into a single, immutable SquashFS file named `custom-ml-stack.sif`.

---

## Step 5: Execute and Test with Singularity on the Cluster

### 5.1 Run the Container's Embedded Entrypoint

Execute the default `CMD` defined in the original `Dockerfile`:

```bash
singularity run custom-ml-stack.sif
```

### 5.2 Run with Cluster GPU Passthrough (`--nv`)

On an HPC compute node containing NVIDIA GPUs, pass the `--nv` flag. Singularity will dynamically locate host NVIDIA kernel modules, runtime libraries (`libcuda.so`, `libnvidia-ml.so`), and utility binaries, injecting them into the container environment:

```bash
singularity run --nv custom-ml-stack.sif
```

*Expected Output:*
```text
=======================================================
       AI/ML Container Self-Check Report
=======================================================
Python Version    : 3.10.x
PyTorch Version   : 2.4.x+cu124
Transformers Ver  : 4.x.x
CUDA Available    : True
Device Name       : NVIDIA A100-SXM4-80GB (or host model)
Device Capability : (8, 0)
GPU Matmul Test   : PASSED (Tensor Norm: 501.24)
=======================================================
```

### 5.3 Interactive Shell Inspection

Launch an interactive shell inside the container to inspect host driver bindings and GPU visibility:

```bash
singularity shell --nv custom-ml-stack.sif
```

Inside the container shell:

```bash
# Verify that the host nvidia-smi tool is mounted and working
nvidia-smi

# Check GPU accessibility from Python
python3 -c "import torch; print('CUDA Devices Visible:', torch.cuda.device_count())"

# Exit the container
exit
```

---

## Summary Comparison

| Phase | Tool | Command Example | Artifact / Destination |
| :--- | :--- | :--- | :--- |
| **1. Define** | Editor | `nano Dockerfile` | Build recipe |
| **2. Build** | Docker | `docker build -t <user>/custom-ml-stack:1.0 .` | Local Docker daemon cache |
| **3. Push** | Docker | `docker push <user>/custom-ml-stack:1.0` | Docker Hub (`docker.io`) |
| **4. Pull & Convert** | Singularity | `singularity pull custom-ml-stack.sif docker://<user>/custom-ml-stack:1.0` | Standalone SIF file on HPC scratch |
| **5. Run** | Singularity | `singularity run --nv custom-ml-stack.sif` | GPU execution on cluster compute node |
