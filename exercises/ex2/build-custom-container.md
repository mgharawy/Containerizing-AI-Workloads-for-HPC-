# Exercise 2: Build a Custom AI/ML Container with Podman

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

## Step 1: Write the Verification Script and Dockerfile

Create a dedicated workspace directory on your development machine:

```bash
mkdir -p ~/ml-docker-lab && cd ~/ml-docker-lab
```

### 1.1 Create the Verification Script (`verify_ml.py`)

Create a script to verify that PyTorch, CUDA bindings, and Hugging Face libraries are functional:

```bash
cat << 'EOF' > verify_ml.py
import sys
import torch
import transformers

print("=" * 55)
print("       AI/ML Container Self-Check Report")
print("=" * 55)
print(f"Python Version    : {sys.version.split()[0]}")
print(f"PyTorch Version   : {torch.__version__}")
print(f"Transformers Ver  : {transformers.__version__}")
print(f"CUDA Available    : {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"Device Name       : {torch.cuda.get_device_name(0)}")
    print(f"Device Capability : {torch.cuda.get_device_capability(0)}")
    # Run a simple tensor matrix multiplication on GPU
    a = torch.rand(1000, 1000, device="cuda")
    b = torch.matmul(a, a)
    print(f"GPU Matmul Test   : PASSED (Tensor Norm: {torch.norm(b):.2f})")
else:
    print("Device Mode       : CPU Mode (CUDA not initialized)")
    a = torch.rand(500, 500)
    b = torch.matmul(a, a)
    print(f"CPU Matmul Test   : PASSED (Tensor Norm: {torch.norm(b):.2f})")
print("=" * 55)
EOF
```

### 1.2 Write the `Dockerfile`

Create a `Dockerfile` based on NVIDIA's official CUDA runtime image on Ubuntu 22.04:

```dockerfile
cat << 'EOF' > Dockerfile
# Stage 1: Base CUDA runtime image on Ubuntu 22.04
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

# Prevent interactive prompts during apt installation
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8

# Install Python and core OS utilities
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-dev \
    git \
    curl \
    ca-certificates \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip and install wheel utilities
RUN python3 -m pip install --no-cache-dir --upgrade pip setuptools wheel

# Install PyTorch built for CUDA 12.4
RUN python3 -m pip install --no-cache-dir \
    torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# Install common ML dependencies
RUN python3 -m pip install --no-cache-dir \
    transformers \
    accelerate \
    datasets \
    scikit-learn \
    tqdm

# Set workspace and embed the verification script
WORKDIR /app
COPY verify_ml.py /app/verify_ml.py

# Default execution entrypoint
CMD ["python3", "/app/verify_ml.py"]
EOF
```

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
export SINGULARITY_CACHEDIR=$SCRATCH/.singularity/cache
export SINGULARITY_TMPDIR=$SCRATCH/.singularity/tmp

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