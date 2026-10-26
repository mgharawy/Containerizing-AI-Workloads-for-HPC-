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
