import os
import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim

def print_separator(title=""):
    print(f"\n{'=' * 20} {title} {'=' * 20}")

print_separator("Environment & Hardware Diagnostics")
print(f"Host Hostname           : {os.uname().nodename}")
print(f"Slurm Job ID            : {os.getenv('SLURM_JOB_ID', 'N/A')}")
print(f"Python Binary           : {sys.executable}")
print(f"PyTorch Version         : {torch.__version__}")
print(f"CUDA Available          : {torch.cuda.is_available()}")
print(f"CUDA_VISIBLE_DEVICES    : {os.getenv('CUDA_VISIBLE_DEVICES', 'Unset')}")

if not torch.cuda.is_available():
    print("\n[ERROR] CUDA is not available to PyTorch. Exiting.")
    sys.exit(1)

device_id = 0
device_name = torch.cuda.get_device_name(device_id)
device_capability = torch.cuda.get_device_capability(device_id)
total_mem_gb = torch.cuda.get_device_properties(device_id).total_memory / (1024 ** 3)

print(f"Active GPU Device       : [{device_id}] {device_name}")
print(f"Compute Capability      : {device_capability[0]}.{device_capability[1]}")
print(f"Total Dedicated Memory  : {total_mem_gb:.2f} GB")

print_separator("Running Synthetic Neural Network Benchmark")
# Define a multi-layer perceptron
model = nn.Sequential(
    nn.Linear(8192, 4096),
    nn.ReLU(),
    nn.Linear(4096, 2048),
    nn.ReLU(),
    nn.Linear(2048, 10)
).to("cuda")

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# Synthetic batch parameters
batch_size = 256
num_batches = 100
print(f"Batch Size: {batch_size} | Iterations: {num_batches}")

start_time = time.time()
for step in range(1, num_batches + 1):
    inputs = torch.randn(batch_size, 8192, device="cuda")
    labels = torch.randint(0, 10, (batch_size,), device="cuda")

    optimizer.zero_grad()
    outputs = model(inputs)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()

    if step % 25 == 0 or step == num_batches:
        allocated = torch.cuda.memory_allocated() / (1024 ** 2)
        reserved = torch.cuda.memory_reserved() / (1024 ** 2)
        print(f"Step [{step:03d}/{num_batches:03d}] - Loss: {loss.item():.4f} | "
              f"VRAM Allocated: {allocated:.1f} MB | VRAM Reserved: {reserved:.1f} MB")

elapsed = time.time() - start_time
throughput = (num_batches * batch_size) / elapsed

print_separator("Benchmark Summary")
print(f"Total Elapsed Time      : {elapsed:.2f} s")
print(f"Training Throughput     : {throughput:.2f} samples/s")
print(f"Peak VRAM Usage         : {torch.cuda.max_memory_allocated() / (1024 ** 2):.1f} MB")
print("Job execution completed successfully.\n")
