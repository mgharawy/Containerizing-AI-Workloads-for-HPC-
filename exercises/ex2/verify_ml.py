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
