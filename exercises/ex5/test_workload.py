import sys
import torch

print(f"Python Executable : {sys.executable}")
print(f"PyTorch Version   : {torch.__version__}")
print(f"CUDA Available    : {torch.cuda.is_available()}")

if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"Active Device     : {torch.cuda.get_device_name(0)}")
    x = torch.randn(2048, 2048, device=device)
    y = torch.matmul(x, x)
    print(f"Matrix Multiply   : SUCCESS (GPU Result Norm: {torch.norm(y):.2f})")
else:
    print("Warning: CUDA not active. Running on CPU.")
    x = torch.randn(512, 512)
    y = torch.matmul(x, x)
    print(f"Matrix Multiply   : SUCCESS (CPU Result Norm: {torch.norm(y):.2f})")
