import torch

print('=== PyTorch Build Configuration ===')
print('PyTorch Version :', torch.__version__)
print('Compiled CUDA   :', torch.version.cuda)
print('cuDNN Version   :', torch.backends.cudnn.version())
print('NCCL Version    :', torch.cuda.nccl.version() if hasattr(torch.cuda, 'nccl') else 'N/A')
print('Supported SMs   :', torch.cuda.get_arch_list())
