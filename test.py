import torch
print(torch.cuda.is_available())
print(f"PyTorch Version: {torch.__version__}")
print(f"CUDA Compiled Version: {torch.version.cuda}")