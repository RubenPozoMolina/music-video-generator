import sys
import torch
print('Python:', sys.version)
print(
    'Torch:',
    torch.__version__,
    'CUDA available:',
    torch.cuda.is_available()
)