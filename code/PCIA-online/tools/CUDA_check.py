import torch

# Check the version of PyTorch. It should be PyTorch 1.8.0.
print(torch.__version__)
# Check if the version requirements of CUDA for PyTorch are met. It should be CUDA 11.1.
print(torch.version.cuda)
# Check if CUDA can be used.
print(torch.cuda.is_available())
# Return current the sequence number of GPUs.
print(torch.cuda.current_device())
# Return the number of GPUs you have.
print(torch.cuda.device_count())
# Return the name of your first GPU.
print(torch.cuda.get_device_name(0))
