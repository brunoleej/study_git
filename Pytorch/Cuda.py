import torch

# Returns a bool indicating if CUDA is currently available.
print(torch.cuda.is_available())    # True

# Returns the index of a currently selected device
print(torch.cuda.current_device())  # 0

# Returns the number of GPUs available.
print(torch.cuda.device_count())    # 1

# Gets the name of a device
print(torch.cuda.get_device_name(0))    # GeForce RTX 3090

# Context-manager that changes the selected deivice
# device (torch.device or int) - device index to select.
print(torch.cuda.device(0)) # <torch.cuda.device object at 0x00000189C46B8070>

# Default CUDA device
cuda = torch.device('cuda')

# allocates a tensor on default GPU
a = torch.tensor([1., 2.], device = cuda)
print(a)

# transfers a tensor from 'C'PU to 'G'PU
b = torch.tensor([1.,2.]).cuda()

# Same with .cuda()
b2 = torch.tensor([1., 2.]).to(device = cuda)


