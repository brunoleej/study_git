# Conv1d
# torch.nn.functional.conv1d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1) → Tensor

filters = torch.randn(33, 16, 3)
inputs = torch.randn(20, 16, 50)
F.conv1d(inputs, filters)

# conv2d
# torch.nn.functional.conv2d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1) → Tensor

# With square kernels and equal stride
filters = torch.randn(8,4,3,3)
inputs = torch.randn(1,4,5,5)
F.conv2d(inputs, filters, padding=1)

# Conv3d
# torch.nn.functional.conv3d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1) → Tensor

filters = torch.randn(33, 16, 3, 3, 3)
inputs = torch.randn(20, 16, 50, 10, 20)
F.conv3d(inputs, filters)

# conv_transpose1d

