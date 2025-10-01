import torch
from torch import nn

a = torch.arange(0, 14.0, requires_grad=True).view(1, 1, 1, -1)
print(a)
m = nn.AdaptiveAvgPool2d((None, 4))
b = m(a)
print(b)
