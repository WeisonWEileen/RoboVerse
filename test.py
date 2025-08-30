import torch

a = torch.exp(torch.tensor([-10 / 50.0])) - torch.exp(torch.tensor([-128 / 2.0 / 50.0]))
print(a)
