import torch

image = torch.randint(0, 255, (1, 3, 1024, 1024), dtype=torch.uint8)
print(image.dtype)
print(image.shape)

image = image / 255.0 - 0.5
print(image.dtype)
print(image.shape)