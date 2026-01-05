import torch
import torch.nn as nn

vision_encoder = nn.Sequential(
    nn.Conv2d(3, 64, kernel_size=8, stride=4),
    nn.ReLU(inplace=True),
    nn.Conv2d(64, 128, kernel_size=4, stride=2),
    nn.ReLU(inplace=True),
    nn.Conv2d(128, 64, kernel_size=3, stride=1),
    nn.ReLU(inplace=True),
    nn.AdaptiveAvgPool2d((1, 1)),
    nn.Flatten(),
    nn.Linear(64, 512),
    nn.ReLU(inplace=True),
)


def hook_fn(name):
    def _hook(module, inp, out):
        # inp is a tuple; out may be Tensor or tuple/list of Tensors
        in_shape = tuple(inp[0].shape) if isinstance(inp, (tuple, list)) and torch.is_tensor(inp[0]) else "?"
        if torch.is_tensor(out):
            out_shape = tuple(out.shape)
        elif isinstance(out, (tuple, list)) and len(out) > 0 and torch.is_tensor(out[0]):
            out_shape = [tuple(t.shape) for t in out]
        else:
            out_shape = type(out)
        print(f"{name:<25} | {module.__class__.__name__:<18} | in: {in_shape} -> out: {out_shape}")

    return _hook


hooks = []
for i, m in enumerate(vision_encoder):
    hooks.append(m.register_forward_hook(hook_fn(f"layer[{i}]")))

# dummy input: batch=2
x = torch.randn(2, 3, 100, 160)
with torch.no_grad():
    y = vision_encoder(x)

print("\nFinal output:", tuple(y.shape))

# cleanup
for h in hooks:
    h.remove()
