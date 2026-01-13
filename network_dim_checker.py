import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# Reproducibility
torch.manual_seed(0)
np.random.seed(0)

device = "cpu"


def build_actor(obs_dim=64, hidden_dims=(256, 256), num_actions=12, use_layernorm=False, activation=nn.ELU()):
    layers = []
    in_dim = obs_dim
    for h in hidden_dims:
        layers.append(nn.Linear(in_dim, h))
        if use_layernorm:
            layers.append(nn.LayerNorm(h))
        layers.append(activation)
        in_dim = h
    layers.append(nn.Linear(in_dim, num_actions))
    layers.append(nn.Tanh())
    return nn.Sequential(*layers)


# Config
obs_dim = 64
hidden_dims = (256, 256)
num_actions = 12

# Build two policies with identical initial Linear weights (fair-ish comparison)
actor_no_ln = build_actor(obs_dim, hidden_dims, num_actions, use_layernorm=False).to(device)
actor_ln = build_actor(obs_dim, hidden_dims, num_actions, use_layernorm=True).to(device)


# Copy matching Linear parameters from no-LN to LN model
def copy_linear_params(src: nn.Module, dst: nn.Module):
    src_linears = [m for m in src.modules() if isinstance(m, nn.Linear)]
    dst_linears = [m for m in dst.modules() if isinstance(m, nn.Linear)]
    assert len(src_linears) == len(dst_linears)
    for s, d in zip(src_linears, dst_linears):
        d.weight.data.copy_(s.weight.data)
        d.bias.data.copy_(s.bias.data)


copy_linear_params(actor_no_ln, actor_ln)

# Generate random observations
batch = 4096
obs = torch.randn(batch, obs_dim, device=device)

with torch.no_grad():
    act_no_ln = actor_no_ln(obs)  # (B, A)
    act_ln = actor_ln(obs)

# Per-action mean over batch
mean_per_action_no = act_no_ln.mean(dim=0).cpu().numpy()
mean_per_action_ln = act_ln.mean(dim=0).cpu().numpy()

# Overall mean across batch and action dims (a single scalar)
overall_mean_no = act_no_ln.mean().item()
overall_mean_ln = act_ln.mean().item()

# Distribution of per-sample mean action (mean across action dims)
per_sample_mean_no = act_no_ln.mean(dim=1).cpu().numpy()
per_sample_mean_ln = act_ln.mean(dim=1).cpu().numpy()

print("Overall mean (no LayerNorm):", overall_mean_no)
print("Overall mean (with LayerNorm):", overall_mean_ln)

# Plot 1: per-action means
x = np.arange(num_actions)
plt.figure()
plt.bar(x - 0.2, mean_per_action_no, width=0.4, label="No LayerNorm")
plt.bar(x + 0.2, mean_per_action_ln, width=0.4, label="With LayerNorm")
plt.axhline(0.0, linewidth=1)
plt.xlabel("Action dimension")
plt.ylabel("Mean over batch")
plt.title("Policy output mean per action dim (random obs)")
plt.legend()
plt.tight_layout()
plt.show()

# Plot 2: histogram of per-sample mean(action)
plt.figure()
plt.hist(per_sample_mean_no, bins=60, alpha=0.6, label="No LayerNorm")
plt.hist(per_sample_mean_ln, bins=60, alpha=0.6, label="With LayerNorm")
plt.axvline(0.0, linewidth=1)
plt.xlabel("Per-sample mean(action)")
plt.ylabel("Count")
plt.title("Distribution of per-sample mean(action) (random obs)")
plt.legend()
plt.tight_layout()
plt.show()

# Plot 3: mean ± std per action
std_per_action_no = act_no_ln.std(dim=0).cpu().numpy()
std_per_action_ln = act_ln.std(dim=0).cpu().numpy()

plt.figure()
plt.errorbar(x, mean_per_action_no, yerr=std_per_action_no, fmt="o", label="No LayerNorm")
plt.errorbar(x, mean_per_action_ln, yerr=std_per_action_ln, fmt="o", label="With LayerNorm")
plt.axhline(0.0, linewidth=1)
plt.xlabel("Action dimension")
plt.ylabel("Mean ± std over batch")
plt.title("Policy output mean±std per action dim (random obs)")
plt.legend()
plt.tight_layout()
plt.show()
