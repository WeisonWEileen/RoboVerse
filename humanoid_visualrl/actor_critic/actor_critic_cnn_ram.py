# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import warnings

import torch
from rsl_rl.networks import Memory
from rsl_rl.utils import resolve_nn_activation
from torch import nn
from torch.distributions import Normal
from torch.nn import functional as F


class ActorCritic(nn.Module):
    is_recurrent = False

    def __init__(
        self,
        num_actor_obs,
        num_critic_obs,
        num_actions,
        action_masking,
        actor_hidden_dims=[256, 256, 256],
        critic_hidden_dims=[256, 256, 256],
        activation="elu",
        init_noise_std=1.0,
        noise_std_type: str = "scalar",
        **kwargs,
    ):
        if kwargs:
            print(
                "ActorCritic.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs.keys()])
            )
        super().__init__()
        activation = resolve_nn_activation(activation)

        mlp_input_dim_a = num_actor_obs
        mlp_input_dim_c = num_critic_obs
        # Policy
        actor_layers = []
        actor_layers.append(nn.Linear(mlp_input_dim_a, actor_hidden_dims[0]))
        actor_layers.append(activation)
        for layer_index in range(len(actor_hidden_dims)):
            if layer_index == len(actor_hidden_dims) - 1:
                actor_layers.append(nn.Linear(actor_hidden_dims[layer_index], num_actions))
            else:
                actor_layers.append(nn.Linear(actor_hidden_dims[layer_index], actor_hidden_dims[layer_index + 1]))
                actor_layers.append(activation)
        self.actor = nn.Sequential(*actor_layers)

        # Value function
        critic_layers = []
        critic_layers.append(nn.Linear(mlp_input_dim_c, critic_hidden_dims[0]))
        critic_layers.append(activation)
        for layer_index in range(len(critic_hidden_dims)):
            if layer_index == len(critic_hidden_dims) - 1:
                critic_layers.append(nn.Linear(critic_hidden_dims[layer_index], 1))
            else:
                critic_layers.append(nn.Linear(critic_hidden_dims[layer_index], critic_hidden_dims[layer_index + 1]))
                critic_layers.append(activation)
                critic_layers.append(nn.LayerNorm(critic_hidden_dims[layer_index + 1]))
        self.critic = nn.Sequential(*critic_layers)

        print(f"Actor MLP: {self.actor}")
        print(f"Critic MLP: {self.critic}")

        # Action noise
        self.noise_std_type = noise_std_type
        if self.noise_std_type == "scalar":
            self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        elif self.noise_std_type == "log":
            self.log_std = nn.Parameter(torch.log(init_noise_std * torch.ones(num_actions)))
        else:
            raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")

        # Action distribution (populated in update_distribution)
        self.distribution = None
        # disable args validation for speedup
        Normal.set_default_validate_args(False)

        self.mask = action_masking.clone()
        from loguru import logger as log

        log.info(f"Action Masking: {self.mask}")

    @staticmethod
    # not used at the moment
    def init_weights(sequential, scales):
        [
            torch.nn.init.orthogonal_(module.weight, gain=scales[idx])
            for idx, module in enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))
        ]

    def reset(self, dones=None):
        pass

    def forward(self):
        raise NotImplementedError

    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    @property
    def entropy(self):
        entropy = self.distribution.entropy()
        return (entropy * self.mask).sum(dim=-1)

    def update_distribution(self, observations):
        # compute mean
        mean = self.actor(observations)
        # compute standard deviation
        if self.noise_std_type == "scalar":
            std = self.std.expand_as(mean)
        elif self.noise_std_type == "log":
            std = torch.exp(self.log_std).expand_as(mean)
        else:
            raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")
        # way masking 1
        # mean[..., 0:8] *= 0.0
        # create distribution
        masked_mean = mean * self.mask  # self.mask shape [num_actions], 1 for active, 0 for masked
        masked_std = std * self.mask + (1.0 - self.mask) * 1e-6  # 防止 std 为 0 出 nan
        # self.distribution = Normal(mean, std)

        self.distribution = Normal(masked_mean.detach() + (mean - mean.detach()) * self.mask, masked_std)

        # way masking 2
        # Apply action masking to the mean
        # masked_mean = mean * self.mask
        # # create distribution with masked mean

    def act(self, observations, **kwargs):
        self.update_distribution(observations)
        return self.distribution.sample()

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions * self.mask).sum(dim=-1)

    def act_inference(self, observations):
        actions_mean = self.actor(observations)
        return actions_mean

    def evaluate(self, critic_observations, **kwargs):
        value = self.critic(critic_observations)
        return value

    def load_state_dict(self, state_dict, strict=True):
        """Load the parameters of the actor-critic model.

        Args:
            state_dict (dict): State dictionary of the model.
            strict (bool): Whether to strictly enforce that the keys in state_dict match the keys returned by this
                           module's state_dict() function.

        Returns:
            bool: Whether this training resumes a previous training. This flag is used by the `load()` function of
                  `OnPolicyRunner` to determine how to load further parameters (relevant for, e.g., distillation).
        """

        super().load_state_dict(state_dict, strict=strict)
        return True


#     def forward(self, x):
#         # x: (B, 3, 96, 128)
#         x = self.act1(self.conv1(x))
#         x = self.act2(self.conv2(x))
#         x = self.flatten(x)
#         x = self.act3(self.fc(x))
#         return x  # 返回 (B, output_dim)


def conv_output_size(h_w, kernel_size=1, stride=1, pad=0, dilation=1):
    """
    Utility function to compute the output size of a convolution layer.

    h_w: Tuple[int, int] - height and width of the input
    kernel_size: int or Tuple[int, int] - size of the convolution kernel
    stride: int or Tuple[int, int] - stride of the convolution
    pad: int or Tuple[int, int] - padding
    dilation: int or Tuple[int, int] - dilation rate
    """
    if isinstance(kernel_size, tuple):
        kernel_h, kernel_w = kernel_size
    else:
        kernel_h, kernel_w = kernel_size, kernel_size

    if isinstance(stride, tuple):
        stride_h, stride_w = stride
    else:
        stride_h, stride_w = stride, stride

    if isinstance(pad, tuple):
        pad_h, pad_w = pad
    else:
        pad_h, pad_w = pad, pad

    h = (h_w[0] + 2 * pad_h - dilation * (kernel_h - 1) - 1) // stride_h + 1
    w = (h_w[1] + 2 * pad_w - dilation * (kernel_w - 1) - 1) // stride_w + 1
    return h, w


class ActorCriticCNNRecurrent(ActorCritic):
    is_recurrent = True

    def __init__(
        self,
        num_actor_obs,
        num_critic_obs,
        num_actions,
        action_masking,
        actor_hidden_dims=[256, 256, 256],
        critic_hidden_dims=[256, 256, 256],
        activation="elu",
        rnn_type="lstm",
        rnn_hidden_dim=256,
        rnn_num_layers=1,
        init_noise_std=1.0,
        vision_height=96,
        vision_width=128,
        **kwargs,
    ):
        if "rnn_hidden_size" in kwargs:
            warnings.warn(
                "The argument `rnn_hidden_size` is deprecated and will be removed in a future version. "
                "Please use `rnn_hidden_dim` instead.",
                DeprecationWarning,
            )
            if rnn_hidden_dim == 256:  # Only override if the new argument is at its default
                rnn_hidden_dim = kwargs.pop("rnn_hidden_size")
        if kwargs:
            print(
                "ActorCriticRecurrent.__init__ got unexpected arguments, which will be ignored: " + str(kwargs.keys()),
            )

        super().__init__(
            num_actor_obs=rnn_hidden_dim,
            num_critic_obs=rnn_hidden_dim,
            num_actions=num_actions,
            action_masking=action_masking,
            actor_hidden_dims=actor_hidden_dims,
            critic_hidden_dims=critic_hidden_dims,
            activation=activation,
            init_noise_std=init_noise_std,
        )

        activation = resolve_nn_activation(activation)

        h, w = 96, 128
        kernel_sizes = [8, 4, 3, 3]
        h, w = conv_output_size((h, w), kernel_size=kernel_sizes[0], stride=4, pad=0)

        self.vision_encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=8, stride=4),  # (96×128) → (23×31), C=64
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2),  # (23×31) → (10×14), C=128
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, stride=1),  # (10×14) → (8×12),  C=64
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(64, 512),
            nn.ReLU(inplace=True),
        )
        # self.vision_encoder = VisionBackbonePDC(vision_height, vision_width, output_dim=64)

        # FIXME hard code here
        vision_fea_dim = self.vision_encoder(torch.zeros(1, 3, 96, 128)).shape[1]

        self.memory_a = Memory(
            num_actor_obs + vision_fea_dim, type=rnn_type, num_layers=rnn_num_layers, hidden_size=rnn_hidden_dim
        )
        self.memory_c = Memory(
            num_critic_obs + vision_fea_dim, type=rnn_type, num_layers=rnn_num_layers, hidden_size=rnn_hidden_dim
        )

        print(f"Actor RNN: {self.memory_a}")
        print(f"Critic RNN: {self.memory_c}")

    def reset(self, dones=None):
        self.memory_a.reset(dones)
        self.memory_c.reset(dones)

    def act(self, observations, masks=None, hidden_states=None, **kwargs):
        state, vision = observations

        # 检查输入维度，如果有时间维度需要特殊处理
        if state.dim() == 3:  # [time, batch, features] - 来自 recurrent_mini_batch_generator
            time_steps, batch_size = state.shape[:2]
            # 展平时间和批次维度进行vision编码
            vision_flat = vision.reshape(time_steps * batch_size, *vision.shape[2:])
            with torch.no_grad():
                vision_fea_flat = self.vision_encoder(self.preprocess_image(vision_flat))
            # 重新组织成 [time, batch, features]
            vision_fea = vision_fea_flat.reshape(time_steps, batch_size, -1)

            concat_inputs = torch.cat([state, vision_fea], dim=-1)
            inputs = self.memory_a(concat_inputs, masks, hidden_states)
            # inputs 已经是展平的，所以不需要 squeeze(0)
            self.update_distribution(inputs)
        else:  # [batch, features] - 来自推理时
            with torch.no_grad():
                vision_fea = self.vision_encoder(self.preprocess_image(vision))
            concat_inputs = torch.cat([state, vision_fea], dim=-1)
            inputs = self.memory_a(concat_inputs, masks, hidden_states)
            self.update_distribution(inputs.squeeze(0))

        actions = self.distribution.sample()
        # mask those actions dimension which is not related to the task
        # actions_masked = actions * self.masks
        return actions

    def act_inference(self, observations):
        state, vision = observations
        with torch.no_grad():
            vision_fea = self.vision_encoder(self.preprocess_image(vision))
        concat_inputs = torch.cat([state, vision_fea], dim=-1)
        inputs = self.memory_a(concat_inputs)
        # self.update_distribution(inputs.squeeze(0))
        # self.update_distribution(inputs.squeeze(0))
        mean = self.actor(inputs.squeeze(0))
        mean = mean * self.mask
        return mean

    def evaluate(self, critic_observations, masks=None, hidden_states=None):
        state, vision = critic_observations

        # 检查输入维度，如果有时间维度需要特殊处理
        if state.dim() == 3:  # [time, batch, features] - 来自 recurrent_mini_batch_generator
            time_steps, batch_size = state.shape[:2]
            # 展平时间和批次维度进行vision编码
            vision_flat = vision.reshape(time_steps * batch_size, *vision.shape[2:])
            vision_fea_flat = self.vision_encoder(self.preprocess_image(vision_flat))
            # 重新组织成 [time, batch, features]
            vision_fea = vision_fea_flat.reshape(time_steps, batch_size, -1)

            concat_inputs = torch.cat([state, vision_fea], dim=-1)
            input_c = self.memory_c(concat_inputs, masks, hidden_states)
            # input_c 已经是展平的，所以不需要 squeeze(0)
            value = self.critic(input_c)
        else:  # [batch, features] - 来自推理时
            vision_fea = self.vision_encoder(self.preprocess_image(vision))
            concat_inputs = torch.cat([state, vision_fea], dim=-1)
            input_c = self.memory_c(concat_inputs, masks, hidden_states)
            value = self.critic(input_c.squeeze(0))

        return value

    def get_hidden_states(self):
        return self.memory_a.hidden_states, self.memory_c.hidden_states

    def preprocess_image(self, image):
        """Input: image: torch.Tensor.int8, shape (B, 3, H, W). Output: image: torch.Tensor.float, shape (B, 3, H, W)."""
        # image = image.to(torch.int8)
        return image / 255.0 - 0.5
        # return image



class CNNGlimpseEncoder(nn.Module):
    def __init__(self, k=4, hw=(50, 80), original_resolution=(100, 160)):
        '''apply 4 CNN at each 25,40 patch'''
        super().__init__()
        self.k = k  # scale times
        self.hw = hw  # samllest height and width patch. for vega, original resolution is
        # (100, 160). hw =

        self.indices = []
        self.linear_layers = []
        height = self.hw[0]
        width = self.hw[1]
        center_height = original_resolution[0] // 2
        center_width = original_resolution[1] // 2

        # for i in range(self.k):
        #     # format (height_start, height_end, width_start, width_end)

        #     height_start = center_height - int(height // 2 * (i + 1) + 1)
        #     height_end = center_height + int(height // 2 * (i + 1) + 1)
        #     width_start = center_width - int(width // 2 * (i + 1))
        #     width_end = center_width + int(width // 2 * (i + 1))

        self.indices = [(25, 75, 40, 120), (13, 87, 20, 140), (0, 100, 0, 160)]
        # self.linear_layers.append(nn.Linear(3 * height * width, 512))
        # input_dim = k * 3 * height * width
        self.conv_net_1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # g -> g/2
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),  # (B,64,1,1)
            nn.Flatten(),
            nn.Linear(64, 255),
            nn.ReLU(inplace=True),
        )
        self.conv_net_2 = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # g -> g/2
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),  # (B,64,1,1)
            nn.Flatten(),
            nn.Linear(64, 164),
            nn.ReLU(inplace=True),
        )
        self.conv_net_3 = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # g -> g/2
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),  # (B,64,1,1)
            nn.Flatten(),
            nn.Linear(64, 93),
            nn.ReLU(inplace=True),
        )
        # 使用 AdaptiveAvgPool2d 将所有 patches 统一调整为 (25, 40)
        self.adaptive_pool = nn.AdaptiveAvgPool2d(self.hw)

    def foveated_image(self, image):
        phi = []
        for i in range(self.k):
            indices = self.indices[i]
            image_patch = image[:, :, indices[0] : indices[1], indices[2] : indices[3]]

            if i != 0:
                image_patch = self.adaptive_pool(image_patch)
            # 将 [B, H, W, C] 转换为 [B, C, H, W] 格式
            # image_patch = image_patch.permute(0, 3, 1, 2)
            # 使用 average pooling 将所有 patches 统一调整为 (25, 40)
            # 转换回 [B, H, W, C] 并 reshape 为 [B, 1, -1]
            # image_patch = image_patch.permute(0, 2, 3, 1)
            # image_patch = image_patch.reshape(image.shape[0], 1, -1)
            phi.append(image_patch)
        return phi

    def forward(self, image):
        phi = self.foveated_image(image)
        phi_1 = self.conv_net_1(phi[0])
        phi_2 = self.conv_net_2(phi[1])
        phi_3 = self.conv_net_3(phi[2])
        phi_out = torch.cat([phi_1, phi_2, phi_3], dim=1)
        return phi_out

class ActorCriticCNNRAM(ActorCriticCNNRecurrent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.vision_encoder = None
        self.vision_encoder = CNNGlimpseEncoder(k=3, hw=(50, 80), original_resolution=(100, 160))
