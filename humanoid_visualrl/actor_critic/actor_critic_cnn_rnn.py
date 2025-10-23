# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import warnings

import torch
from rsl_rl.modules import ActorCritic

# from humanoid_visualrl.actor_critic.actor_critic_cnn import ActorCriticCNN
from rsl_rl.networks import Memory
from rsl_rl.utils import resolve_nn_activation
from torch import nn


# class VisionBackbonePDC(nn.Module):
#     def __init__(self, image_height, image_width, output_dim=128):
#         super().__init__()
#         # 假定两个 conv 层，每层 32 通道
#         # 第一个卷积：kernel 8, stride 4 → 从 96×128 到 ( (96-8)//4 +1 , (128-8)//4 +1 ) = (23, 31)
#         self.conv1 = nn.Conv2d(3, 32, kernel_size=8, stride=4, padding=0)
#         self.act1 = nn.ReLU(inplace=True)
#         # 第二个卷积：kernel 4, stride 2 → (23,31) → (10,14)
#         self.conv2 = nn.Conv2d(32, 32, kernel_size=4, stride=2, padding=0)
#         self.act2 = nn.ReLU(inplace=True)
#         # Flatten + FC 映射到 desired 输出维度
#         # 输出的 conv 特征图尺寸要算清楚
#         # conv2 输出的 spatial dims: ((23-4)//2 +1 = 10, (31-4)//2 +1 = 14)
#         # 所以 特征图是 32 × 10 × 14 = 4480 维
#         self.flatten = nn.Flatten()
#         self.fc = nn.Linear(32 * 10 * 14, output_dim)
#         self.act3 = nn.ReLU(inplace=True)

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
        masking_all=False,
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
            masking_all=masking_all,
        )

        activation = resolve_nn_activation(activation)

        h, w = 96, 128
        filter_sizes = [16, 32, 64, 128]
        kernel_sizes = [8, 4, 3, 3]
        h, w = conv_output_size((h, w), kernel_size=kernel_sizes[0], stride=4, pad=0)
        layer1_norm_shape = [filter_sizes[0], h, w]
        h, w = conv_output_size((h, w), kernel_size=kernel_sizes[1], stride=2, pad=0)
        layer2_norm_shape = [filter_sizes[1], h, w]
        h, w = conv_output_size((h, w), kernel_size=kernel_sizes[2], stride=1, pad=0)
        layer3_norm_shape = [filter_sizes[2], h, w]
        h, w = conv_output_size((h, w), kernel_size=kernel_sizes[3], stride=1, pad=0)
        layer4_norm_shape = [filter_sizes[3], h, w]

        #  hisotry version 128 × 96
        # self.vision_encoder = nn.Sequential(
        #     nn.Conv2d(3, filter_sizes[0], kernel_size=kernel_sizes[0], stride=4, padding=0),
        #     nn.ReLU(inplace=True),
        #     nn.LayerNorm(layer1_norm_shape),
        #     nn.Conv2d(filter_sizes[0], filter_sizes[1], kernel_size=kernel_sizes[1], stride=2, padding=0),
        #     nn.ReLU(inplace=True),
        #     nn.LayerNorm(layer2_norm_shape),
        #     nn.Conv2d(filter_sizes[1], filter_sizes[2], kernel_size=kernel_sizes[2], stride=1, padding=0),
        #     nn.LayerNorm(layer3_norm_shape),
        #     nn.Conv2d(filter_sizes[2], filter_sizes[3], kernel_size=kernel_sizes[3], stride=1, padding=0),
        #     nn.LayerNorm(layer4_norm_shape),
        #     nn.ReLU(inplace=True),
        #     nn.AdaptiveAvgPool2d((1, 1)),  # 全局平均池化 → (1×1), C=filter_sizes[3]
        #     nn.Flatten(),  # (B, filter_sizes[3])
        #     nn.Linear(filter_sizes[3], 32),  # 压缩 / 投影到 32 维
        #     nn.ReLU(inplace=True),
        # )
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
                vision_fea_flat = self.vision_encoder(vision_flat)
            # 重新组织成 [time, batch, features]
            vision_fea = vision_fea_flat.reshape(time_steps, batch_size, -1)

            concat_inputs = torch.cat([state, vision_fea], dim=-1)
            inputs = self.memory_a(concat_inputs, masks, hidden_states)
            # inputs 已经是展平的，所以不需要 squeeze(0)
            self.update_distribution(inputs)
        else:  # [batch, features] - 来自推理时
            with torch.no_grad():
                vision_fea = self.vision_encoder(vision)
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
            vision_fea = self.vision_encoder(vision)
        concat_inputs = torch.cat([state, vision_fea], dim=-1)
        inputs = self.memory_a(concat_inputs)
        self.update_distribution(inputs.squeeze(0))
        return self.distribution.sample()

    def evaluate(self, critic_observations, masks=None, hidden_states=None):
        state, vision = critic_observations

        # 检查输入维度，如果有时间维度需要特殊处理
        if state.dim() == 3:  # [time, batch, features] - 来自 recurrent_mini_batch_generator
            time_steps, batch_size = state.shape[:2]
            # 展平时间和批次维度进行vision编码
            vision_flat = vision.reshape(time_steps * batch_size, *vision.shape[2:])
            vision_fea_flat = self.vision_encoder(vision_flat)
            # 重新组织成 [time, batch, features]
            vision_fea = vision_fea_flat.reshape(time_steps, batch_size, -1)

            concat_inputs = torch.cat([state, vision_fea], dim=-1)
            input_c = self.memory_c(concat_inputs, masks, hidden_states)
            # input_c 已经是展平的，所以不需要 squeeze(0)
            value = self.critic(input_c)
        else:  # [batch, features] - 来自推理时
            vision_fea = self.vision_encoder(vision)
            concat_inputs = torch.cat([state, vision_fea], dim=-1)
            input_c = self.memory_c(concat_inputs, masks, hidden_states)
            value = self.critic(input_c.squeeze(0))

        return value

    def get_hidden_states(self):
        return self.memory_a.hidden_states, self.memory_c.hidden_states
