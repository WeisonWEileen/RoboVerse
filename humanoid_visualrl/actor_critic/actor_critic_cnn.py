# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

# import numpy as np

import torch
import torch.nn as nn
from torch.distributions import Normal
from torch.nn.modules import rnn

from rsl_rl.utils import resolve_nn_activation

from humanoid_visualrl.actor_critic.actor_critic_cnn_rnn import VisionBackbonePDC


class ActorCriticCNN(nn.Module):
    """Actor-Critic network with vanilla CNN feature extractor."""

    is_recurrent = False

    def __init__(
        self,
        num_actor_obs,
        num_critic_obs,
        num_actions,
        obs_context_len=1,
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

        # =============== CNN feature extractor ================
        self.obs_context_len = obs_context_len
        # self.vision_encoder = nn.Sequential(
        #     nn.Conv2d(3, 64, kernel_size=8, stride=4),
        #     nn.ReLU(),
        #     nn.Conv2d(64, 128, kernel_size=4, stride=2),
        #     nn.ReLU(),
        #     nn.Conv2d(128, 64, kernel_size=3, stride=1),
        #     nn.ReLU(),
        #     nn.Flatten(),
        # )

        # self.vision_encoder = nn.Sequential(
        #     nn.Conv2d(3, 64, kernel_size=8, stride=4),  # (96×128) → (23×31), C=64
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(64, 128, kernel_size=4, stride=2),  # (23×31) → (10×14), C=128
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(128, 64, kernel_size=3, stride=1),  # (10×14) → (8×12),  C=64
        #     nn.ReLU(inplace=True),
        #     # ↓↓↓ 新增 ↓↓↓
        #     nn.AdaptiveAvgPool2d((1, 1)),  # 全局平均池化 → (1×1), C=64
        #     nn.Flatten(),  # (B, 64)
        #     nn.Linear(64, 512),  # 压缩 / 投影到 512 维
        #     nn.ReLU(inplace=True),
        # )

        # FIXME hard code here
        # 原有的VisionBackbonePDC - 已注释
        # self.vision_encoder = VisionBackbonePDC(output_dim=64)

        # 新的ResNet-18预训练backbone
        resnet18 = models.resnet18(pretrained=True)
        # 移除最后的分类层，保留特征提取部分
        self.vision_encoder = nn.Sequential(*list(resnet18.children())[:-1])  # 移除最后的fc层
        # 添加一个线性层来匹配输出维度
        self.vision_projection = nn.Linear(512, 64)  # ResNet-18的fc层输出是512维

        # 计算vision特征维度
        with torch.no_grad():
            test_input = torch.zeros(1, 3, 96, 128)
            vision_fea = self.vision_encoder(test_input)
            vision_fea = vision_fea.view(vision_fea.size(0), -1)  # flatten
            vision_fea = self.vision_projection(vision_fea)
            vision_fea_dim = vision_fea.shape[1]
        # vision_fea_dim = (torch.zeros(1, 3, 96, 128)).shape[1]
        mlp_input_dim_a = num_actor_obs
        mlp_input_dim_c = num_critic_obs

        # =============== Policy ================
        actor_layers = []
        actor_layers.append(nn.Linear(mlp_input_dim_a + vision_fea_dim, actor_hidden_dims[0]))
        actor_layers.append(activation)
        for layer_index in range(len(actor_hidden_dims)):
            if layer_index == len(actor_hidden_dims) - 1:
                actor_layers.append(nn.Linear(actor_hidden_dims[layer_index], num_actions))
            else:
                actor_layers.append(nn.Linear(actor_hidden_dims[layer_index], actor_hidden_dims[layer_index + 1]))
                actor_layers.append(activation)
        self.actor = nn.Sequential(*actor_layers)

        # =============== Value function ================
        critic_layers = []
        critic_layers.append(nn.Linear(mlp_input_dim_c + vision_fea_dim, critic_hidden_dims[0]))
        critic_layers.append(activation)
        for layer_index in range(len(critic_hidden_dims)):
            if layer_index == len(critic_hidden_dims) - 1:
                critic_layers.append(nn.Linear(critic_hidden_dims[layer_index], 1))
            else:
                critic_layers.append(nn.Linear(critic_hidden_dims[layer_index], critic_hidden_dims[layer_index + 1]))
                critic_layers.append(activation)
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
        return self.distribution.entropy().sum(dim=-1)

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
        # create distribution
        self.distribution = Normal(mean, std)

    def act(self, observations, **kwargs):
        state, vision = observations
        with torch.no_grad():
            vision_fea = self.vision_encoder(vision)
        inputs = torch.cat([state, vision_fea], dim=-1)
        if self.obs_context_len != 1:
            inputs = inputs[..., -1, :]
        self.update_distribution(inputs)
        return self.distribution.sample()

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, observations):
        state, vision = observations
        with torch.no_grad():
            vision_fea = self.vision_encoder(vision)
        inputs = torch.cat([state, vision_fea], dim=-1)
        if self.obs_context_len != 1:
            inputs = inputs[..., -1, :]
        actions_mean = self.actor(inputs)
        return actions_mean

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False
        # self.eval()

    def evaluate(self, critic_observations, **kwargs):
        state, vision = critic_observations
        # with torch.no_grad():
        vision_fea = self.vision_encoder(vision)
        inputs = torch.cat([state, vision_fea], dim=-1)
        if self.obs_context_len != 1:
            inputs = inputs[..., -1, :]
        value = self.critic(inputs)
        return value


def get_activation(act_name):
    if act_name == "elu":
        return nn.ELU()
    elif act_name == "selu":
        return nn.SELU()
    elif act_name == "relu":
        return nn.ReLU()
    elif act_name == "crelu":
        return nn.ReLU()
    elif act_name == "lrelu":
        return nn.LeakyReLU()
    elif act_name == "tanh":
        return nn.Tanh()
    elif act_name == "sigmoid":
        return nn.Sigmoid()
    else:
        print("invalid activation function!")
        return None
