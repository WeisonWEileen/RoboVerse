# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import warnings

import torch
# from rsl_rl.modules import ActorCritic

# from humanoid_visualrl.actor_critic.actor_critic_cnn import ActorCriticCNN
from rsl_rl.networks import Memory
from rsl_rl.utils import resolve_nn_activation
from torch import nn
import torch.nn.functional as F

from torch.distributions import Normal

# we directly see the location as action

from rsl_rl.utils import resolve_nn_activation


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
        masking_all: bool = False,
        **kwargs,
    ):
        if kwargs:
            print(
                "ActorCritic.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs.keys()])
            )
        super().__init__()
        activation = resolve_nn_activation(activation)
        # foveated_action_dim = 2

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

        # Apply action masking to the mean
        masked_mean = mean * self.mask
        # create distribution with masked mean
        self.distribution = Normal(masked_mean, std)

    def act(self, observations, **kwargs):
        self.update_distribution(observations)
        return self.distribution.sample()

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

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


class ActorCriticCNNRecurrentFoveated(ActorCritic):
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

        # modified to be more foveated
        self.vision_encoder = nn.Sequential(
            # ───── 1 ───── 160×120 → 26×20  （感受野 12）
            nn.Conv2d(3, 64, kernel_size=12, stride=6, padding=3),
            nn.ReLU(inplace=True),

            # ───── 2 ───── 26×20 → 8×6     （感受野 12 + (6·6)=48 → 60）
            nn.Conv2d(64, 128, kernel_size=6, stride=3, padding=1),
            nn.ReLU(inplace=True),

            # ───── 3 ───── 8×6 → 4×3        （感受野 60 + (3·4)=12 → 72）
            nn.Conv2d(128, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),

            # ───── Global Pool ───── 4×3 → 1×1
            nn.AdaptiveAvgPool2d(1),   # 128 × 1 × 1

            nn.Flatten(),              # 128
            nn.Linear(128, 32),
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
