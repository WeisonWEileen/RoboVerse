"""A humanoid base wrapper for skillBench tasks."""

from __future__ import annotations

from collections import deque
from copy import deepcopy

import numpy as np
import torch

from humanoid_visualrl.cfg.humanoidFixedGazingCfg import BaseTableHumanoidTaskCfg
from humanoid_visualrl.utils.utils import (
    get_body_reindexed_indices_from_substring,
    sample_int_from_float,
    sample_wp,
    torch_rand_float,
)
from metasim.scenario.scenario import ScenarioCfg
from metasim.types import TensorState
from roboverse_learn.rl.rsl_rl.rsl_rl_wrapper import RslRlWrapper
from humanoid_visualrl.wrapper.base_humanoid_wrapper import HumanoidBaseWrapper


class ActiveVisionWrapper(HumanoidBaseWrapper):
    """Wraps Metasim environments to be compatible with rsl_rl OnPolicyRunner.

    Note that rsl_rl is designed for parallel training fully on GPU, with robust support for Isaac Gym and Isaac Lab.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        tensor_state = self.env.get_states()
        self._init_target_wp(tensor_state)
        # self.marker_viz = self.env.init_marker_viz()

    def _init_buffers(self):
        super()._init_buffers()
        # self.wrist_pose = torch.zeros(self.num_envs, 2, 7, device=self.device)

    def _refreshed_tensors(self, tensor_state: TensorState):
        super()._refreshed_tensors(tensor_state)
        # Convert from HWC (H, W, C) to CHW (C, H, W) format for PyTorch CNN
        # Convert from uint8 to float and normalize to [0, 1]
        vision_rgb = tensor_state.cameras[self.cfg.camera.name].rgb

        # Display image in OpenCV window if enabled
        if self.enable_opencv_display and self.opencv_renderer is not None:
            # Use the original uint8 RGB image for display (before normalization)
            # vision_rgb is in format (batch_size, height, width, channels)
            display_image = vision_rgb[0]  # Take first environment

            # Display the image and check if window is still open
            window_open = self.opencv_renderer.display(display_image)
            if not window_open:
                # User closed the window, disable further display
                self.enable_opencv_display = False
                print("OpenCV display window closed by user")

    def _compute_observations(self) -> None:
        q = (self.dof_pos - self.default_joint_pd_target) * self.cfg.normalization.obs_scales.dof_pos
        dq = self.dof_vel * self.cfg.normalization.obs_scales.dof_vel
        tensor_states = self.env.get_states()
        wrist_pos = tensor_states.robots[self.robot.name].body_state[:, self.wrist_indices, :7]
        diff = wrist_pos - self.ref_wrist_pos

        ref_wrist_pos_obs = torch.flatten(self.ref_wrist_pos, start_dim=1)  # [num_envs, 14]
        wrist_pos_obs = torch.flatten(wrist_pos, start_dim=1)  # [num_envs, 14]
        diff_obs = torch.flatten(diff, start_dim=1)  # [num_envs, 14]

        self.privileged_obs_buf = torch.cat(
            (
                ref_wrist_pos_obs,  # 14
                wrist_pos_obs,  # 14
                q,  # |A|
                dq,  # |A|
                self.actions,  # |A|
                diff_obs,
            ),
            dim=-1,
        )

        obs_buf = torch.cat(
            (
                diff_obs,  # 3
                q,  # |A|
                dq,  # |A|
                self.actions,
            ),
            dim=-1,
        )

        obs_now = obs_buf.clone()
        self.obs_history.append(obs_now)
        self.critic_history.append(self.privileged_obs_buf)
        obs_buf_all = torch.stack([self.obs_history[i] for i in range(self.obs_history.maxlen)], dim=1)
        self.obs_buf = obs_buf_all.reshape(self.num_envs, -1)
        self.privileged_obs_buf = torch.cat([self.critic_history[i] for i in range(self.cfg.c_frame_stack)], dim=1)
        self.privileged_obs_buf = torch.clip(
            self.privileged_obs_buf, -self.cfg.normalization.clip_observations, self.cfg.normalization.clip_observations
        )

    def _update_target_wp(self, reset_env_ids):
        """Update target wrist positions."""
        # self.target_wp_i specifies which seq to use for each env, and self.target_wp_j specifies the timestep in the seq
        self.ref_wrist_pos = (
            self.target_wp[self.target_wp_i, self.target_wp_j] + self.ori_wrist_pos
        )  # [num_envs, 2, 7], two hands
        self.delayed_obs_target_wp = self.target_wp[
            self.target_wp_i, torch.maximum(self.target_wp_j - self.delayed_obs_target_wp_steps_int, torch.tensor(0))
        ]
        resample_i = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        if self.common_step_counter % self.target_wp_update_steps_int == 0:
            self.target_wp_j += 1
            wp_eps_end_bool = self.target_wp_j >= self.num_wp
            self.target_wp_j = torch.where(wp_eps_end_bool, torch.zeros_like(self.target_wp_j), self.target_wp_j)
            resample_i[wp_eps_end_bool.nonzero(as_tuple=False).flatten()] = True
            self.target_wp_update_steps_int = sample_int_from_float(self.target_wp_update_steps)
            self.delayed_obs_target_wp_steps_int = sample_int_from_float(self.delayed_obs_target_wp_steps)
        if self.cfg.humanoid_extra_cfg.resample_on_env_reset:
            self.target_wp_j[reset_env_ids] = 0
            resample_i[reset_env_ids] = True
        self.target_wp_i = torch.where(
            resample_i, torch.randint(0, self.num_pairs, (self.num_envs,), device=self.device), self.target_wp_i
        )

    def _update_marker_viz(self):
        # convert to world frame
        world_pos = self.ref_wrist_pos[:, :, :3] + self._env_origins[:, None, :3]
        pos = world_pos.reshape(-1, 3)
        ori = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).repeat(pos.shape[0], 1)
        idx = torch.zeros(pos.shape[0], dtype=torch.long, device=self.device)
        self.marker_viz.visualize(pos, ori, marker_indices=idx)

    def _init_target_wp(self, tensor_state: TensorState) -> None:
        self.ori_wrist_pos = (
            tensor_state.robots[self.robot.name].body_state[:, self.wrist_indices, :7].clone()
        )  # [num_envs, 2, 7], two hands
        self.target_wp, self.num_pairs, self.num_wp = sample_wp(
            self.device, num_points=2000000, num_wp=10, ranges=self.command_ranges
        )  # relative, self.target_wp.shape=[num_pairs, num_wp, 2, 7]
        self.target_wp_i = torch.randint(
            0, self.num_pairs, (self.num_envs,), device=self.device
        )  # for each env, choose one seq, [num_envs]
        self.target_wp_j = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )  # for each env, the timestep in the seq is initialized to 0, [num_envs]
        self.target_wp_dt = 1 / self.cfg.humanoid_extra_cfg.freq
        self.target_wp_update_steps = self.target_wp_dt / self.dt  # not necessary integer
        assert self.dt <= self.target_wp_dt, (
            f"self.dt {self.dt} must be less than self.target_wp_dt {self.target_wp_dt}"
        )
        self.target_wp_update_steps_int = sample_int_from_float(self.target_wp_update_steps)

        self.ref_wrist_pos = None
        self.ref_action = self.default_joint_pd_target
        self.delayed_obs_target_wp = None
        self.delayed_obs_target_wp_steps = self.cfg.humanoid_extra_cfg.delay / self.target_wp_dt
        self.delayed_obs_target_wp_steps_int = sample_int_from_float(self.delayed_obs_target_wp_steps)
        self._update_target_wp(torch.tensor([], dtype=torch.long, device=self.device))

    # ==== reward functions ====
    def _reward_wrist_pos(self, tensor_state: TensorState, robot_name: str, cfg: BaseTableHumanoidTaskCfg):
        """Reward for reaching the target position."""
        wrist_pos = tensor_state.robots[robot_name].body_state[:, self.wrist_indices, :7]  # [num_envs, 2, 7], two hands
        wrist_pos_diff = (
            wrist_pos[:, :, :3] - self.ref_wrist_pos[:, :, :3]
        )  # [num_envs, 2, 3], two hands, position only
        wrist_pos_diff = torch.flatten(wrist_pos_diff, start_dim=1)  # [num_envs, 6]
        wrist_pos_error = torch.mean(torch.abs(wrist_pos_diff), dim=1)
        return torch.exp(-4 * wrist_pos_error), wrist_pos_error

    def _reward_upper_body_pos(
        self, states: TensorState, robot_name: str, cfg: BaseTableHumanoidTaskCfg
    ) -> torch.Tensor:
        """Keep upper body joints close to default positions."""
        upper_body_diff = states.robots[robot_name].joint_pos - self.default_joint_pd_target
        upper_body_error = torch.mean(torch.abs(upper_body_diff), dim=1)
        return torch.exp(-4 * upper_body_error), upper_body_error

    def _reward_default_joint_pos(
        self, states: TensorState, robot_name: str, cfg: BaseTableHumanoidTaskCfg
    ) -> torch.Tensor:
        """Keep joint positions close to defaults (penalize yaw/roll)."""
        joint_diff = states.robots[robot_name].joint_pos - self.default_joint_pd_target
        return -0.01 * torch.norm(joint_diff, dim=1)

    def _reward_torques(self, states: TensorState, robot_name: str, cfg: BaseTableHumanoidTaskCfg) -> torch.Tensor:
        """Penalize high torques."""
        return torch.sum(torch.square(states.robots[robot_name].joint_effort_target), dim=1)

    def _reward_dof_vel(self, states: TensorState, robot_name: str, cfg: BaseTableHumanoidTaskCfg) -> torch.Tensor:
        """Penalize high dof velocities."""
        return torch.sum(torch.square(states.robots[robot_name].joint_vel), dim=1)

    def _reward_dof_acc(self, states: TensorState, robot_name: str, cfg: BaseTableHumanoidTaskCfg) -> torch.Tensor:
        """Penalize high DOF accelerations."""
        return torch.sum(
            torch.square((self.last_dof_vel - self.dof_vel) / self.dt),
            dim=1,
        )

    def _check_reset(self):
        self.reset_buf = self.time_out_buf
