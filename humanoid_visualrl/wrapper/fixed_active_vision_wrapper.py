"""A humanoid base wrapper for skillBench tasks."""

from __future__ import annotations

from collections import deque
from copy import deepcopy

import numpy as np
import torch

from humanoid_visualrl.cfg.humanoidFixedGazingCfg import BaseTableHumanoidTaskCfg
from humanoid_visualrl.utils.utils import (
    sample_int_from_float,
    sample_wp,
)
from metasim.scenario.scenario import ScenarioCfg
from metasim.types import TensorState
from roboverse_learn.rl.rsl_rl.rsl_rl_wrapper import RslRlWrapper
from humanoid_visualrl.wrapper.base_humanoid_wrapper import HumanoidBaseWrapper
from humanoid_visualrl.wrapper.reset_18_extractor import Reset18Extractor

class ActiveVisionWrapper(HumanoidBaseWrapper):
    """Wraps Metasim environments to be compatible with rsl_rl OnPolicyRunner.

    Note that rsl_rl is designed for parallel training fully on GPU, with robust support for Isaac Gym and Isaac Lab.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


        self.image_center_x = self.cfg.camera.width / 2
        self.image_center_y = self.cfg.camera.height / 2
        # self.marker_viz = self.env.init_marker_viz()
        self.feature_extractor = Reset18Extractor(device=self.device)

    def _init_buffers(self):
        super()._init_buffers()
        # self.wrist_pose = torch.zeros(self.num_envs, 2, 7, device=self.device)

    def _refreshed_tensors(self, tensor_state: TensorState):
        super()._refreshed_tensors(tensor_state)
        self.cube_pose_buf = tensor_state.objects["cube"].root_state[:, :7]

        # Convert from HWC (H, W, C) to CHW (C, H, W) format for PyTorch CNN
        # Convert from uint8 to float and normalize to [0, 1]
        vision_rgb = tensor_state.cameras[self.cfg.camera.name].rgb
        self.resnet_features = self.feature_extractor.extract_visual_features(vision_rgb)
        # vision_seg = tensor_state.cameras[self.cfg.camera.name].instance_id_seg

        self.vision_seg_buf = tensor_state.cameras[self.cfg.camera.name].instance_id_seg
        self.vision_seg_info = tensor_state.cameras[self.cfg.camera.name].instance_id_seg_id2label

        # target_id = info["cube"]

        # Convert single channel to three channels by repeating
        # vision_seg shape: [1, 96, 128] -> [1, 96, 128, 3]
        # if vision_seg is not None:
        #     vision_rgb = vision_seg.unsqueeze(-1).repeat(1, 1, 1, 3)  # Repeat the channel dimension 3 times
        # else:
        #     vision_rgb = None

        # Display image in OpenCV window if enabled
        if self.enable_opencv_display and self.opencv_renderer is not None and vision_rgb is not None:
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
        # diff = wrist_pos - self.ref_wrist_pos

        # ref_wrist_pos_obs = torch.flatten(self.ref_wrist_pos, start_dim=1)  # [num_envs, 14]
        wrist_pos_obs = torch.flatten(wrist_pos, start_dim=1)  # [num_envs, 14]
        # diff_obs = torch.flatten(diff, start_dim=1)  # [num_envs, 14]

        visual_features = self.resnet_features
        cube_pose_obs = self.cube_pose_buf

        self.privileged_obs_buf = torch.cat(
            (
                # ref_wrist_pos_obs,  # 14
                cube_pose_obs,
                # wrist_pos_obs,  # 14
                q,  # |A|
                dq,  # |A|
                self.actions,  # |A|
                # diff_obs,
                visual_features,
            ),
            dim=-1,
        )

        obs_buf = torch.cat(
            (
                # diff_obs,  # 3
                q,  # |A|
                dq,  # |A|
                self.actions,
                visual_features,
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

 

    def _check_reset(self):
        # move 0.05 to config
        terminate = torch.abs(self.cube_pose_buf[:, 2] - self.cfg.init_states[0]["objects"]["cube"]["pos"][2]) > 0.5
        self.reset_buf = self.time_out_buf | terminate
        return self.reset_buf

    # ==== reward functions ====
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

    # def _reward_gaze_at_cube(self, tensor_state: TensorState, robot_name: str, cfg: BaseTableHumanoidTaskCfg):
    #     """Reward for gazing at the cube."""
    #     target_id = next(k for k, v in self.vision_seg_info.items() if "cube" in v)
    #     coords = torch.nonzero(self.vision_seg_buf[0, ..., 0] == target_id)  # (N,2)
    #     # coords[:,1] 是 x (u)，coords[:,0] 是 y (v)

    #     print(coords)


    def _reward_gaze_at_cube(self, tensor_state: TensorState, robot_name: str, cfg: BaseTableHumanoidTaskCfg):
        """Reward for gazing at the cube."""
        target_id = next(k for k, v in self.vision_seg_info.items() if "cube" in v)

        # 创建掩码：shape (num_envs, height, width)
        mask = self.vision_seg_buf == target_id

        # 获取图像尺寸
        height, width = self.vision_seg_buf.shape[1], self.vision_seg_buf.shape[2]
        # image_center_y, image_center_x = height / 2, width / 2

        # 创建坐标网格
        y_coords, x_coords = torch.meshgrid(
            torch.arange(height, device=self.device), torch.arange(width, device=self.device), indexing="ij"
        )

        # 为每个环境计算加权中心点
        rewards = torch.zeros(self.num_envs, device=self.device)

        # 计算每个环境的像素数量
        pixel_counts = mask.sum(dim=(1, 2))  # (num_envs,)

        # 只处理有目标像素的环境
        valid_envs = pixel_counts > 0

        if valid_envs.any():
            # 计算加权中心点
            weighted_y = (mask[valid_envs] * y_coords.unsqueeze(0)).sum(dim=(1, 2))  # (num_valid_envs,)
            weighted_x = (mask[valid_envs] * x_coords.unsqueeze(0)).sum(dim=(1, 2))  # (num_valid_envs,)

            # 归一化
            center_y = weighted_y / pixel_counts[valid_envs]
            center_x = weighted_x / pixel_counts[valid_envs]

            # 计算距离
            distance = torch.sqrt((center_x - self.image_center_x) ** 2 + (center_y - self.image_center_y) ** 2)

            # 计算奖励
            rewards[valid_envs] = torch.exp(-distance / 50.0)

        return rewards


