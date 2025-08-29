"""A humanoid base wrapper for skillBench tasks."""
# TODO success filter
# render reset frame to before compute obs
from __future__ import annotations

from collections import deque
from copy import deepcopy

import numpy as np
import torch

from humanoid_visualrl.cfg.humanoidFixedGazingCfg import BaseTableHumanoidTaskCfg
from humanoid_visualrl.utils.utils import (
    sample_int_from_float,
)
from metasim.types import TensorState
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
        self.done_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        # self.marker_viz = self.env.init_marker_viz()
        self.feature_extractor = Reset18Extractor(device=self.device)

        self.pixel_reward_offset = torch.exp(torch.tensor([-self.cfg.camera.width / 2.0 / 50.0], device=self.device))
        self.left_right_flag = 1

    def _init_buffers(self):
        super()._init_buffers()
        # self.wrist_pose = torch.zeros(self.num_envs, 2, 7, device=self.device)

        height, width = self.cfg.camera.height, self.cfg.camera.width

        # 创建坐标网格
        self.y_coords, self.x_coords = torch.meshgrid(
            torch.arange(height, device=self.device), torch.arange(width, device=self.device), indexing="ij"
        )

    def _refreshed_tensors(self, tensor_state: TensorState):
        super()._refreshed_tensors(tensor_state)
        self.cube_pose_buf = tensor_state.objects["cube"].root_state[:, :7]

        # Convert from HWC (H, W, C) to CHW (C, H, W) format for PyTorch CNN
        # Convert from uint8 to float and normalize to [0, 1]
        vision_rgb = tensor_state.cameras[self.cfg.camera.name].rgb
        self.vision_rgb_buf = vision_rgb
        self.resnet_features = self.feature_extractor.extract_visual_features(vision_rgb)
        # vision_seg = tensor_state.cameras[self.cfg.camera.name].instance_id_seg

        self.vision_seg_buf = tensor_state.cameras[self.cfg.camera.name].instance_id_seg
        self.vision_seg_info = tensor_state.cameras[self.cfg.camera.name].instance_id_seg_id2label

        self._compute_pixel_distance()

        # target_id = info["cube"]

        # Convert single channel to three channels by repeating
        # vision_seg shape: [1, 96, 128] -> [1, 96, 128, 3]
        # if vision_seg is not None:
        #     vision_rgb = vision_seg.unsqueeze(-1).repeat(1, 1, 1, 3)  # Repeat the channel dimension 3 times
        # else:
        #     vision_rgb = None

        # Display image in OpenCV window if enabled
        # if self.enable_opencv_display and self.opencv_renderer is not None and vision_rgb is not None:
        #     # Use the original uint8 RGB image for display (before normalization)
        #     # vision_rgb is in format (batch_size, height, width, channels)
        #     display_image = vision_rgb[0]  # Take first environment

        #     # Display the image and check if window is still open
        #     window_open = self.opencv_renderer.display(display_image)
        #     if not window_open:
        #         # User closed the window, disable further display
        #         self.enable_opencv_display = False
        #         print("OpenCV display window closed by user")

    def _compute_pixel_distance(self):
        target_id = next(k for k, v in self.vision_seg_info.items() if "cube" in v)

        # 创建掩码：shape (num_envs, height, width)
        mask = self.vision_seg_buf == target_id

        # 为每个环境计算加权中心点
        self.pixel_rewards_buf = torch.zeros(self.num_envs, device=self.device)

        # 计算每个环境的像素数量
        pixel_counts = mask.sum(dim=(1, 2))

        # 只处理有目标像素的环境
        valid_envs = pixel_counts > 0

        if valid_envs.any():
            # 计算加权中心点
            weighted_y = (mask[valid_envs] * self.y_coords.unsqueeze(0)).sum(dim=(1, 2))  # (num_valid_envs,)
            weighted_x = (mask[valid_envs] * self.x_coords.unsqueeze(0)).sum(dim=(1, 2))  # (num_valid_envs,)

            # 归一化
            center_y = weighted_y / pixel_counts[valid_envs]
            center_x = weighted_x / pixel_counts[valid_envs]

            # 计算距离
            # distance = torch.sqrt((center_x - self.image_center_x) ** 2 + (center_y - self.image_center_y) ** 2)
            # since now we have no pitch dof for waist, we only consider x pixel distance
            distance = torch.sqrt((center_x - self.image_center_x) ** 2)

            # 计算奖励
            self.pixel_rewards_buf[valid_envs] = torch.exp(-distance / 50.0)
            # print(f"rewards: {rewards[0]}")

            # 在env 0的图像上绘制坐标点
            if 0 in torch.where(valid_envs)[0] and self.enable_opencv_display:
                # 找到env 0在valid_envs中的索引
                env_0_idx = torch.where(valid_envs)[0] == 0
                if env_0_idx.any():
                    env_0_pos = torch.where(env_0_idx)[0][0]
                    # 获取env 0的中心点坐标
                    center_x_0 = int(center_x[env_0_pos].item())
                    center_y_0 = int(center_y[env_0_pos].item())

                    # 获取env 0的RGB图像并转换为numpy格式用于绘制
                    import cv2

                    rgb_image = self.vision_rgb_buf[0].cpu().numpy()  # shape: (H, W, C)

                    # 确保图像是uint8格式
                    if rgb_image.dtype != np.uint8:
                        rgb_image = (rgb_image * 255).astype(np.uint8)

                    # 绘制计算出的中心点（红色圆圈）
                    cv2.circle(rgb_image, (center_x_0, center_y_0), 5, (0, 0, 255), -1)  # 红色实心圆

                    # 绘制图像中心点（绿色圆圈）
                    cv2.circle(
                        rgb_image, (int(self.image_center_x), int(self.image_center_y)), 3, (0, 255, 0), -1
                    )  # 绿色实心圆

                    # 绘制连接线
                    cv2.line(
                        rgb_image,
                        (center_x_0, center_y_0),
                        (int(self.image_center_x), int(self.image_center_y)),
                        (255, 255, 0),
                        1,
                    )  # 黄色线

                    # 更新显示缓冲区
                    # self.vision_rgb_buf[0] = torch.from_numpy(rgb_image).to(self.device)

                    if (
                        self.enable_opencv_display
                        and self.opencv_renderer is not None
                        and self.vision_rgb_buf is not None
                    ):
                        # Use the original uint8 RGB image for display (before normalization)
                        # vision_rgb is in format (batch_size, height, width, channels)
                        # display_image = self.vision_rgb_buf[0]  # Take first environment

                        # Display the image and check if window is still open
                        window_open = self.opencv_renderer.display(rgb_image)
                        if not window_open:
                            # User closed the window, disable further display
                            self.enable_opencv_display = False
                            print("OpenCV display window closed by user")

        # if pixel distance is less than 10, done
        self.done_buf = self.pixel_rewards_buf > 0.85

    def _compute_observations(self) -> None:
        q = (self.dof_pos - self.default_joint_pd_target) * self.cfg.normalization.obs_scales.dof_pos
        dq = self.dof_vel * self.cfg.normalization.obs_scales.dof_vel
        # tensor_states = self.env.get_states()
        # wrist_pos = tensor_states.robots[self.robot.name].body_state[:, self.wrist_indices, :7]
        # diff = wrist_pos - self.ref_wrist_pos

        # ref_wrist_pos_obs = torch.flatten(self.ref_wrist_pos, start_dim=1)  # [num_envs, 14]
        # wrist_pos_obs = torch.flatten(wrist_pos, start_dim=1)  # [num_envs, 14]
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

        self.extra_buf["observations"]["critic"] = self.privileged_obs_buf

    def _pre_reset_hook(self, env_ids=None):
        # randomly set x of cube
        self.left_right_flag = self.left_right_flag * -1
        self.init_states.objects["cube"].root_state[env_ids, 1] = (
            torch.rand(len(env_ids), device=self.device) * self.cfg.randomize_cube_y_range
            + self.cfg.randomize_cube_y_offset
        ) * self.left_right_flag
        self.done_buf[env_ids] = False
        # self.pixel_rewards_buf[env_ids] = 0.0
    
    def _post_reset_hook(self, env_ids):
        self.cube_pose_buf[env_ids] = self.init_states.objects["cube"].root_state[env_ids, :7]
        self.resnet_features[env_ids] = torch.zeros(len(env_ids), 512, device=self.device)

    def _check_reset(self):
        # move 0.05 to config
        terminate = torch.abs(self.cube_pose_buf[:, 2] - self.cfg.init_states[0]["objects"]["cube"]["pos"][2]) > 0.5
        self.reset_buf = self.timeout_buf | terminate | self.done_buf
        return self.reset_buf

    # def _reset(self, env_ids=None):
    #     super()._reset(env_ids)
    #     self.resnet_features[env_ids] = torch.zeros(len(env_ids), 512, device=self.device)


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

    def _reward_gaze_at_cube(self, tensor_state: TensorState, robot_name: str, cfg: BaseTableHumanoidTaskCfg):
        """Reward for gazing at the cube."""

        # return self.pixel_rewards_buf - self.pixel_reward_offset
        return self.pixel_rewards_buf
