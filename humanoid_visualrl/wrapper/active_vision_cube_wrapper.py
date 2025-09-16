"""A wrapper for fixed upper body and use cnn inside the policy class."""

# TODO success filter
# TODO add vision buf into HumanoidBaseWrapper
# render reset frame to before compute obs
from __future__ import annotations

import cv2
import numpy as np
import torch

from humanoid_visualrl.cfg.active_vision_cube_cfg import BaseTableHumanoidTaskCfg
from humanoid_visualrl.wrapper.base_humanoid_wrapper import HumanoidBaseWrapper
from humanoid_visualrl.wrapper.reset_18_extractor import Reset18Extractor
from metasim.types import TensorState
from metasim.utils.math import quat_apply, quat_mul
from loguru import logger as log
from metasim.task.registry import register_task


@register_task("active_vision")
class ActiveVisionWrapper(HumanoidBaseWrapper):
    """Wraps Metasim environments to be compatible with rsl_rl OnPolicyRunner.

    Note that rsl_rl is designed for parallel training fully on GPU, with robust support for Isaac Gym and Isaac Lab.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.image_center_x = self.cfg.cameras[0].width / 2
        self.image_center_y = self.cfg.cameras[0].height / 2
        self.done_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self.feature_extractor = Reset18Extractor(device=self.device)

        self.pixel_reward_offset = torch.exp(
            -torch.sqrt(
                torch.tensor([self.cfg.cameras[0].width ** 2 + self.cfg.cameras[0].height ** 2], device=self.device)
            )
            / 2.0
            / 50.0
        )
        self.sucess_thres = (
            torch.exp(torch.tensor([-10 / 50.0], device=self.device)) - self.pixel_reward_offset
        ).item()
        if self.cfg.curriculum_cube_yaw:
            self.curriculum_cube_yaw_range = 0.2 * self.cfg.randomize_cube_yaw_range
        else:
            self.curriculum_cube_yaw_range = self.cfg.randomize_cube_yaw_range

        self._reset(list(range(self.num_envs)))

        # get segmatic id
        # tensor_state = self.env.get_states()
        # TODO hard code for now
        self.target_id = 2
        self.last_reward = 0.0
        self.last_curriculum_update_step = 0  # Track when curriculum was last updated

        # calucalte camera pos due to the bug that camera is not updated
        if len(self.cfg.cameras) > 0 and self.cfg.cameras[0].mount_to is not None:
            name = self.env.get_body_names(self.robot.name)
            # get virtual mount link by spilt /
            mount_link_name = self.cfg.cameras[0].mount_link
            mount_link_name = mount_link_name.split("/")[0]
            self.camera_mount_link_idx = name.index(mount_link_name)
            self.camera_mount_link_pos = torch.zeros(self.num_envs, 3, device=self.device)
            self.camera_mount_link_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).repeat(
                self.num_envs, 1
            )
            self.camera_quat_w = self.camera_mount_link_quat = torch.tensor(
                [1.0, 0.0, 0.0, 0.0], device=self.device
            ).repeat(self.num_envs, 1)

            self.camera_pos_w = torch.tensor([0.0, 0.0, 0.0]).to(self.device).repeat(self.num_envs, 1)
            self.camera_tran_pos = torch.tensor([0.05762, 0.01753, 0.42987]).to(self.device).repeat(self.num_envs, 1)

            self.camera_tran_quat = torch.tensor([0.91496, 0.0, 0.40355, 0.0]).to(self.device).repeat(self.num_envs, 1)

    def _init_buffers(self):
        super()._init_buffers()
        self.obs_buf_state = torch.zeros(self.num_envs, self.cfg.num_observations, device=self.device)
        self.vision_rgb_buf = torch.zeros(
            self.num_envs, 3, self.cfg.cameras[0].height, self.cfg.cameras[0].width, device=self.device
        )
        self.obs_buf = (self.obs_buf_state, self.vision_rgb_buf)
        # self.wrist_pose = torch.zeros(self.num_envs, 2, 7, device=self.device)

        height, width = self.cfg.cameras[0].height, self.cfg.cameras[0].width

        # 创建坐标网格
        self.y_coords, self.x_coords = torch.meshgrid(
            torch.arange(height, device=self.device), torch.arange(width, device=self.device), indexing="ij"
        )

        self.cube_pose_buf = self.init_states.objects["cube"].root_state[:, :7].clone()
        if "semantic_seg" in self.cfg.cameras[0].data_types:
            self.semantic_seg = True
        else:
            self.semantic_seg = False
        if self.semantic_seg:
            self.vision_seg_buf = torch.zeros(
                self.num_envs,
                self.cfg.cameras[0].height,
                self.cfg.cameras[0].width,
                device=self.device,
                dtype=torch.int32,
            )

    def _refreshed_tensors(self, tensor_state: TensorState):
        super()._refreshed_tensors(tensor_state)
        self.cube_pose_buf = tensor_state.objects["cube"].root_state[:, :7]

        # Convert from HWC (H, W, C) to CHW (C, H, W) format for PyTorch CNN
        # Convert from uint8 to float and normalize to [0, 1]
        vision_rgb = tensor_state.cameras[self.cfg.cameras[0].name].rgb
        self.vision_rgb_buf = vision_rgb.permute(0, 3, 1, 2).float() / 255.0
        # self.resnet_features = self.feature_extractor.extract_visual_features(vision_rgb)
        # vision_seg = tensor_state.cameras[self.cfg.cameras[0].name].instance_id_seg
        if self.semantic_seg:
            self.vision_seg_buf = tensor_state.cameras[self.cfg.cameras[0].name].semantic_seg_data
            self.vision_seg_info = tensor_state.cameras[self.cfg.cameras[0].name].instance_id_seg_id2label
            self._compute_pixel_distance()

        if self.camera_mount_link_idx is not None:
            self.camera_mount_link_pos = tensor_state.robots[self.robot.name].body_state[
                :, self.camera_mount_link_idx, :3
            ]
            self.camera_mount_link_quat = tensor_state.robots[self.robot.name].body_state[
                :, self.camera_mount_link_idx, 3:7
            ]

            self.camera_pos_w = self.camera_mount_link_pos + quat_apply(
                self.camera_mount_link_quat, self.camera_tran_pos
            )
            self.camera_quat_w = quat_mul(self.camera_mount_link_quat, self.camera_tran_quat)

    def _compute_pixel_distance(self):
        # target_id = next(k for k, v in self.vision_seg_info.items() if "cube" in v)

        # 创建掩码：shape (num_envs, height, width)
        mask = self.vision_seg_buf == self.target_id

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
            distance = torch.sqrt((center_x - self.image_center_x) ** 2 + (center_y - self.image_center_y) ** 2)
            # since now we have no pitch dof for waist, we only consider x pixel distance
            # distance = torch.abs(center_x - self.image_center_x)

            # 计算奖励
            self.pixel_rewards_buf[valid_envs] = torch.exp(-distance / 50.0) - self.pixel_reward_offset
            # print(f"rewards: {rewards[0]}")

            # 在env 0的图像上绘制坐标点
            if 0 in torch.where(valid_envs)[0] and self.enable_opencv_display and self.env._render_viewport:
                # 找到env 0在valid_envs中的索引
                env_0_idx = torch.where(valid_envs)[0] == 0
                if env_0_idx.any():
                    env_0_pos = torch.where(env_0_idx)[0][0]
                    # 获取env 0的中心点坐标
                    center_x_0 = int(center_x[env_0_pos].item())
                    center_y_0 = int(center_y[env_0_pos].item())

                    # 获取env 0的RGB图像并转换为numpy格式用于绘制

                    rgb_image = self.vision_rgb_buf[0].permute(1, 2, 0).cpu().numpy()

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
                    )

                    # 更新显示缓冲区
                    # self.vision_rgb_buf[0] = torch.from_numpy(rgb_image).to(self.device)

                    # distance_0 = distance[env_0_pos].item()
                    # distance_text = f"Distance: {distance_0:.1f} px"
                    # font = cv2.FONT_HERSHEY_SIMPLEX
                    # font_scale = 0.6
                    # font_color = (255, 255, 255)  # 白色文字
                    # font_thickness = 2
                    # text_x, text_y = 10, 25

                    # cv2.putText(rgb_image, distance_text, (text_x, text_y),
                    #           font, font_scale, font_color, font_thickness)

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

        # # if pixel distance is less than 10, done
        # self.done_buf = self.pixel_rewards_buf > self.sucess_thres

    def _compute_observations(self) -> None:
        q = (self.dof_pos - self.default_joint_pd_target) * self.cfg.normalization.obs_scales.dof_pos
        dq = self.dof_vel * self.cfg.normalization.obs_scales.dof_vel

        # visual_features = self.resnet_features
        # cube_pose_obs = self.cube_pose_buf

        self.privileged_obs_buf = torch.cat(
            (
                # ref_wrist_pos_obs,  # 14
                # cube_pose_obs,
                # wrist_pos_obs,  # 14
                q,  # |A|
                dq,  # |A|
                self.actions,  # |A|
                # diff_obs,
                # visual_features,
            ),
            dim=-1,
        )

        obs_buf = torch.cat(
            (
                # diff_obs,  # 3
                q,  # |A|
                dq,  # |A|
                self.actions,
                # visual_features,
            ),
            dim=-1,
        )

        obs_now = obs_buf.clone()
        self.obs_history.append(obs_now)
        self.critic_history.append(self.privileged_obs_buf)
        # obs_buf_all = torch.stack([self.obs_history[i] for i in range(self.obs_history.maxlen)], dim=1)
        self.obs_buf = obs_now.reshape(self.num_envs, -1)
        self.privileged_obs_buf = torch.cat([self.critic_history[i] for i in range(self.cfg.c_frame_stack)], dim=1)
        self.privileged_obs_buf = torch.clip(
            self.privileged_obs_buf, -self.cfg.normalization.clip_observations, self.cfg.normalization.clip_observations
        )

        self.obs_buf = (self.obs_buf, self.vision_rgb_buf)
        self.extra_buf["observations"]["critic"] = (self.privileged_obs_buf, self.vision_rgb_buf)

    def _pre_reset_hook(self, env_ids=None):
        # randomly set y of cube in range (-randomize_cube_y_range, randomize_cube_y_range)
        # if self.cfg.randomize_cube_y = True

        if self.cfg.randomization:
            yaw = 2 * (torch.rand(len(env_ids), device=self.device) - 0.5) * self.curriculum_cube_yaw_range

            cube_x = torch.cos(yaw) * self.cfg.randomize_cube_radius
            cube_y = torch.sin(yaw) * self.cfg.randomize_cube_radius
            self.init_states.objects["cube"].root_state[env_ids, 0] = cube_x
            self.init_states.objects["cube"].root_state[env_ids, 1] = cube_y
            self.done_buf[env_ids] = False

    def _post_reset_hook(self, env_ids):
        self.cube_pose_buf[env_ids] = self.init_states.objects["cube"].root_state[env_ids, :7]
        self.env.scene.sensors["camera_first_person"].update(dt=0)
        self.env.sim.render()
        camera_data = self.env.scene.sensors["camera_first_person"].data.output
        self.vision_rgb_buf[env_ids] = camera_data["rgb"][env_ids].permute(0, 3, 1, 2).float() / 255.0
        if self.semantic_seg:
            # 添加分割数据的更新
            self.vision_seg_buf[env_ids] = camera_data["semantic_segmentation"].squeeze(-1)[env_ids]

    def _check_reset(self):
        # move 0.05 to config
        terminate = torch.abs(self.cube_pose_buf[:, 2] - self.cfg.init_states[0]["objects"]["cube"]["pos"][2]) > 0.5
        self.reset_buf = self.timeout_buf | terminate
        # self.reset_buf = self.timeout_buf | terminate | self.done_buf
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

    def _reward_pixel_norm_at_cube(self, tensor_state: TensorState, robot_name: str, cfg: BaseTableHumanoidTaskCfg):
        """Reward for gazing at the cube."""
        # return self.pixel_rewards_buf - self.pixel_reward_offset
        return self.pixel_rewards_buf

    def _reward_look_at_cube(self, tensor_state: TensorState, robot_name: str, cfg: BaseTableHumanoidTaskCfg):
        """Reward for looking at the cube."""
        # 获取相机的世界坐标位置 (num_envs, 3)
        camera_pos = self.camera_pos_w

        # 获取立方体的世界坐标位置 (num_envs, 3)
        cube_pos = tensor_state.objects["cube"].root_state[:, :3]

        # 计算从相机到立方体的方向向量 (num_envs, 3)
        direction_vec = cube_pos - camera_pos
        direction_vec = direction_vec / (torch.norm(direction_vec, dim=1, keepdim=True) + 1e-8)  # 归一化

        # 获取相机的朝向向量 (num_envs, 3)
        # 相机的朝向通常是+X方向（根据CameraState的注释）
        # camera_quat = tensor_state.cameras[self.cfg.cameras[0].name].quat_world  # (num_envs, 4) - (w, x, y, z)
        camera_quat = self.camera_quat_w  # (num_envs, 4) - (w, x, y, z)
        # 将相机的+X轴方向向量转换到世界坐标系
        camera_forward = torch.tensor([1.0, 0.0, 0.0], device=self.device).expand(self.num_envs, 3)
        camera_vec = quat_apply(camera_quat, camera_forward)  # 应用四元数旋转

        # 计算两个向量的点积，得到相似度 (num_envs,)
        dot_product = torch.sum(direction_vec * camera_vec, dim=1)

        # 将点积转换为奖励 - 当相机完全对准立方体时点积为1，奖励最大
        # 使用平滑的奖励函数：当dot_product接近1时奖励接近1
        reward = torch.clamp(dot_product, min=0.0)  # 只考虑正向的对准

        if self.env._render_viewport:
            self._update_marker_viz(camera_pos, camera_quat, direction_vec)

        return reward

    def _update_marker_viz(self, position: torch.Tensor, orientation: torch.Tensor, direction_vec: torch.Tensor):
        # cupdate
        # world_pos = position + self._env_origins[:, :3]
        world_pos = position + self._env_origins[:, :3]
        # move up  0.5 to be clear to see
        world_pos[:, 2] += 0.7
        pos = world_pos

        # 准备两组标记：相机方向（蓝色）和指向立方体的方向（红色）
        # 相机方向使用 camera_quat（已为 (N,4) 形状）
        camera_ori = orientation

        # 指向立方体的方向：从 direction_vec 创建四元数
        # direction_vec 已经是归一化的，我们需要将其转换为四元数
        # 假设默认方向是 +X 轴，计算从 +X 轴到 direction_vec 的旋转四元数
        default_forward = torch.tensor([1.0, 0.0, 0.0], device=self.device).expand(self.num_envs, 3)

        # 计算旋转轴 (cross product)
        cross = torch.cross(default_forward, direction_vec, dim=1)
        # 计算旋转角度 (dot product)
        dot = torch.sum(default_forward * direction_vec, dim=1)
        # 数值稳定：限制到 [-1, 1]
        dot = torch.clamp(dot, -1.0, 1.0)

        # 处理平行向量的情况
        cross_norm = torch.norm(cross, dim=1, keepdim=True)
        cross_normalized = cross / (cross_norm + 1e-8)

        # 计算四元数的 sin(θ/2) 和 cos(θ/2)
        # θ = arccos(dot), 所以 cos(θ/2) = sqrt((1 + cos(θ))/2), sin(θ/2) = sqrt((1 - cos(θ))/2)
        cos_half_angle = torch.sqrt((1 + dot) / 2).unsqueeze(1)
        sin_half_angle = torch.sqrt((1 - dot) / 2).unsqueeze(1)

        # 构建四元数 [w, x, y, z]
        direction_quat = torch.cat(
            [
                cos_half_angle,  # w
                cross_normalized * sin_half_angle,  # x, y, z
            ],
            dim=1,
        )

        # 处理完全相反的向量情况 (dot ≈ -1)
        opposite_mask = dot < -0.999
        if opposite_mask.any():
            # 选择一个垂直轴进行180度旋转
            perp_axis = torch.zeros_like(direction_vec)
            perp_axis[opposite_mask, 1] = 1.0  # 使用Y轴
            direction_quat[opposite_mask] = torch.cat(
                [
                    torch.zeros(opposite_mask.sum(), 1, device=self.device),  # w = 0 (180度旋转)
                    perp_axis[opposite_mask],  # x, y, z
                ],
                dim=1,
            )

        # 处理完全相同的向量情况 (dot ≈ 1)
        same_mask = dot > 0.999
        if same_mask.any():
            direction_quat[same_mask] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).expand(
                same_mask.sum(), 4
            )

        # 指向立方体方向的四元数（已为 (N,4) 形状）
        direction_ori = direction_quat

        # 创建标记索引：0 表示相机方向（蓝色），1 表示指向立方体的方向（红色）
        camera_idx = torch.zeros(pos.shape[0], dtype=torch.long, device=self.device)
        direction_idx = torch.ones(pos.shape[0], dtype=torch.long, device=self.device)

        # 合并位置、方向和索引
        all_pos = torch.cat([pos, pos], dim=0)  # (2N, 3)
        all_ori = torch.cat([camera_ori, direction_ori], dim=0)  # (2N, 4)
        all_idx = torch.cat([camera_idx, direction_idx], dim=0)  # (2N,)

        self.env._marker_viz.visualize(all_pos, all_ori, marker_indices=all_idx)

    def _update_curriculum(self):
        self._update_curriculum_cube_yaw_range()

    def _update_curriculum_cube_yaw_range(self):
        if self.cfg.curriculum_cube_yaw:
            # update curriculum_cube_yaw_range
            # Check curriculum update every 200 steps instead of 50 to prevent too frequent updates
            if (self.common_step_counter % self.cfg.ppo_cfg.num_steps_per_env) % 100 == 0:
                # if average reward added by 0.1
                reward = self.episode_sums["pixel_norm_at_cube"].mean() * self.cfg.reward_weights["pixel_norm_at_cube"]

                # Always update last_reward to track current performance
                reward_improvement = reward - self.last_reward
                self.last_reward = reward

                # Only increase range if there's significant improvement (threshold: 0.01)
                # and we haven't increased range too recently (minimum 800 steps between updates)
                steps_since_last_update = self.common_step_counter - self.last_curriculum_update_step
                if reward_improvement > 0.05 or steps_since_last_update >= 400:
                    if self.curriculum_cube_yaw_range < self.cfg.randomize_cube_yaw_range:
                        self.curriculum_cube_yaw_range += self.cfg.randomize_cube_yaw_range * 0.05
                        self.last_curriculum_update_step = self.common_step_counter
                        log.info(
                            f"curriculum_cube_yaw_range: {self.curriculum_cube_yaw_range}, reward_improvement: {reward_improvement:.4f}"
                        )
