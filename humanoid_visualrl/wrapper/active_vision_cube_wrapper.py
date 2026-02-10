"""A wrapper for fixed upper body and use cnn inside the policy class."""

from __future__ import annotations

import os

import numpy as np
import torch
from loguru import logger as log

from humanoid_visualrl.cfg.active_vision_cube_cfg import BaseTableHumanoidTaskCfg
from humanoid_visualrl.utils.domain_randomization_helper import DomainRandomizationHelper
from humanoid_visualrl.utils.utils import (
    get_body_reindexed_indices_from_substring,
    get_joint_reindexed_indices_from_substring,
)
from humanoid_visualrl.wrapper.base_humanoid_wrapper import HumanoidBaseWrapper
from metasim.task.registry import register_task

from metasim.types import TensorState
from metasim.utils.math import euler_xyz_from_quat, quat_apply, quat_from_euler_xyz, quat_mul

import cv2


@register_task("active_vision_cube")
class ActiveVisionWrapper(HumanoidBaseWrapper):
    """Wraps Metasim environments to be compatible with rsl_rl OnPolicyRunner.

    Note that rsl_rl is designed for parallel training fully on GPU, with robust support for Isaac Gym and Isaac Lab.
    """

    def __init__(self, *args, **kwargs):
        # Extract parameters that are specific to this subclass
        self.autotune = kwargs.pop("autotune", False)
        phase = kwargs.pop("phase", None)

        if self.autotune:
            self.phase = 0
        else:
            self.phase = phase if phase is not None else self.cfg.phase

        # Pass remaining kwargs to parent class
        super().__init__(*args, **kwargs)

        # Set phase after parent initialization

        # Re-initialize reward functions with phase-specific weights
        # Parent's __init__ already called _prepare_reward_function, but we need to update it
        # with phase-specific weights after phase is set
        self._prepare_reward_function(self.cfg)

        # self.env.filter_collisions(self.robot.name, "object")
        # print(self.env.scene.physics_context.get_filtered_pairs())

        self.image_center_x = self.cfg.cameras[0].width / 2
        self.image_center_y = self.cfg.cameras[0].height / 2
        self.done_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        # self.feature_extractor = Reset18Extractor(device=self.device)

        self.pixel_reward_offset = torch.exp(
            -torch.sqrt(
                torch.tensor([self.cfg.cameras[0].width ** 2 + self.cfg.cameras[0].height ** 2], device=self.device)
            )
            / 2.0
            / self.cfg.reward_pixel_norm_at_object_exp_sharpness
        )
        self.see_flag_avg = 0.0
        # self.pixel_rewards_buf = torch.zeros(self.num_envs, device=self.device)

        self.success_thres = (
            torch.exp(torch.tensor([-10 / 50.0], device=self.device)) - self.pixel_reward_offset
        ).item()
        if self.cfg.curriculum_object_yaw and self.phase == 0:
            self.curriculum_object_yaw_range = (
                self.cfg.curriculum_initial_object_yaw_range * self.cfg.randomize_object_yaw_range
            )
        else:
            self.curriculum_object_yaw_range = self.cfg.randomize_object_yaw_range
        # Initialize episode_metrics if it doesn't exist
        if "episode_metrics" not in self.extra_buf:
            self.extra_buf["episode_metrics"] = {}
        self.extra_buf["episode_metrics"]["curriculum_object_yaw_range"] = self.curriculum_object_yaw_range
        log.info(f"curriculum_object_yaw_range: {self.curriculum_object_yaw_range}")

        self.extra_buf["episode_metrics"]["curriculum_obj_mass"] = self.cfg.objects[1].mass
        # log.info(f"curriculum_object_yaw_range: {self.curriculum_object_yaw_range}")
        self.extra_buf["episode_metrics"]["finger_dist_buf"] = 0.5

        # exit()
        self.see_flag_float = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        self.stage = torch.zeros(self.num_envs, device=self.device, dtype=torch.int32)

        for obj in self.cfg.objects:
            if obj.name == "object":
                self.obj = obj
                break

        self.randomize_robot_yaw_range = self.cfg.randomize_robot_yaw_range * 0.1
        # robot_yaw_limit = self.robot.joint_limits["waist_yaw_joint"]
        # self.robot_yaw_limit = list(robot_yaw_limit)
        # self.robot_yaw_limit[0] = robot_yaw_limit[0] * 0.2
        # self.robot_yaw_limit[1] = robot_yaw_limit[1] * 0.2

        # Get joint indices BEFORE filter_collisions
        self.robot_waist_yaw_joint_indices = get_joint_reindexed_indices_from_substring(
            self.env, self.robot.name, ["waist_yaw_joint"], device=self.device
        )
        self.curriculum_robot_yaw_range = self.cfg.randomize_robot_yaw_range
        right_arm_joint_names = self.robot.right_arm_joints
        joint_names = self.env._get_joint_names(self.robot.name, sort=True)
        self.right_arm_joints_indices = [joint_names.index(jn) for jn in right_arm_joint_names]

        if self.cfg.randomize_material:
            self.domain_randomization_helper = DomainRandomizationHelper(
                self.cfg.mode,
                self.cfg.randomize_cfg,
                self.scenario.objects,
                self.scenario.lights,
                self.num_envs,
                self.env,
                self.cfg.env_spacing,
                self.cfg.seed,
                self.device,
            )
            # randomize all
            if self.domain_randomization_helper.scene_randomizer is not None:
                self.domain_randomization_helper.scene_randomizer(env_ids=list(range(self.num_envs)))
        # for mimic waist yaw joint control
        self.robot_yaw_buffer = torch.zeros((self.num_envs, 1), device=self.device, dtype=torch.float)
        # for pd control
        self.last_robot_yaw_buffer = torch.zeros((self.num_envs, 1), device=self.device, dtype=torch.float)
        self.robot_yaw_buffer_action = torch.zeros((self.num_envs, 1), device=self.device, dtype=torch.float)

        self.mass_curriculum_trigger = False
        self.mass_curriculum_trigger_count = 0

        # Load recorded qpos and cube_pos from npz file
        # the data that cube embedded into the box

        # self._reset(list(range(self.num_envs)))
        self._update_camera_pose = False

        # get segmatic id
        # tensor_state = self.env.get_states()
        # TODO hard code for now
        self.target_id = 2
        self.last_reward = 0.0
        self.last_curriculum_updatpixel_norm_at_objecte_step = 0  # Track when curriculum was last updated

        # Curriculum tracking based on see_flag
        self.see_flag_history = torch.zeros(
            self.cfg.see_flag_his_win_length, self.num_envs, device=self.device, dtype=torch.bool
        )
        self.see_flag_history_ptr = 0
        self.see_flag_history_full = False
        # for pixel distance calculation
        self.see_flag = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self.mask = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self.pixel_counts = torch.zeros(self.num_envs, device=self.device, dtype=torch.int32)
        self.right_wrist_indice = self.wrist_indices[1]
        self.lift_offset = torch.exp(
            -torch.tensor(
                self.cfg.reward_lift_object_exp_shapeness * self.cfg.reward_lift_offset**2, device=self.device
            )
        )

        if "pixel_norm_at_object" in self.cfg.reward_weights:
            self.compute_pixel_distance_reward = True
            self.pixel_rewards_buf = torch.zeros(self.num_envs, device=self.device)
        else:
            self.compute_pixel_distance_reward = False

        self._get_joint_masking_indices()
        self._get_joint_energy_penalty_coffe()
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

        self.vision_seg_buf = torch.zeros(
            self.num_envs, self.cfg.cameras[0].height, self.cfg.cameras[0].width, device=self.device, dtype=torch.int32
        )
        # if self.cfg.randomize_obj_material:
        #     self.env.randomize_obj_material(list(range(self.num_envs)), self.obj)

        # find the objcfg with name "object"
        # find the objcfg with name "object"
        # randomize episode length buffer
        self.episode_length_buf = torch.randint_like(
            self.episode_length_buf, high=int(self.cfg.max_episode_length_s / self.dt)
        )

        self._ema_reward = 0.05
        self.last_curriculum_update_step = 0

        # obj randomizer
        from metasim.randomization.object_randomizer import ObjectRandomCfg, ObjectRandomizer

        self.obj_randomizer = ObjectRandomizer(cfg=ObjectRandomCfg(obj_name="object"), device=self.device)
        self.obj_randomizer.bind_handler(self.env)

        self.vega_stretch_joint_pos = torch.tensor(
            [
                -1.1486,
                0.0509,
                0.1644,
                -2.1385,
                0.1022,
                -0.2556,
                0.5274,
                0.0,
                0.0,  # R_ff_j1
                0.0,  # R_mf_j1
                0.0,  # R_rf_j1
                0.0,  # R_lf_j1
                0.0,
                0.0,
                0.0,
                1.47,
                -0.10,
                0.0,
                -0.7990,
                -0.1641,
                0.1135,
            ],
            device="cuda:0",
            # get indices of the arms joints
        )
        if self.phase == 2 and self.cfg.task_name == "active_vision_cube":
            self.init_states.robots["vega"].joint_pos[: self.num_envs // 2] = self.vega_stretch_joint_pos.repeat(
                self.num_envs // 2, 1
            )

        self.finger_dist_buf = torch.ones(self.num_envs, device=self.device)

        # Track previous phase to detect phase changes
        self.previous_phase = self.phase

        npz_path = "outputs/active_vision_cube/2026_0129_220225/ik_curriculum_data_active_vision_cube.npz"
        if os.path.exists(npz_path):
            data = np.load(npz_path)
            recorded_qpos_raw = torch.tensor(data["qpos"], device=self.device, requires_grad=False)  # [400, num_joints]
            # from isaacsim deleted joint version to none deleted version
            self.recorded_qpos = recorded_qpos_raw[:, self.env._none_static_joint_idx_reindexed]
            self.recorded_cube_pos = torch.tensor(data["cube_pos"], device=self.device, requires_grad=False)

        #
        # if self.phase == 2:
        #     self.step_limits[: self.num_envs // 2] = int(self.step_limits[0] / 3)

        self.success = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool, requires_grad=False)

    def _parse_indices(self, robot):
        super()._parse_indices(robot)
        if self.robot.name == "g1_static_dex1":
            # self.right_hand_palm_indices = get_body_reindexed_indices_from_substring(
            #     self.env, self.robot.name, self.robot.right_hand_palm_links, device=self.device
            # )
            self.left_index_intermediate_link_indices = get_body_reindexed_indices_from_substring(
                self.env, self.robot.name, ["left_hand_Link1_1"], device=self.device
            )
            # self.right_hand_palm_indices = self.wrist_indices
        elif self.robot.name == "g1_static_inpire_left_fixed":
            self.right_hand_palm_indices = get_body_reindexed_indices_from_substring(
                self.env, self.robot.name, self.robot.right_hand_palm_links, device=self.device
            )
            # self.right_index_intermediate_link_indices = get_body_cyreindexed_indices_from_substring(
            #     self.env, self.robot.name, self.robot.right_index_intermediate_link, device=self.device
            # )
        elif self.robot.name == "vega":
            self.left_index_intermediate_link_indices = get_body_reindexed_indices_from_substring(
                self.env,
                self.robot.name,
                self.robot.tip_link_names,
                device=self.device,
            )

            # use middle finger to palm pose
            self.right_palm_index = get_body_reindexed_indices_from_substring(
                self.env,
                self.robot.name,
                self.robot.right_palm_link,
                device=self.device,
            )

    def _init_buffers(self):
        super()._init_buffers()
        self.obs_buf_state = torch.zeros(self.num_envs, self.cfg.num_observations, device=self.device)
        self.vision_rgb_buf = torch.zeros(
            self.num_envs,
            3,
            self.cfg.cameras[0].height,
            self.cfg.cameras[0].width,
            device=self.device,
            dtype=torch.uint8,
        )
        self.obs_buf = (self.obs_buf_state, self.vision_rgb_buf)
        # self.wrist_pose = torch.zeros(self.num_envs, 2, 7, device=self.device)

        height, width = self.cfg.cameras[0].height, self.cfg.cameras[0].width

        # 创建坐标网格
        self.y_coords, self.x_coords = torch.meshgrid(
            torch.arange(height, device=self.device), torch.arange(width, device=self.device), indexing="ij"
        )

        self.object_pose_buf = self.init_states.objects["object"].root_state[:, :7].clone()

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

    def success_checker(self, object_pose_buf: torch.Tensor):
        if self.phase == 2:
            object_z_thres = object_pose_buf[:, 2] > self.cfg.success_thres_z
            success = object_z_thres
            return success
        else:
            return torch.zeros(self.num_envs, device=self.device, dtype=torch.bool, requires_grad=False)

    def _refreshed_tensors(self, tensor_state: TensorState):
        super()._refreshed_tensors(tensor_state)

        # ====== update object pose ======
        self.object_pose_buf = tensor_state.objects["object"].root_state[:, :7]

        # ======update vision rgb and seg======
        # Convert from HWC (H, W, C) to CHW (C, H, W) format for PyTorch CNN
        # Convert from uint8 to float and normalize to [0, 1]
        # vision_rgb = tensor_state.cameras[self.cfg.cameras[0].name].rgb
        # TODO: normalize it to get better results?
        # # mean_tensor = torch.mean(vision_rgb, dim=(1, 2), keepdim=True)
        # vision_rgb -= 0.5

        # mean_tensor = torch.mean(vision_rgb, dim=(1, 2), keepdim=True)

        # self.vision_rgb_buf = vision_rgb.permute(0, 3, 1, 2)
        # if self.common_step_counte
        # if self.common_step_counter % self.cfg.vision_slow_down_scale == 0:
        self.vision_rgb_buf = tensor_state.cameras[self.cfg.cameras[0].name].rgb.permute(0, 3, 1, 2).clone()
        # else:
        #     self.vision_rgb_buf = torch.zeros(self.num_envs, 3, self.cfg.cameras[0].height, self.cfg.cameras[0].width, device=self.device, dtype=torch.uint8)
        # self.vision_rgb_buf.copy_(tensor_state.cameras[self.cfg.cameras[0].name].rgb.permute(0, 3, 1, 2))
        # self.resnet_features = self.feature_extractor.extract_visual_features(vision_rgb)
        # vision_seg = tensor_state.cameras[self.cfg.cameras[0].name].instance_id_seg
        if self.semantic_seg:
            self.vision_seg_buf = tensor_state.cameras[self.cfg.cameras[0].name].semantic_seg_data
            self.vision_seg_info = tensor_state.cameras[self.cfg.cameras[0].name].instance_id_seg_id2label

        # uncomment this if you want to use the camera mount link(generally for camera pose usage)
        if self._update_camera_pose:
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

        # ========== update see flag ======
        # 创建掩码：shape (num_envs, height, width)
        self.mask = self.vision_seg_buf == self.target_id

        # 为每个环境计算加权中心点
        # self.pixel_rewards_buf = torch.zeros(self.num_envs, device=self.device)

        # 计算每个环境的像素数量
        self.pixel_counts = self.mask.sum(dim=(1, 2))

        # 只处理有目标像素的环境
        valid_envs = self.pixel_counts > 0
        self.see_flag = valid_envs.clone()
        # self.see_flag_float = self.see_flag.float()
        self.see_flag_float = self.see_flag.float()

        # Update see_flag history for curriculum
        self._update_see_flag_history()

        if self.compute_pixel_distance_reward or (self.enable_opencv_display and self.env._render_viewport):
            self._compute_pixel_distance()

        self.last_robot_yaw_buffer = self.robot_yaw_buffer.clone()
        # from met
        _, _, robot_yaw_buffer = euler_xyz_from_quat(tensor_state.robots[self.robot.name].root_state[:, 3:7])
        # greater than \pi just subject 2 pi

        robot_yaw_buffer[robot_yaw_buffer > torch.pi] -= 2 * torch.pi
        self.robot_yaw_buffer[:, 0] = robot_yaw_buffer

        # update finger dist
        finger_tip_pos = tensor_state.robots[self.robot.name].body_state[
            :, self.left_index_intermediate_link_indices, :3
        ]

        # get mean distance from finger tips to object, then subtract expected contact distance
        # Note: This computes the mean distance across all finger tips, then subtracts 0.015m (expected contact distance)
        self.finger_dist_buf = torch.norm(finger_tip_pos - self.object_pose_buf[:, None, :3], dim=2).mean(dim=1)

        # check stage change
        self.extra_buf["episode_metrics"]["finger_dist_buf"] = self.finger_dist_buf.mean()
        self.extra_buf["episode_metrics"]["phase"] = self.phase

        # Check phase change and update reward functions if needed
        if self.autotune:
            self._check_phase()

    def _stage_checker(self):
        # for search and pick task, stage is
        # 0: search
        # 1: reaching
        # 2: grasp and lift
        # TODO implement this
        pass

    def _compute_pixel_distance(self):
        if self.see_flag.any():
            weighted_y = (self.mask[self.see_flag] * self.y_coords.unsqueeze(0)).sum(dim=(1, 2))  # (num_valid_envs,)
            weighted_x = (self.mask[self.see_flag] * self.x_coords.unsqueeze(0)).sum(dim=(1, 2))  # (num_valid_envs,)

            self.center_y = weighted_y / self.pixel_counts[self.see_flag]
            self.center_x = weighted_x / self.pixel_counts[self.see_flag]

        # Display the image and check if window is still open

        # if self.opencv_render_env_idx in torch.where(self.see_flag)[0]:

        # if specific env draw
        if self.env._render_viewport and self.enable_opencv_display:
            rgb_image = self.vision_rgb_buf[self.opencv_render_env_idx]

            # rgb_image = torch.clamp(rgb_image, 0, 1)
            # 确保转换为可写的、连续的 numpy 数组，以便 OpenCV 可以修改
            rgb_image = rgb_image.permute(1, 2, 0).contiguous().cpu().numpy().copy()
            env_idx = torch.where(self.see_flag)[0] == self.opencv_render_env_idx
            # 确保图像是uint8格式
            # rgb_image = self.vision_rgb_buf[self.opencv_render_env_idx] + 0.3
            # ensure the image is in the range of [0, 1]
            if env_idx.any():
                env_pos = torch.where(env_idx)[0][0]

                weighted_y = (self.mask[self.see_flag] * self.y_coords.unsqueeze(0)).sum(
                    dim=(1, 2)
                )  # (num_valid_envs,)
                weighted_x = (self.mask[self.see_flag] * self.x_coords.unsqueeze(0)).sum(
                    dim=(1, 2)
                )  # (num_valid_envs,)

                # 获取env idx的中心点坐标
                center_x = int(self.center_x[env_pos].item())
                center_y = int(self.center_y[env_pos].item())

                if rgb_image.dtype != np.uint8:
                    rgb_image = (rgb_image * 255).astype(np.uint8)

                cv2.circle(rgb_image, (center_x, center_y), 5, (0, 0, 255), -1)  # 红色实心

                # 绘制中空绿色圆圈（半径15像素）
                cv2.circle(
                    rgb_image,
                    (int(self.image_center_x), int(self.image_center_y)),
                    self.cfg.thres_radius,
                    (0, 255, 0),
                    2,
                )

                # 绘制图像中心点（绿色圆圈）
                cv2.circle(
                    rgb_image, (int(self.image_center_x), int(self.image_center_y)), 3, (0, 255, 0), -1
                )  # 绿色实心圆

                cv2.line(
                    rgb_image,
                    (center_x, center_y),
                    (int(self.image_center_x), int(self.image_center_y)),
                    (255, 255, 0),
                    1,
                )

            window_open = self.opencv_renderer.display(rgb_image)

            if not window_open:
                # User closed the window, disable further display
                self.enable_opencv_display = False

    def _update_see_flag_history(self):
        """Update the see_flag history buffer for curriculum learning."""
        # Store current see_flag in history
        self.see_flag_history[self.see_flag_history_ptr] = self.see_flag

        # Update pointer
        self.see_flag_history_ptr += 1
        if self.see_flag_history_ptr >= self.cfg.see_flag_his_win_length:
            self.see_flag_history_ptr = 0
            self.see_flag_history_full = True

    def _compute_velocity(self, actions):
        """Compute velocity from actions. for L_wheel_j1 and R_wheel_j1."""
        # scale the actions (generally output from policy)
        action_scaled = self.scenario.task.action_scale * actions
        velocity = self._pv_gains * (action_scaled - self.robot_yaw_buffer) - self._dv_gains * (
            self.robot_yaw_buffer - self.last_robot_yaw_buffer
        )
        # velocity *= -1
        velocity = torch.clip(velocity, -1000, 1000)
        return velocity

    def _compute_observations(self) -> None:
        q = (self.dof_pos - self.default_joint_pd_target) * self.cfg.normalization.obs_scales.dof_pos
        dq = self.dof_vel * self.cfg.normalization.obs_scales.dof_vel
        obs = torch.cat(
            (
                q,
                dq,
                self.actions,
            ),
            dim=-1,
        )
        self.obs_buf_state = obs
        self.privileged_obs_buf = obs
        self.obs_buf = (self.obs_buf_state, self.vision_rgb_buf)
        self.extra_buf["observations"]["critic"] = (self.privileged_obs_buf, self.vision_rgb_buf)
        self.extra_buf["episode_metrics"]["curriculum_object_yaw_range"] = self.curriculum_object_yaw_range

    def _pre_reset_hook(self, env_ids=None):
        if self.cfg.randomize_material:
            self.domain_randomization_helper.randomization(env_ids=env_ids, step_count=self.common_step_counter)

        if self.cfg.randomization:
            # yaw: 物体相对于机器人的方位角（用于计算物体位置）
            object_relative_yaw = (
                2 * (torch.rand(len(env_ids), device=self.device) - 0.5) * self.curriculum_object_yaw_range
            )
            # if occlusion cube yaw is close to 45 degree, add radius

            # radius bias randomize_object_radius_range
            radius_bias = (
                2 * (torch.rand(len(env_ids), device=self.device) - 0.5) * self.cfg.randomize_object_radius_range
            )
            radius = self.cfg.randomize_object_radius + radius_bias

            # set occlude cube position in front of the object

            if self.cfg.occlude_cube:
                occlusion_cube_radius = radius - 0.1
                # +0.1 radius random 抖动
                occlusion_cube_yaw = (
                    object_relative_yaw
                    + (torch.rand(len(env_ids), device=self.device) - 0.5) * 2 * self.cfg.occlude_cube_yaw_range
                )
                occlusion_cube_x = torch.cos(occlusion_cube_yaw) * occlusion_cube_radius
                occlusion_cube_y = torch.sin(occlusion_cube_yaw) * occlusion_cube_radius
                self.init_states.objects["occlusion_cube"].root_state[env_ids, 0] = occlusion_cube_x
                self.init_states.objects["occlusion_cube"].root_state[env_ids, 1] = occlusion_cube_y
                self.init_states.objects["occlusion_cube"].root_state[env_ids, 3:7] = quat_from_euler_xyz(
                    torch.zeros(len(env_ids), device=self.device),
                    torch.zeros(len(env_ids), device=self.device),
                    occlusion_cube_yaw,
                )

            object_x = torch.cos(object_relative_yaw) * radius
            object_y = torch.sin(object_relative_yaw) * radius
            self.init_states.objects["object"].root_state[env_ids, 0] = object_x
            self.init_states.objects["object"].root_state[env_ids, 1] = object_y
            object_rotation_yaw = 2 * (torch.rand(len(env_ids), device=self.device) - 0.5) * 3.14
            quat = quat_from_euler_xyz(
                torch.zeros(len(env_ids), device=self.device),
                torch.zeros(len(env_ids), device=self.device),
                object_rotation_yaw,
            )
            self.init_states.objects["object"].root_state[env_ids, 3:7] = quat

            # if robot is in the phase 2, align robot base yaw joint to face the object
            # set those pos from ik
            if self.phase == 2:
                # only those < num_envs//2 are in stretch pose
                # Convert env_ids to tensor for comparison
                env_ids_tensor = torch.tensor(env_ids, device=self.device, dtype=torch.long)
                mask = env_ids_tensor < self.num_envs // 2
                stretch_env_ids = env_ids_tensor[mask]

                # align robot base yaw joint to face the object
                if len(stretch_env_ids) > 0:
                    env_ids_tensor = torch.tensor(env_ids, device=self.device, dtype=torch.long)
                    mask = env_ids_tensor < self.num_envs // 2
                    stretch_env_ids = env_ids_tensor[mask]

                    # align robot base yaw joint to face the object
                    if len(stretch_env_ids) > 0:
                        # If we have recorded poses, randomly sample from them
                        if self.recorded_qpos is not None and self.recorded_cube_pos is not None:
                            num_stretch = len(stretch_env_ids)
                            # Randomly select indices from recorded poses
                            selected_indices = torch.randint(
                                0, len(self.recorded_qpos), (num_stretch,), device=self.device
                            )

                            # Directly copy recorded cube positions to object init_state
                            self.init_states.objects["object"].root_state[stretch_env_ids, :7] = self.recorded_cube_pos[
                                selected_indices, :
                            ]

                            self.init_states.robots["vega"].joint_pos[stretch_env_ids, :] = self.recorded_qpos[
                                selected_indices, :
                            ]

    def _check_phase(self):
        if self.phase == 2:
            return True
        elif self.phase == 1:
            # reaching --> reverse
            if self.finger_dist_buf.mean() < self.cfg.stage_finger_close_to_object_change_thres:
                old_phase = self.phase
                self.phase = 2
                # dynamics time limit
                # if reverse reset, only 280 / 3 = 93 steps = 2.325 s to lift up to success threshold
                # self.step_limits[: self.num_envs // 2] = int(self.step_limits[0] / 3)

                # open hand mask in self.right_arm_joints_indices
                # relax all joint masking
                self.action_masking.fill_(1.0)
                # Update reward functions when phase changes
                if old_phase != self.phase:
                    self._update_reward_functions_for_phase()
        elif self.phase == 0:
            # searching --> reaching
            if abs(self.curriculum_object_yaw_range - self.cfg.randomize_object_yaw_range) < 1e-4:
                old_phase = self.phase
                self.phase = 1
                # open arm mask
                # relax right arm joint masking
                for idx in self.right_arm_joints_indices:
                    self.action_masking[idx] = 1.0
                if old_phase != self.phase:
                    self._update_reward_functions_for_phase()
        else:
            raise ValueError(f"Invalid phase: {self.phase}")

    def _post_reset_hook(self, env_ids):
        self.stage[env_ids] = 0
        self.object_pose_buf[env_ids] = self.init_states.objects["object"].root_state[env_ids, :7]
        self.env.scene.sensors["camera_first_person"].update(dt=0)
        self.env.sim.render()
        camera_data = self.env.scene.sensors["camera_first_person"].data.output
        self.vision_rgb_buf[env_ids] = camera_data["rgb"][env_ids].permute(0, 3, 1, 2)
        if self.semantic_seg:
            # 添加分割数据的更新
            self.vision_seg_buf[env_ids] = camera_data["semantic_segmentation"].squeeze(-1)[env_ids]
        # FIXME: this is a hack to reset the object_showup
        self.see_flag_float[env_ids] = 0.0
        self.robot_yaw_buffer[env_ids] = 0.0
        self.last_robot_yaw_buffer[env_ids] = 0.0
        self.robot_yaw_buffer_action[env_ids] = 0.0

    def _check_reset(self):
        # cube fall down
        terminate = self.cfg.init_states[0]["objects"]["object"]["pos"][2] - self.object_pose_buf[:, 2] > 0.1
        # cube success: lift up to z threshold
        success = self.success_checker(self.object_pose_buf)

        self.extra_buf["episode_metrics"]["success"] = self.success.float().mean()
        too_far = torch.norm(self.object_pose_buf[:, :2], dim=1) > (self.cfg.success_thres_z)
        failure = self.timeout_buf | terminate | too_far
        # set self.success to False if failure
        self.success[failure] = False
        # set self.success to True if success
        self.success[success] = True

        self.reset_buf = failure | success
        return self.reset_buf

    def _reward_pixel_norm_at_object(self, tensor_state: TensorState, robot_name: str, cfg: BaseTableHumanoidTaskCfg):
        """Reward for gazing at the object."""
        # return self.pixel_rewards_buf - self.pixel_reward_offset
        self.pixel_rewards_buf.zero_()

        if self.see_flag.any():
            # 计算加权中心点
            weighted_y = (self.mask[self.see_flag] * self.y_coords.unsqueeze(0)).sum(dim=(1, 2))  # (num_valid_envs,)
            weighted_x = (self.mask[self.see_flag] * self.x_coords.unsqueeze(0)).sum(dim=(1, 2))  # (num_valid_envs,)

            # 归一化
            center_y = weighted_y / self.pixel_counts[self.see_flag]
            center_x = weighted_x / self.pixel_counts[self.see_flag]

            # 计算距离
            distance = torch.sqrt((center_x - self.image_center_x) ** 2 + (center_y - self.image_center_y) ** 2)
            # since now we have no pitch dof for waist, we only consider x pixel distance
            # distance = torch.abs(center_x - self.image_center_x)

            self.pixel_rewards_buf[self.see_flag] = (
                torch.exp(-distance / self.cfg.reward_pixel_norm_at_object_exp_sharpness) - self.pixel_reward_offset
            )
        # ema calculate the average reward

        self._ema_reward = (
            self.cfg.ema_alpha * self.pixel_rewards_buf.mean() + (1 - self.cfg.ema_alpha) * self._ema_reward
        )
        # print(f"ema_reward: {self._ema_reward}")
        return self.pixel_rewards_buf

    def _reward_see_object(self, tensor_state: TensorState, robot_name: str, cfg: BaseTableHumanoidTaskCfg):
        """Reward for seeing the object."""
        return self.see_flag_float

    def _reward_object_showup(self, tensor_state: TensorState, robot_name: str, cfg: BaseTableHumanoidTaskCfg):
        """Reward for being in the pixel range of the object."""
        return self.see_flag_float

    def _reward_hand_to_object_dist(self, tensor_state: TensorState, robot_name: str, cfg: BaseTableHumanoidTaskCfg):
        hand_pos = tensor_state.robots[robot_name].body_state[:, self.right_hand_palm_indices, :3]
        hand_to_object_pos_error = torch.norm(hand_pos - self.object_pose_buf[:, None, :3], dim=-1).max(dim=-1).values
        return self.see_flag_float * torch.exp(
            -self.cfg.reward_hand_object_dist_exp_sharpness * hand_to_object_pos_error
        )

    def _reward_finger_close_to_object(self, tensor_state: TensorState, robot_name: str, cfg: BaseTableHumanoidTaskCfg):

        # get mean

        reward = self.see_flag_float * torch.exp(
            -self.cfg.reward_wrist_close_to_object_exp_sharpness * self.finger_dist_buf
        )

        # TODO integrate into the auto masking scheduling
        if self.cfg.curriculum_object_mass_flag and not self.mass_curriculum_trigger:
            dist_mean = self.finger_dist_buf.mean()
            if dist_mean < self.cfg.stage_finger_close_to_object_change_thres:
                self.mass_curriculum_trigger_count += 1
                log.info(f"mass_curriculum_trigger_count: {self.mass_curriculum_trigger_count}, dist_mean: {dist_mean}")
                if self.mass_curriculum_trigger_count > 30:
                    self.start_mass_curriculum_iter = int(self.common_step_counter / self.cfg.ppo_cfg.num_steps_per_env)
                    self.end_mass_curriculum_iter = (
                        self.start_mass_curriculum_iter + self.cfg.curriculum_object_mass_update_interval
                    )
                    self.mass_curriculum_trigger = True

                    log.info(
                        f"UPDATE curriculum_object_mass: start_mass_curriculum_iter: {self.start_mass_curriculum_iter}, end_mass_curriculum_iter: {self.end_mass_curriculum_iter}"
                    )

                    # release finger mask, set all finger mask to 1.0

        dist_close_to_object = self.finger_dist_buf < self.cfg.stage_finger_close_to_object_change_thres
        # assign those both are stage 0 and dist_close_to_object to stage 1
        self.stage[dist_close_to_object] = 1
        return reward

    def _reward_stage(self, tensor_state: TensorState, robot_name: str, cfg: BaseTableHumanoidTaskCfg):
        return self.stage

    def _reward_energy_consumption(self, tensor_state: TensorState, robot_name: str, cfg: BaseTableHumanoidTaskCfg):
        torque = tensor_state.robots[robot_name].joint_effort[:, self.actuated_index] * self.action_masking
        return torch.sum(torch.square(torque), dim=1)

    def _reward_goal_object_dist(self, tensor_state: TensorState, robot_name: str, cfg: BaseTableHumanoidTaskCfg):
        # TODO: implement this
        pass

    def _reward_lift_object(self, tensor_state: TensorState, robot_name: str, cfg: BaseTableHumanoidTaskCfg):
        # offset

        # when the right wrist is close to the object, this reward go up
        dist = torch.square(self.object_pose_buf[:, 2] - self.cfg.reward_lift_object_z)
        return self.see_flag_float * (torch.exp(-self.cfg.reward_lift_object_exp_shapeness * dist) - self.lift_offset)

    def _reward_contact_force(self, tensor_state: TensorState, robot_name: str, cfg: BaseTableHumanoidTaskCfg):
        """Reward for contact. thums 5 times important than other 4, encourage 5 fingers simultaneously contact the object."""
        contact_force_1 = self.env.contact_sensor_1.data.force_matrix_w.squeeze(1).squeeze(1)
        contact_force_2 = self.env.contact_sensor_2.data.force_matrix_w.squeeze(1).squeeze(1)
        contact_force_3 = self.env.contact_sensor_3.data.force_matrix_w.squeeze(1).squeeze(1)
        contact_force_4 = self.env.contact_sensor_4.data.force_matrix_w.squeeze(1).squeeze(1)
        contact_force_5 = self.env.contact_sensor_5.data.force_matrix_w.squeeze(1).squeeze(1)
        contact_force_6 = self.env.contact_sensor_6.data.force_matrix_w.squeeze(1).squeeze(1)
        contact_force_matrix_sum = (
            (torch.norm(contact_force_1, dim=1) > 0.0).float()
            + (torch.norm(contact_force_2, dim=1) > 0.0).float()
            + (torch.norm(contact_force_3, dim=1) > 0.0).float()
            + (torch.norm(contact_force_4, dim=1) > 0.0).float()
            + 3.0 * (torch.norm(contact_force_5, dim=1) > 0.0).float()
            + 3.0 * (torch.norm(contact_force_6, dim=1) > 0.0).float()
        )

        # z axis force downward
        self.upward_force = (
            contact_force_1[:, 2]
            + contact_force_2[:, 2]
            + contact_force_3[:, 2]
            + contact_force_4[:, 2]
            + 2.5 * contact_force_5[:, 2]
            + 2.5 * contact_force_6[:, 2]
        )
        # print(contact_for≤ce_matrix_sum[0])
        return contact_force_matrix_sum

    def _reward_contact_force_upward(self, tensor_state: TensorState, robot_name: str, cfg: BaseTableHumanoidTaskCfg):
        #  if > 0 penalty, if < 0 reward
        return -torch.clamp(self.upward_force, min=-25.0, max=0.0)
        # return self.upward_force

    # def _reward_contact_force_two_much_penalty(self, tensor_state: TensorState, robot_name: str, cfg: BaseTableHumanoidTaskCfg):
    #     contact_force_1 = self.env.contact_sensor_1.data.net_forces_w
    #     contact_force_2 = self.env.contact_sensor_2.data.net_forces_w
    #     contact_force_3 = self.env.contact_sensor_3.data.net_forces_w
    #     contact_force_4 = self.env.contact_sensor_4.data.net_forces_w
    #     contact_force_5 = self.env.contact_sensor_5.data.net_forces_w
    #     contact_force_6 = self.env.contact_sensor_6.data.net_forces_w
    #     # contact_force_matrix_sum = torch.sum
    #     return 1

    # def _reward_curl_pose(self, tensor_state: TensorState, robot_name: str, cfg: BaseTableHumanoidTaskCfg):
    #     # TODO: define curl pose
    #     pass
    def _reward_action_smoothness(self, tensor_state: TensorState, robot_name: str, cfg: BaseTableHumanoidTaskCfg):
        action_smoothness = torch.sum(torch.square(self.last_actions - self.actions), dim=1)
        return action_smoothness

    def _update_marker_viz(self, position: torch.Tensor, orientation: torch.Tensor, direction_vec: torch.Tensor):
        # cupdate
        # world_pos = position + self._env_origins[:, :3]
        world_pos = position + self._env_origins[:, :3]
        # move up  0.5 to be clear to see
        # world_pos[:, 2] += 0.7
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

    def _randomize_occlusion_cube(self, object_state, sample_object_yaw):
        # in front and around the object, random the yaw.
        occlusion_cube_radius = torch.ones(self.num_envs, device=self.device) * (
            self.cfg.randomize_object_radius - 0.13
        )
        occlusion_cube_x = torch.cos(sample_object_yaw) * occlusion_cube_radius
        occlusion_cube_y = torch.sin(sample_object_yaw) * occlusion_cube_radius
        occlusion_cube_state = self.init_states.objects["occlusion_cube"].root_state
        occlusion_cube_state[:, 0] = occlusion_cube_x
        occlusion_cube_state[:, 1] = occlusion_cube_y

        # if too close to the object, move it left or right randomly
        # too close distance env id list
        too_close_env_ids = torch.norm(occlusion_cube_state[:, :3] - object_state[:, :3], dim=1) < 0.1

        if too_close_env_ids.sum() > 0:
            too_close_env_ids = too_close_env_ids.nonzero().squeeze()
            yaw_too_close = (
                sample_object_yaw[too_close_env_ids]
                + (torch.randint(0, 2, (too_close_env_ids.shape[0],), device=self.device) * 2 - 1)
                * (torch.rand(too_close_env_ids.shape[0], device=self.device) + 1)
                * 0.15
            )
            occlusion_cube_state[too_close_env_ids, 0] = (
                torch.cos(yaw_too_close) * occlusion_cube_radius[too_close_env_ids]
            )
            occlusion_cube_state[too_close_env_ids, 1] = (
                torch.sin(yaw_too_close) * occlusion_cube_radius[too_close_env_ids]
            )
            # occlusion_cube_state[too_close_env_ids, 3:7] = quat_from_euler_xyz(
            #     torch.zeros(too_close_env_ids.shape[0], device=env_wrapper.device),
            #     torch.zeros(too_close_env_ids.shape[0], device=env_wrapper.device),
            #     yaw_too_close,
            #
            self.env._set_object_pose(
                self.cfg.objects[3],
                occlusion_cube_state[too_close_env_ids, :3],
                occlusion_cube_state[too_close_env_ids, 3:7],
                env_ids=too_close_env_ids,
            )

    # def _reward_lift_object(self, tensor_state, robot_name, cfg):
    #     finger_tip_pos = tensor_state.robots[robot_name].body_state[:, self.left_index_intermediate_link_indices, :3]
    #     # get mean
    #     dist = torch.norm(finger_tip_pos[:, :, :3] - self.object_pose_buf[:, None, :3], dim=2).mean(dim=1)

    #     close = (dist < self.cfg.reward_lift_object_z).float()  # e.g. 0.06 m

    #     lift = self.object_pose_buf[:, 2] - self.init_states.objects["object"].root_state[:, 2]
    #     lifted = (lift > self.cfg.reward_lift_object_z).float()  # e.g. 0.10 m

    #     reward = self.see_flag_float * close * lifted * self.cfg.reward_lift_object_z  # e.g. 10.0
    #     return reward

    def _reward_wrist_lower_than_table(self, tensor_state: TensorState, robot_name: str, cfg: BaseTableHumanoidTaskCfg):
        below_distance = torch.clamp(
            tensor_state.robots[robot_name].body_state[:, self.right_palm_index, 2]
            - self.cfg.objects[0].default_position[2],
            max=0.0,
        )
        return below_distance.squeeze(1) * (self.stage == 0)

    # def _reward_object_holding(self, tensor_state: TensorState, robot_name: str, cfg: BaseTableHumanoidTaskCfg):
    def _reward_success(self, tensor_state: TensorState, robot_name: str, cfg: BaseTableHumanoidTaskCfg):
        return self.success.float()

    def _update_curriculum(self):
        current_iteration = int(self.common_step_counter / self.cfg.ppo_cfg.num_steps_per_env)
        self._update_obj_material()
        if (self.common_step_counter % self.cfg.ppo_cfg.num_steps_per_env) == 0:
            self._update_curriculum_object_yaw_range(current_iteration)
            self._update_curriculum_object_mass(current_iteration)

    def _update_obj_material(self):
        if (
            self.cfg.randomize_obj_material
            and self.common_step_counter % self.cfg.update_obj_material_step_interval == 0
        ):
            self.env.randomize_obj_material(list(range(self.num_envs)), self.obj)
            log.info("Updated object material")

    def _update_curriculum_object_yaw_range(self, current_iteration):
        if self.cfg.curriculum_object_yaw:
            # Only check and log once per 100 iterations, and only at the exact iteration boundary
            # if (self.common_step_counter % self.cfg.ppo_cfg.num_steps_per_env) == 0:
            # reward = self.episode_sums["pixel_norm_at_object"].mean()
            # Always update last_reward to track current performance
            # reward_improvement_ratio = (reward - self.last_reward) / (self.last_reward + 1e-8)
            # self.last_reward = reward

            # iterations_since_last_update = current_iteration - (
            #     self.last_curriculum_update_step / self.cfg.ppo_cfg.num_steps_per_env
            # )

            if self._ema_reward > self.cfg.ema_reward_threshold:
                if self.curriculum_object_yaw_range < self.cfg.randomize_object_yaw_range:
                    self._ema_reward = 0
                    self.curriculum_object_yaw_range += self.cfg.randomize_object_yaw_range * 0.05
                    self.last_curriculum_update_step = self.common_step_counter
                else:
                    pass  # move it to the wandb
            else:
                pass

    def _update_curriculum_object_mass(self, current_iteration):
        if self.cfg.curriculum_object_mass_flag and self.mass_curriculum_trigger:
            # set the upper bound of mass, to encourage contact
            if current_iteration == 0:
                mass = self.cfg.curriculum_object_mass_range[1] * torch.ones((self.num_envs, 1), device="cpu")
                log.info(f"UPDATE curriculum_object_mass: {self.cfg.curriculum_object_mass_range[1]}")
                self.obj_randomizer.set_mass("object", mass, env_ids=list(range(self.num_envs)))
                self.extra_buf["episode_metrics"]["curriculum_obj_mass"] = mass[0][0].item()

            elif current_iteration > self.end_mass_curriculum_iter:
                return

            elif current_iteration > self.start_mass_curriculum_iter:
                # linearly decrease to 0.05
                mass = self.cfg.curriculum_object_mass_range[1] + (
                    self.cfg.curriculum_object_mass_range[1] - self.cfg.curriculum_object_mass_range[0]
                ) * (self.start_mass_curriculum_iter - current_iteration) / (
                    self.cfg.curriculum_object_mass_update_interval
                )

                log.info(f"UPDATE curriculum_object_mass: {mass}")
                # randomize around the mass
                mass = mass + (torch.rand((self.num_envs, 1), device="cpu") - 0.5) * 0.05
                self.obj_randomizer.set_mass("object", mass, env_ids=list(range(self.num_envs)))
                self.extra_buf["episode_metrics"]["curriculum_obj_mass"] = mass.mean().item()

    def _get_reward_weights_for_phase(self, task_cfg: BaseTableHumanoidTaskCfg):
        """Get reward weights based on current phase."""
        if self.phase == 0:
            return task_cfg.reward_weights_phase0
        elif self.phase == 1:
            return task_cfg.reward_weights_phase1
        elif self.phase == 2:
            return task_cfg.reward_weights_phase2
        else:
            raise ValueError(f"Invalid phase: {self.phase}")

    def _setup_reward_functions_from_weights(self, reward_weights: dict):
        """Setup reward scales and functions from reward weights dictionary."""
        # Process reward scales
        self.reward_scales = dict(reward_weights)
        for key in list(self.reward_scales.keys()):
            scale = self.reward_scales[key]
            if scale == 0:
                self.reward_scales.pop(key)
            else:
                self.reward_scales[key] *= self.dt

        # Setup reward functions
        self.reward_functions = []
        self.reward_names = []
        for name, scale in self.reward_scales.items():
            if name == "termination":
                continue
            self.reward_names.append(name)
            method_name = "_reward_" + name
            fn = getattr(self, method_name, None)
            if fn is None or not callable(fn):
                raise KeyError(f"No reward function named '{method_name}' on {self.__class__.__name__}")
            self.reward_functions.append(fn)

    def _prepare_reward_function(self, task: BaseTableHumanoidTaskCfg):
        """Override to use phase-specific reward weights."""
        reward_weights = self._get_reward_weights_for_phase(task)
        self._setup_reward_functions_from_weights(reward_weights)

        # Initialize episode_sums and episode_metrics
        self.episode_sums = {
            name: torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
            for name in self.reward_scales.keys()
        }
        self.episode_metrics = {}

    def _update_reward_functions_for_phase(self):
        """Update reward functions when phase changes."""
        reward_weights = self._get_reward_weights_for_phase(self.cfg)
        self._setup_reward_functions_from_weights(reward_weights)

        # Update episode_sums to include new reward terms (preserve existing ones)
        new_episode_sums = {}
        for name in self.reward_scales.keys():
            if name in self.episode_sums:
                new_episode_sums[name] = self.episode_sums[name]
            else:
                new_episode_sums[name] = torch.zeros(
                    self.num_envs, dtype=torch.float, device=self.device, requires_grad=False
                )
        self.episode_sums = new_episode_sums

        log.info(f"Updated reward functions for phase {self.phase}")

    # def _reward_look_at_object(self, tensor_state: TensorState, robot_name: str, cfg: BaseTableHumanoidTaskCfg):
    #     """Reward for looking at the object."""
    #     # 获取相机的世界坐标位置 (num_envs, 3)
    #     camera_pos = self.camera_pos_w

    #     # 获取立方体的世界坐标位置 (num_envs, 3)
    #     object_pos = tensor_state.objects["object"].root_state[:, :3]

    #     # 计算从相机到立方体的方向向量 (num_envs, 3)
    #     direction_vec = object_pos - camera_pos
    #     direction_vec = direction_vec / (torch.norm(direction_vec, dim=1, keepdim=True) + 1e-8)  # 归一化

    #     # 获取相机的朝向向量 (num_envs, 3)
    #     # 相机的朝向通常是+X方向（根据CameraState的注释）
    #     # camera_quat = tensor_state.cameras[self.cfg.cameras[0].name].quat_world  # (num_envs, 4) - (w, x, y, z)
    #     camera_quat = self.camera_quat_w  # (num_envs, 4) - (w, x, y, z)
    #     # 将相机的+X轴方向向量转换到世界坐标系
    #     camera_forward = torch.tensor([1.0, 0.0, 0.0], device=self.device).expand(self.num_envs, 3)
    #     camera_vec = quat_apply(camera_quat, camera_forward)  # 应用四元数旋转

    #     # 计算两个向量的点积，得到相似度 (num_envs,)
    #     dot_product = torch.sum(direction_vec * camera_vec, dim=1)

    #     # 将点积转换为奖励 - 当相机完全对准立方体时点积为1，奖励最大
    #     # 使用平滑的奖励函数：当dot_product接近1时奖励接近1
    #     reward = torch.clamp(dot_product, min=0.0)  # 只考虑正向的对准

    #     # if self.env._render_viewport:
    #     #     self._update_marker_viz(camera_pos, camera_quat, direction_vec)

    #     return reward
