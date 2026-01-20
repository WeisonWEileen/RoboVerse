"""A wrapper for fixed upper body and use cnn inside the policy class."""

from __future__ import annotations

import torch

from humanoid_visualrl.wrapper.active_vision_cube_wrapper import ActiveVisionWrapper as ActiveVisionCubeWrapper
from metasim.task.registry import register_task
from metasim.utils.math import quat_from_euler_xyz
import math
from metasim.types import TensorState


@register_task("active_vision_insertion")
class ActiveVisionWrapper(ActiveVisionCubeWrapper):
    """Wraps Metasim environments to be compatible with rsl_rl OnPolicyRunner.

    Note that rsl_rl is designed for parallel training fully on GPU, with robust support for Isaac Gym and Isaac Lab.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Initialize box center positions for insertion task
        self.box_center_x = self.cfg.init_states[0]["objects"]["insertion_female_box"]["pos"][0]
        self.box_center_y = self.cfg.init_states[0]["objects"]["insertion_female_box"]["pos"][1]

        self.yaw_offset = math.pi / 3

        self.recorded_cube_pos = torch.tensor([0.488, 0.142, 0.5650, 0.9677, 0.0, 0.0, -0.2522], device=self.device)

    # def _pre_reset_hook(self, env_ids=None):
    #     super()._pre_reset_hook(env_ids=env_ids)

    #     # if self.cfg.randomization:
    #     #     # randomize box xy position
    #     #     noise_xy = torch.empty(len(env_ids), 2, device=self.device).uniform_(
    #     #         -self.cfg.randomize_box_xy_range_scale, self.cfg.randomize_box_xy_range_scale
    #     #     )
    #     #     self.init_states.objects["insertion_female_box"].root_state[env_ids, 0:2] = (
    #     #         torch.tensor([self.box_center_x, self.box_center_y], device=self.device)
    #     #         + noise_xy * self.cfg.occlude_cube_yaw_range
    #     #     )

    #     #     # add randomize around the yaw in the initialize state of the box
    #     #     box_rotation_yaw = torch.empty(len(env_ids), device=self.device).uniform_(
    #     #         -self.cfg.randomize_box_rot_range_scale, self.cfg.randomize_box_rot_range_scale
    #     #     )

    #     #     # yaw_offset + random yaw
    #     #     self.init_states.objects["insertion_female_box"].root_state[env_ids, 3:7] = quat_from_euler_xyz(
    #     #         torch.zeros(len(env_ids), device=self.device),
    #     #         torch.zeros(len(env_ids), device=self.device),
    #     #         self.yaw_offset + box_rotation_yaw,
    #     #     )

    def _pre_reset_hook(self, env_ids=None):
        # return
        # if self.cfg.randomize_material:
        #     self.domain_randomization_helper.randomization(env_ids=env_ids, step_count=self.common_step_counter)

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

            object_x = torch.cos(object_relative_yaw) * radius
            object_y = torch.sin(object_relative_yaw) * radius
            self.init_states.objects["object"].root_state[env_ids, 0] = object_x
            self.init_states.objects["object"].root_state[env_ids, 1] = object_y
            # self.done_buf[env_ids] = False
            # randomize object's own rotation yaw (物体自身的旋转角度)
            object_rotation_yaw = 2 * (torch.rand(len(env_ids), device=self.device) - 0.5) * 3.14
            quat = quat_from_euler_xyz(
                torch.zeros(len(env_ids), device=self.device),
                torch.zeros(len(env_ids), device=self.device),
                object_rotation_yaw,
            )
            self.init_states.objects["object"].root_state[env_ids, 3:7] = quat

            if self.cfg.phase == 2:
                # only those < num_envs//2 are in stretch pose
                # Convert env_ids to tensor for comparison
                env_ids_tensor = torch.tensor(env_ids, device=self.device, dtype=torch.long)
                mask = env_ids_tensor < self.num_envs // 2
                stretch_env_ids = env_ids_tensor[mask]

                # align robot base yaw joint to face the object
                if len(stretch_env_ids) > 0:
                    # If we have recorded poses, randomly sample from them
                    if self.recorded_qpos is not None and self.recorded_cube_pos is not None:
                        num_stretch = len(stretch_env_ids)
                        # Randomly select indices from recorded poses
                        selected_indices = torch.randint(0, len(self.recorded_qpos), (num_stretch,), device=self.device)

                        # Set joint positions from recorded data
                        # self.init_states.robots["vega"].joint_pos[stretch_env_ids, :][:, self.actuated_local_index] = (
                        #     self.recorded_qpos[selected_indices, :]
                        # )

                        # Override base_joint_index with object_relative_yaw
                        # self.init_states.robots["vega"].joint_pos[stretch_env_ids, self.base_joint_index] = (
                        #     object_relative_yaw[mask] - 0.3
                        # )

                        # Directly copy recorded cube positions to object init_state
                        self.init_states.objects["object"].root_state[stretch_env_ids, :7] = (
                            self.recorded_cube_pos.repeat(num_stretch, 1)
                        )

    def success_checker(self, object_pose_buf: torch.Tensor):
        if self.cfg.phase == 2:
            # check if the object is in the hand
            object_x_thres = object_pose_buf[:, 0] > 0.50
            object_z_thres = object_pose_buf[:, 2] > 0.55
            success = object_x_thres & object_z_thres
            return success
        else:
            # num envs false tensor
            return torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)

    def _check_reset(self):
        terminate = self.cfg.init_states[0]["objects"]["object"]["pos"][2] - self.object_pose_buf[:, 2] > 0.1
        too_far = torch.norm(self.object_pose_buf[:, :2], dim=1) > (self.cfg.randomize_object_radius + 0.13)
        self.success = self.success_checker(self.object_pose_buf)
        self.reset_buf = self.timeout_buf | terminate | too_far | self.success

    
    def _reward_success(self, tensor_state: TensorState, robot_name: str, cfg):
        # if self.cfg.phase == 2:
        return self.success.float()


    

