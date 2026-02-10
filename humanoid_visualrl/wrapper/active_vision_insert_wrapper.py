"""A wrapper for fixed upper body and use cnn inside the policy class."""

from __future__ import annotations

import torch

from humanoid_visualrl.wrapper.active_vision_cube_wrapper import ActiveVisionWrapper as ActiveVisionCubeWrapper
from metasim.task.registry import register_task
from metasim.utils.math import quat_from_euler_xyz
import math
from metasim.types import TensorState
import numpy as np
import os


@register_task("active_vision_insertion")
class ActiveVisionWrapper(ActiveVisionCubeWrapper):
    """Wraps Metasim environments to be compatible with rsl_rl OnPolicyRunner.

    Note that rsl_rl is designed for parallel training fully on GPU, with robust support for Isaac Gym and Isaac Lab.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Initialize box center positions for insertion task
        # self.box_center_x = self.cfg.init_states[0]["objects"]["insertion_female_box"]["pos"][0]
        # self.box_center_y = self.cfg.init_states[0]["objects"]["insertion_female_box"]["pos"][1]

        self.yaw_offset = math.pi / 3

        self.recorded_cube_pos = torch.tensor([0.488, 0.142, 0.5650, 0.9677, 0.0, 0.0, -0.2522], device=self.device)

        npz_path = "ik_curriculum_data_merged.npz"
        # 0:49  embed cube into the box data

        self.curriculum_object_yaw_range = self.cfg.randomize_object_yaw_range

        # 50: 00 random ik data
        if os.path.exists(npz_path):
            data = np.load(npz_path)
            recorded_qpos_raw = torch.tensor(data["qpos"], device=self.device, requires_grad=False)  # [400, num_joints]
            # from isaacsim deleted joint version to none deleted version
            self.recorded_qpos_raw = recorded_qpos_raw[:, self.env._none_static_joint_idx_reindexed]
            self.recorded_cube_pos_raw = torch.tensor(data["cube_pos"], device=self.device, requires_grad=False)  # [400, 7]

            # 0: 100
            # 100: 200
            # first inverse curriculum use embed cube into the box data and double it
            self.recorded_qpos = self.recorded_qpos_raw[100:200, :].repeat(2, 1)
            self.recorded_cube_pos = self.recorded_cube_pos_raw[100:200, :].repeat(2, 1)

            # after reaching, directly use the recorded
            if self.phase == 2:
                # random in the air data:
                recorded_qpos_in_the_air = self.recorded_qpos_raw[:100, :].repeat(2, 1).clone()
                recorded_cube_pos_in_the_air = self.recorded_cube_pos_raw[:100, :].repeat(2, 1).clone()

                # if self.num_envs > 100:


                # ensure

                # fill all init_states with recorded_qpos


                self.init_states.robots["vega"].joint_pos[:, :] = recorded_qpos_in_the_air[: self.num_envs, :].clone()
                self.init_states.objects["object"].root_state[:, :7] = recorded_cube_pos_in_the_air[
                    : self.num_envs, :
                ].clone()



            # # Filter out data where sqrt(x^2 + y^2) < threshold
            # threshold = self.cfg.init_states[0]["objects"]["object"]["pos"][0]
            # cube_xy_dist = torch.sqrt(self.recorded_cube_pos[:, 0] ** 2 + self.recorded_cube_pos[:, 1] ** 2)
            # mask = cube_xy_dist >= threshold
            # original_count = len(self.recorded_qpos)
            # self.recorded_qpos = self.recorded_qpos[mask]
            # self.recorded_cube_pos = self.recorded_cube_pos[mask]

    def _pre_reset_hook(self, env_ids=None):
        # return
        # if self.cfg.randomize_material:
        #     self.domain_randomization_helper.randomization(env_ids=env_ids, step_count=self.common_step_counter)

        if self.cfg.randomization:
            # yaw: 物体相对于机器人的方位角（用于计算物体位置）
            if self.phase == 0 or self.phase == 1 :
                object_x = self.cfg.randomize_object_x + (torch.rand(len(env_ids), device=self.device) - 0.5) * self.cfg.randomize_object_range
                object_y = self.cfg.randomize_object_y + (torch.rand(len(env_ids), device=self.device) - 0.5) * self.cfg.randomize_object_range
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
            elif self.phase == 2:
                # not need to randomize object position since reset
                pass

            elif self.phase == 3:
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

                        # Directly copy recorded cube positions to object init_state
                        self.init_states.objects["object"].root_state[stretch_env_ids, :7] = (
                            self.recorded_cube_pos[selected_indices, :]
                        )

                        self.init_states.robots["vega"].joint_pos[stretch_env_ids, :]=self.recorded_qpos[selected_indices, :]

    def success_checker(self, object_pose_buf: torch.Tensor):
        with torch.no_grad():
        
            if self.phase == 3:
                # check if the object is in the hand
                object_x_thres = object_pose_buf[:, 1] > self.cfg.y_threshold
                object_z_thres = object_pose_buf[:, 2] > self.cfg.z_threshold
                success = object_x_thres & object_z_thres
                return success
            elif self.phase == 2:
                # hight enough and in the hand
                # object_x_thres = object_pose_buf[:, 0] > 0.50
                object_z_thres = object_pose_buf[:, 2] > self.cfg.z_threshold
                success = object_z_thres
                # if success.any():
                #     print(success)
                return success
            elif self.phase == 1:
                return torch.zeros(self.num_envs, device=self.device, dtype=torch.bool, requires_grad=False)

            else:
                # num envs false tensor
                return torch.zeros(self.num_envs, device=self.device, dtype=torch.bool, requires_grad=False)

    def _check_reset(self):
        terminate = self.cfg.init_states[0]["objects"]["object"]["pos"][2] - self.object_pose_buf[:, 2] > 0.1
        too_far = torch.norm(self.object_pose_buf[:, :2], dim=1) > (self.cfg.randomize_object_radius + 0.13)
        
        success = self.success_checker(self.object_pose_buf)
        failure = self.timeout_buf | terminate | too_far
        # set self.success to False if failure
        self.success[failure] = False
        # set self.success to True if success
        self.success[success] = True

        success_rate = self.success.float().mean()
        self.extra_buf["episode_metrics"]["success"] = success_rate

        # when success rate exceed 0.7, the robot have learn to insert. now to learn random ik data to lift
        # if success_rate > 0.35:
        #     self.recorded_qpos = self.recorded_qpos_raw
        #     self.recorded_cube_pos = self.recorded_cube_pos_raw

        self.reset_buf = failure | success
        return self.reset_buf

    def _reward_lift_object(self, tensor_state: TensorState, robot_name: str, cfg):
        """Stage 1 reward: lifting/holding the cube before it reaches z_threshold."""
        # stage 1: cube not yet lifted above threshold
        stage1_mask = (self.object_pose_buf[:, 2] <= self.cfg.z_threshold).float()

        # Same shaping as in the base cube wrapper, but gated by stage1_mask
        dist = torch.square(self.object_pose_buf[:, 2] - self.cfg.reward_lift_object_z)
        base_reward = self.see_flag_float * (
            torch.exp(-self.cfg.reward_lift_object_exp_shapeness * dist) - self.lift_offset
        )
        return base_reward * stage1_mask

    def _reward_success(self, tensor_state: TensorState, robot_name: str, cfg):
        """Stage 2 reward in phase 2: success only after cube z is over z_threshold."""
        stage2_mask = (self.object_pose_buf[:, 2] > self.cfg.z_threshold).float()
        return self.success.float() * stage2_mask

    def _get_reward_weights_for_phase(self, task_cfg):
        """Get reward weights based on current phase."""
        if self.phase == 0:
            return task_cfg.reward_weights_phase0
        elif self.phase == 1:
            return task_cfg.reward_weights_phase1
        elif self.phase == 2:
            return task_cfg.reward_weights_phase2
        elif self.phase == 3:
            return task_cfg.reward_weights_phase3
        else:
            raise ValueError(f"Invalid phase: {self.phase}")
