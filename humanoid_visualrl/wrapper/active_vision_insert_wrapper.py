"""A wrapper for fixed upper body and use cnn inside the policy class."""

from __future__ import annotations

import torch

from humanoid_visualrl.wrapper.active_vision_cube_wrapper import ActiveVisionWrapper as ActiveVisionCubeWrapper
from metasim.task.registry import register_task
from metasim.utils.math import quat_from_euler_xyz
import math

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

        self.yaw_offset = math.pi / 2


    def _pre_reset_hook(self, env_ids=None):
        super()._pre_reset_hook(env_ids=env_ids)

        if self.cfg.randomization:
            # randomize box xy position
            noise_xy = torch.empty(len(env_ids), 2, device=self.device).uniform_(
                -self.cfg.randomize_box_xy_range_scale, self.cfg.randomize_box_xy_range_scale
            )
            self.init_states.objects["insertion_female_box"].root_state[env_ids, 0:2] = (
                torch.tensor([self.box_center_x, self.box_center_y], device=self.device)
                + noise_xy * self.cfg.occlude_cube_yaw_range
            )

            # add randomize around the yaw in the initialize state of the box
            box_rotation_yaw = torch.empty(len(env_ids), device=self.device).uniform_(
                -self.cfg.randomize_box_rot_range_scale, self.cfg.randomize_box_rot_range_scale
            )


            # yaw_offset + random yaw
            self.init_states.objects["insertion_female_box"].root_state[env_ids, 3:7] = quat_from_euler_xyz(
                torch.zeros(len(env_ids), device=self.device),
                torch.zeros(len(env_ids), device=self.device),
                self.yaw_offset + box_rotation_yaw,
            )




