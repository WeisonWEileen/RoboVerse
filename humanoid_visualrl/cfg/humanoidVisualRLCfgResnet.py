from __future__ import annotations

"""Base class for legged-gym style legged-robot tasks."""

from dataclasses import MISSING
from typing import Callable

import torch

from metasim.scenario.robot import RobotCfg
from metasim.scenario.simulator_params import SimParamCfg
from metasim.types import TensorState
from metasim.utils import configclass
from humanoid_visualrl.cfg.humanoidVisualRLCfg import BaseTableHumanoidTaskCfg
from metasim.scenario.cameras import PinholeCameraCfg

@configclass
class HumanoidVisualRLCfgResnet(BaseTableHumanoidTaskCfg):
    """Configuration for the humanoid visual RL task with ResNet."""

    visual_feature_dim: int = 512
    task_name: str = "walking_resnet"

    camera = PinholeCameraCfg(
            name="camera_first_person",
            width=64,
            height=48,
            pos=(1.5, -1.5, 1.5),
            look_at=(0.0, 0.0, 0.0),
            mount_to="g1",
            mount_link="torso_link",
            mount_pos=(0.1, 0.0, 0.9),
            #     quat_xyzw = R.from_euler("xyz", [0, 60, 0], degrees=True).as_quat()
    # quat = (quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2])  # 
            mount_quat=(0.8660254037844387, 0.0, 0.49999999999999994, 0.0),
        )   
    
    logger = None
    # logger = 'wandb'
    

    def __post_init__(self):
        super().__post_init__()
        self.task_name = "walking_resnet"

        self.num_single_obs = 3 * self.num_actions + 6 + self.command_dim + self.visual_feature_dim
        self.num_observations = int(self.frame_stack * self.num_single_obs)
        self.single_num_privileged_obs = 4 * self.num_actions + 23 + self.visual_feature_dim
        self.num_privileged_obs = int(self.c_frame_stack * self.single_num_privileged_obs)