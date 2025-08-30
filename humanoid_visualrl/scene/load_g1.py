"""This script is used to test the static scene."""

from __future__ import annotations

import os
from typing import Literal

import rootutils
import torch
import tyro
from loguru import logger as log
from rich.logging import RichHandler

rootutils.setup_root(__file__, pythonpath=True)
log.configure(handlers=[{"sink": RichHandler(), "format": "{message}"}])
from get_started.utils import ObsSaver
from metasim.constants import SimType
from metasim.scenario.cameras import PinholeCameraCfg
from metasim.scenario.lights import DiskLightCfg, DistantLightCfg, DomeLightCfg
from metasim.scenario.scenario import ScenarioCfg
from metasim.utils import configclass
from metasim.utils.setup_util import get_sim_handler_class

decimation = 3

if __name__ == "__main__":

    @configclass
    class Args:
        """Arguments for the static scene."""

        robot: str = "g1_static"

        ## Handlers
        sim: Literal["isaacsim"] = "isaacsim"

        ## Others
        num_envs: int = 1
        headless: bool = False

        def __post_init__(self):
            """Post-initialization configuration."""
            log.info(f"Args: {self}")

    args = tyro.cli(Args)

    # initialize scenario
    scenario = ScenarioCfg(
        robots=[args.robot],
        simulator=args.sim,
        headless=args.headless,
        num_envs=args.num_envs,
    )
    scenario.lights = [
        # Sky dome light - provides soft ambient lighting from all directions
        DomeLightCfg(
            intensity=800.0,  # Moderate ambient lighting
            color=(0.85, 0.9, 1.0),  # Slightly blue sky color
        ),
        # Sun light - main directional light
        DistantLightCfg(
            intensity=1200.0,  # Strong sunlight
            polar=35.0,  # Sun at 35° elevation (natural angle)
            azimuth=60.0,  # From the northeast
            color=(1.0, 0.98, 0.95),  # Slightly warm sunlight
        ),
        # Soft area light for subtle fill
        DiskLightCfg(
            intensity=300.0,
            radius=1.5,  # Large disk for soft light
            pos=(2.0, -2.0, 4.0),  # Side fill light
            rot=(0.7071, 0.7071, 0.0, 0.0),  # Angled towards scene
            color=(0.95, 0.95, 1.0),
        ),
    ]

    # add cameras
    scenario.cameras = [PinholeCameraCfg(width=1024, height=1024, pos=(1.5, -1.5, 1.5), look_at=(0.0, 0.0, 0.0))]

    # add objects
    scenario.objects = []

    log.info(f"Using simulator: {args.sim}")
    env_class = get_sim_handler_class(SimType(args.sim))
    env = env_class(scenario)

    init_states = [
        {
            "objects": {},
            "robots": {
                "g1": {
                    "pos": torch.tensor([0.0, 0.0, 0.0]),
                    "rot": torch.tensor([1.0, 0.0, 0.0, 0.0]),
                    "dof_pos": {
                        "waist_yaw": 0.0,
                        "left_shoulder_pitch": 0.0,
                        "left_shoulder_roll": 0.0,
                        "left_shoulder_yaw": 0.0,
                        "left_elbow": 0.0,
                        "right_shoulder_pitch": 0.0,
                        "right_shoulder_roll": 0.0,
                        "right_shoulder_yaw": 0.0,
                        "right_elbow": 0.0,
                    },
                },
            },
        }
    ]
    env.launch()
    env.set_states(init_states * scenario.num_envs)
    os.makedirs("get_started/output", exist_ok=True)

    obs = env.get_states(mode="dict")
    ## Main loop
    obs_saver = ObsSaver(video_path=f"get_started/output/1_move_robot_{args.sim}.mp4")
    obs_saver.add(obs)

    num_dof = len(env.get_joint_names(env.robots[0].name))

    lower_body_joints_names = {
        "left_hip_pitch",
        "left_hip_roll",
        "left_hip_yaw",
        "left_knee",
        "left_ankle_pitch",
        "left_ankle_roll",
        "right_hip_pitch",
        "right_hip_roll",
        "right_hip_yaw",
        "right_knee",
        "right_ankle_pitch",
        "right_ankle_roll",
        "waist_yaw",
    }
    sorted_joint_names = env.get_joint_names(env.robots[0].name, sort=True)
    # lower_body_joints_indexes = [sorted_joint_names.index(joint_name) for joint_name in lower_body_joints_names]

    step = 0
    robot = scenario.robots[0]

    while True:
        log.debug(f"Step {step}")

        actions = torch.rand([args.num_envs, num_dof], device="cuda")
        # for joint_name in lower_body_joints_names:
        #     idx = sorted_joint_names.index(joint_name)
        #     actions[:, idx] = init_states[0]["robots"]["g1"]["dof_pos"][joint_name]
        env.set_dof_targets(actions)
        for i in range(decimation):
            env.simulate()
        obs = env.get_states(mode="dict")
        step += 1
