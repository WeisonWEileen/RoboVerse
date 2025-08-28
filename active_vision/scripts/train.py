"""This script is used to test the static scene."""

from __future__ import annotations

from typing import Literal

import rootutils
import torch
import tyro
from loguru import logger as log
from rich.logging import RichHandler

rootutils.setup_root(__file__, pythonpath=True)
log.configure(handlers=[{"sink": RichHandler(), "format": "{message}"}])

# from metasim.scenario.cameras import PinholeCameraCfg
from metasim.scenario.lights import DiskLightCfg, DistantLightCfg, DomeLightCfg
from metasim.scenario.scenario import ScenarioCfg
from metasim.utils import configclass



decimation = 3

from rsl_rl.runners.on_policy_runner import OnPolicyRunner

from active_vision.cfg.scenario_cfg import BaseTableHumanoidTaskCfg
from active_vision.wrapper.active_vision_wrapper import ActiveVisionWrapper

from metasim.scenario.objects import PrimitiveCubeCfg
from metasim.constants import PhysicStateType

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
            intensity=100.0,  # Moderate ambient lighting
            color=(0.85, 0.9, 1.0),  # Slightly blue sky color
        ),
        # Sun light - main directional light
        DistantLightCfg(
            intensity=100.0,  # Strong sunlight
            polar=35.0,  # Sun at 35° elevation (natural angle)
            azimuth=60.0,  # From the northeast
            color=(1.0, 0.98, 0.95),  # Slightly warm sunlight
        ),
        # Soft area light for subtle fill
        DiskLightCfg(
            intensity=100.0,
            radius=1.5,  # Large disk for soft light
            pos=(2.0, -2.0, 4.0),  # Side fill light
            rot=(0.7071, 0.7071, 0.0, 0.0),  # Angled towards scene
            color=(0.95, 0.95, 1.0),
        ),
    ]

    # add cameras
    # scenario.cameras = [PinholeCameraCfg(width=1024, height=1024, pos=(1.5, -1.5, 1.5), look_at=(0.0, 0.0, 0.0))]
    scenario.cameras = []
    # add objects
    scenario.objects = [
        PrimitiveCubeCfg(
            name="cube",
            size=(0.5, 0.5, 0.05),
            color=[0.9, 0.7, 0.7],
            physics=PhysicStateType.RIGIDBODY,
            fix_base_link=True,
            default_position=(0.3, 0.0, 0.8),
        ),
    ]

    task_cfg = BaseTableHumanoidTaskCfg()

    # task assign and override
    scenario.sim_params = task_cfg.sim_params
    scenario.decimation = task_cfg.decimation
    scenario.render_interval = 4
    scenario.task = task_cfg
    scenario.env_spacing = task_cfg.env_spacing

    log.info(f"Using simulator: {args.sim}")
    env = ActiveVisionWrapper(scenario)
    device = torch.device("cuda")
    log_dir = "outputs/active_vision"
    use_wandb = False
    learning_iterations = 1000
    ppo_runner = OnPolicyRunner(
        env=env,
        train_cfg=env.train_cfg,
        device=device,
        log_dir=log_dir,
    )
    ppo_runner.learn(num_learning_iterations=learning_iterations)
