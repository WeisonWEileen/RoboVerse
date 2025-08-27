"""This script is used to test the static scene."""

from __future__ import annotations

from typing import Literal

import rootutils
import torch
import tyro
from metasim.scenario.cameras import PinholeCameraCfg

from loguru import logger as log
from rich.logging import RichHandler

rootutils.setup_root(__file__, pythonpath=True)
log.configure(handlers=[{"sink": RichHandler(), "format": "{message}"}])
from metasim.scenario.scenario import ScenarioCfg
from metasim.utils import configclass


from humanoid_visualrl.actor_critic.on_policy_runner import OnPolicyRunner


from humanoid_visualrl.utils.utils import get_log_dir

if __name__ == "__main__":

    @configclass
    class Args:
        """Arguments for the static scene."""

        robot: str = "g1"
        sim: Literal["isaacsim"] = "isaacsim"  # only support isaacsim
        num_envs: int = 1
        headless: bool = False
        num_learning_iterations: int = 10000
        enable_opencv_display: bool = False
        use_vision: bool = False
        use_resnet: bool = False
        use_reaching: bool = False

        def __post_init__(self):
            """Post-initialization configuration."""
            log.info(f"Args: {self}")

    args = tyro.cli(Args)

    if args.use_resnet and args.use_vision:
        raise ValueError("use_resnet and use_vision cannot be True at the same time")

    # initialize scenario
    scenario = ScenarioCfg(
        robots=[args.robot],
        simulator=args.sim,
        headless=args.headless,
        num_envs=args.num_envs,
    )
    scenario.lights = []

    # look different task cfg
    if args.use_vision:
        from humanoid_visualrl.cfg.humanoidVisualRLVisionCfg import BaseTableHumanoidTaskCfg
    elif args.use_resnet:
        from humanoid_visualrl.cfg.humanoidVisualRLCfgResnet import HumanoidVisualRLCfgResnet as BaseTableHumanoidTaskCfg
    elif args.use_reaching:
        from humanoid_visualrl.cfg.humanoidReaching import HumanoidReachingCfg as BaseTableHumanoidTaskCfg
    else:
        from humanoid_visualrl.cfg.humanoidVisualRLCfg import BaseTableHumanoidTaskCfg
    
    # if args.:
    #     task_cfg = BaseTableHumanoidTaskCfg()
    # else:

    task_cfg = BaseTableHumanoidTaskCfg()

    if args.use_vision or args.use_resnet:
        scenario.cameras = [task_cfg.camera]
    else:
        scenario.cameras = []

    # add objects
    scenario.objects = []

    # task assign and override
    scenario.sim_params = task_cfg.sim_params
    scenario.decimation = task_cfg.decimation
    scenario.render_interval = scenario.decimation
    scenario.task = task_cfg
    scenario.env_spacing = task_cfg.env_spacing

    log.info(f"Using simulator: {args.sim}")

    if args.use_resnet:
        from humanoid_visualrl.wrapper.walking_wrapper_resnet import WalkingWrapperResNet as TaskWrapper
        env = TaskWrapper(scenario, enable_opencv_display=args.enable_opencv_display)
    elif args.use_vision:
        from humanoid_visualrl.wrapper.walking_wrapper_cnn import WalkingWrapperCNN as TaskWrapper
        env = TaskWrapper(scenario, enable_opencv_display=args.enable_opencv_display)
    elif args.use_reaching:
        from humanoid_visualrl.wrapper.reaching_wrapper import ReachingWrapper as TaskWrapper
        env = TaskWrapper(scenario)
    else:
        from humanoid_visualrl.wrapper.walking_wrapper import WalkingWrapper as TaskWrapper
        env = TaskWrapper(scenario)

    device = torch.device("cuda")
    log_dir = get_log_dir(args, scenario)
    ppo_runner = OnPolicyRunner(
        env=env,
        train_cfg=env.train_cfg,
        device=device,
        log_dir=log_dir,
        use_vision=task_cfg.use_vision,
    )
    ppo_runner.learn(num_learning_iterations=args.num_learning_iterations)
