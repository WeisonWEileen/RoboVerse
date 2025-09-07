"""This script is used to test the static scene."""

from __future__ import annotations

from typing import Literal

import rootutils
import torch
from metasim.scenario.lights import DiskLightCfg, DistantLightCfg, DomeLightCfg

import tyro
from metasim.scenario.cameras import PinholeCameraCfg

from loguru import logger as log
from rich.logging import RichHandler

rootutils.setup_root(__file__, pythonpath=True)
log.configure(handlers=[{"sink": RichHandler(), "format": "{message}"}])
from metasim.scenario.scenario import ScenarioCfg
from humanoid_visualrl.actor_critic.on_policy_runner import OnPolicyRunner
from humanoid_visualrl.utils.utils import get_log_dir, get_cfg_cls, get_env_wrapper_cls, get_args, get_load_path

import shutil
import os


if __name__ == "__main__":
    args = get_args()

    if args.use_resnet and args.use_vision:
        raise ValueError("use_resnet and use_vision cannot be True at the same time")

    # initialize scenario
    scenario = ScenarioCfg(
        robots=[args.robot],
        simulator=args.sim,
        headless=args.headless,
        num_envs=args.num_envs,
    )
    scenario.lights = [
        DomeLightCfg(
            intensity=100.0,
            color=(0.85, 0.9, 1.0),
        ),
        DistantLightCfg(
            intensity=100.0,
            polar=35.0,
            azimuth=60.0,
            color=(1.0, 0.98, 0.95),
        ),
        DiskLightCfg(
            intensity=100.0,
            radius=1.5,
            pos=(2.0, -2.0, 4.0),
            rot=(0.7071, 0.7071, 0.0, 0.0),
            color=(0.95, 0.95, 1.0),
        ),
    ]

    # look different task cfg
    task_cfg, cfg_file_path = get_cfg_cls(args)

    if args.use_vision or args.use_resnet or args.use_fixed_gazing:
        scenario.cameras = [task_cfg.camera]
    else:
        scenario.cameras = []

    # add objects
    scenario.objects = task_cfg.objects

    # task assign and override
    scenario.sim_params = task_cfg.sim_params
    scenario.decimation = task_cfg.decimation
    scenario.render_interval = scenario.decimation
    scenario.task = task_cfg
    scenario.env_spacing = task_cfg.env_spacing

    log.info(f"Using simulator: {args.sim}")

    env, env_file_path = get_env_wrapper_cls(args, scenario)
    device = torch.device("cuda")
    log_dir = get_log_dir(args, scenario)

    # copy cfg and env file to log_dir
    shutil.copy(cfg_file_path, os.path.join(log_dir, "cfg.py"))
    shutil.copy(env_file_path, os.path.join(log_dir, "env.py"))

    if args.wandb:
        env.train_cfg["logger"] = "wandb"
    ppo_runner = OnPolicyRunner(
        env=env,
        train_cfg=env.train_cfg,
        device=device,
        log_dir=log_dir,
        use_vision=task_cfg.use_vision,
    )

    if args.resume:
        resume_path = get_load_path(args, scenario)
        if not os.path.exists(resume_path):
            raise FileNotFoundError(f"Resume path {resume_path} does not exist")
        log.info(f"Loading model from: {resume_path}")
        ppo_runner.load(resume_path)
    ppo_runner.learn(num_learning_iterations=args.num_learning_iterations)
