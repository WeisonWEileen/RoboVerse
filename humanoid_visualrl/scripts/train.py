"""This script is used to test the static scene."""

from __future__ import annotations


import rootutils
import torch
from metasim.scenario.lights import DomeLightCfg

from loguru import logger as log
from rich.logging import RichHandler

rootutils.setup_root(__file__, pythonpath=True)
log.configure(handlers=[{"sink": RichHandler(), "format": "{message}"}])
from metasim.scenario.scenario import ScenarioCfg
from humanoid_visualrl.actor_critic.on_policy_runner import OnPolicyRunner
from humanoid_visualrl.utils.utils import get_log_dir, get_args, get_load_path, dump_instance_file

import os
from metasim.task.registry import get_task_class, get_task_cfg_class

if __name__ == "__main__":
    args = get_args()
    # task_cfg, cfg_file_path = get_cfg_cls(args)
    task_cfg_cls = get_task_cfg_class(args.task)

    task_cfg = task_cfg_cls(finetune=args.resume)
    if args.resume:
        log.info(f"Finetuning Model from: {args.load_run}")

    # initialize scenario
    scenario = ScenarioCfg(
        robots=[task_cfg.robot],
        simulator=args.sim,
        headless=args.headless,
        num_envs=args.num_envs,
    )
    scenario.lights = [
        DomeLightCfg(
            intensity=100.0,
            color=(0.85, 0.9, 1.0),
        ),
    ]

    scenario.cameras = task_cfg.cameras
    scenario.objects = task_cfg.objects

    # task assign and override
    scenario.sim_params = task_cfg.sim_params
    scenario.decimation = task_cfg.decimation
    scenario.render_interval = scenario.decimation
    scenario.task = task_cfg
    scenario.env_spacing = task_cfg.env_spacing

    if args.debug:
        scenario.env_spacing = 5
        scenario.env_spacing = 5

    log.info(f"Using simulator: {args.sim}")
    env_cls = get_task_class(args.task)

    if task_cfg.use_vision:
        env = env_cls(scenario, enable_opencv_display=args.enable_opencv_display)
    else:
        env = env_cls(scenario)
    device = torch.device("cuda")
    log_dir, now = get_log_dir(args, scenario)
    
    if args.debug:
        # do not log, faster reset
        log_dir = None
        task_cfg.num_steps_per_env = 48
    else:
        dump_instance_file(task_cfg, os.path.join(log_dir, "cfg.py"))
        dump_instance_file(env, os.path.join(log_dir, "env.py"))

    if args.wandb and not args.debug:
        env.train_cfg["logger"] = "wandb"

    ppo_runner = OnPolicyRunner(
        env=env,
        train_cfg=env.train_cfg,
        device=device,
        log_dir=log_dir,
        use_vision=task_cfg.use_vision,
    )

    if args.resume:
        resume_path = get_load_path(args)
        if not os.path.exists(resume_path):
            raise FileNotFoundError(f"Resume path {resume_path} does not exist")
        log.info(f"Loading model from: {resume_path}")
        ppo_runner.load(resume_path)
    ppo_runner.learn(num_learning_iterations=args.num_learning_iterations   ,     run_name=f"{args.run_name}_{now}")
