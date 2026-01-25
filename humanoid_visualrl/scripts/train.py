"""This script is used to test the static scene."""

from __future__ import annotations

import random

import numpy as np
import rootutils
import torch
from loguru import logger as log
from rich.logging import RichHandler

from metasim.scenario.lights import DomeLightCfg
from metasim.scenario.objects import PrimitiveCubeCfg

rootutils.setup_root(__file__, pythonpath=True)
log.configure(handlers=[{"sink": RichHandler(), "format": "{message}"}])
import os
import shutil

from humanoid_visualrl.actor_critic.on_policy_runner import OnPolicyRunner
from humanoid_visualrl.utils.utils import dump_instance_file, get_args, get_load_path, get_log_dir
from metasim.constants import PhysicStateType
from metasim.scenario.scenario import ScenarioCfg
from metasim.task.registry import get_task_cfg_class, get_task_class

import argparse
from isaaclab.app import AppLauncher
from metasim.sim.isaacsim.isaacsim import set_app_launcher_context

args = get_args()

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args_isaac = parser.parse_args([])
args_isaac.device = args.device
args_isaac.enable_cameras = True
args_isaac.headless = args.headless
app_launcher = AppLauncher(args_isaac)
# 设置全局上下文，这样 launch 函数就可以从上下文获取 app_launcher
set_app_launcher_context(app_launcher)

if __name__ == "__main__":
    # Set random seed for reproducibility
    if args.seed != -1:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        log.info(f"Random seed set to: {args.seed}")
    else:
        log.info("Using random seed (seed=-1)")
    assert args.opencv_render_env_idx < args.num_envs, "opencv_render_env_idx must be less than num_envs"

    assert args.actor_critic_class in [
        "use_vision",
        "use_rnn",
        "use_resnet",
        "use_rnn_foveated",
        "use_vit_rnn",
        "use_rnn_foveated_vit",
        "use_patchcnn_rnn",
        "use_rnn_cnn_ram",
    ], "Invalid actor critic class"
    # task_cfg, cfg_file_path = get_cfg_cls(args)
    task_cfg_cls = get_task_cfg_class(args.task)

    assert args.phase in [0, 1, 2], "Invalid phase"
    # cfg.phase = args.phase

    task_cfg = task_cfg_cls(
        finetune=args.resume,
        actor_critic_class=args.actor_critic_class,
        enable_grasp=args.enable_grasp,
        vision4times_slowdown=args.vision4times_slowdown,
        phase=args.phase,
    )

    if hasattr(task_cfg, "randomize_material"):
        task_cfg.randomize_material = args.randomize_material
    if hasattr(task_cfg, "occlude_cube"):
        task_cfg.occlude_cube = args.occlude_cube

    if args.occlude_cube:
        task_cfg.occlude_cube = True
        task_cfg.objects.append(
            PrimitiveCubeCfg(
                name="occlusion_cube",
                size=(0.10, 0.10, 0.2),
                color=[0.5, 0.5, 0.5],
                physics=PhysicStateType.RIGIDBODY,
                collision_enabled=True,
                fix_base_link=False,
                default_position=(0.3, 0.1, 0.93),
                mass=0.2,  # 增加质量以确保更好的物理行为
            ),
        )
        task_cfg.init_states[0]["objects"]["occlusion_cube"] = {
            "pos": torch.tensor([0.3, 0.1, 0.92]),
            "rot": torch.tensor([1.0, 0.0, 0.0, 0.0]),
        }
        task_cfg.filter_pairs.append((task_cfg.robot, "occlusion_cube"))
        task_cfg.filter_pairs.append(("object", "occlusion_cube"))
    # if not args.debug:
    #  assert args.num_envs == 64

    assert task_cfg.env_spacing > 4.9, "env_spacing must be greater than 5"
    if args.resume:
        log.info(f"Finetuning Model from: {args.load_run}")
    from metasim.utils.setup_util import get_robot

    # robot = get_robot(task_cfg.robot)

    # if args.task == "active_vision_insertion":
    #     robot.modified_joint_limits.update({"base_yaw_joint": (-0.5708, 0.5708)})

    # initialize scenario
    scenario = ScenarioCfg(
        robots=[task_cfg.robot],
        simulator=args.sim,
        headless=args.headless,
        num_envs=args.num_envs,
    )

    scenario.lights = [
        DomeLightCfg(
            intensity=1000.0,
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
    if hasattr(task_cfg, "filter_pairs"):
        scenario.filter_pairs = task_cfg.filter_pairs
    else:
        scenario.filter_pairs = []
    scenario.env_spacing = task_cfg.env_spacing
    scenario.device = args.device
    log_dir, now = get_log_dir(args, scenario)

    if args.debug:
        # do not log, faster reset
        log_dir = None
        task_cfg.max_episode_length_s = 1.5
        # scenario.num_envs = 1
        scenario.sim_params.num_threads = 1
        task_cfg.objects[1].mass = 0.1
        task_cfg.curriculum_object_mass_range = (
            task_cfg.curriculum_object_mass_range[0],
            task_cfg.curriculum_object_mass_range[0],
        )

    log.info(f"Using simulator: {args.sim}")
    env_cls = get_task_class(args.task)

    if task_cfg.use_vision:
        env = env_cls(
            scenario, enable_opencv_display=args.enable_opencv_display, opencv_render_env_idx=args.opencv_render_env_idx
        )
    else:
        env = env_cls(scenario)
    device = torch.device(args.device)

    if not args.debug:
        dump_instance_file(task_cfg, os.path.join(log_dir, "cfg.py"))
        dump_instance_file(env, os.path.join(log_dir, "env.py"))
        shutil.copy("train.sh", os.path.join(log_dir, "train.sh"))

    if args.wandb and not args.debug:
        env.train_cfg["logger"] = "wandb"

    ppo_runner = OnPolicyRunner(
        env=env,
        train_cfg=env.train_cfg,
        device=device,
        log_dir=log_dir,
        use_vision=task_cfg.use_vision,
        debug=args.debug,
    )

    if args.resume:
        resume_path = get_load_path(args)
        if not os.path.exists(resume_path):
            raise FileNotFoundError(f"Resume path {resume_path} does not exist")
        log.info(f"Loading model from: {resume_path}")
        ppo_runner.load(resume_path, load_optimizer=False)

ppo_runner.learn(num_learning_iterations=args.num_learning_iterations, run_name=f"{args.run_name}_{now}")

ppo_runner.env.env.simulation_app.close()
