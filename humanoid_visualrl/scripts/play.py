import rootutils

rootutils.setup_root(__file__, pythonpath=True)


import os

import torch
from loguru import logger as log
from metasim.scenario.scenario import ScenarioCfg

from humanoid_visualrl.actor_critic.on_policy_runner import OnPolicyRunner
from humanoid_visualrl.utils.utils import (
    export_policy_as_jit,
    get_args,
    get_export_jit_path,
    get_load_path,
    load_task_cfg,
    load_wrapper,
)

import random






def play(args):
    """Run the trained policy in the environment.

    Args:
        args: Command line arguments containing configuration
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    load_path = get_load_path(args)

    # get task cfg from cfg.py in load_path
    # breakpoint()
    task_cfg = load_task_cfg(args)
    scenario = ScenarioCfg(
        robots=[task_cfg.robot],
        num_envs=args.num_envs,
        simulator=args.sim,
        headless=args.headless,
        cameras=[],
    )
    scenario.num_envs = 1
    # scenario.task

    task_cfg.commands.curriculum = False
    task_cfg.ppo_cfg.resume = True
    # add objects
    scenario.objects = task_cfg.objects

    scenario.cameras = task_cfg.cameras

    # task assign and override
    scenario.sim_params = task_cfg.sim_params
    scenario.decimation = task_cfg.decimation
    scenario.render_interval = scenario.decimation
    scenario.task = task_cfg
    # breakpoint()
    scenario.env_spacing = task_cfg.env_spacing
    task_cfg.randomization = False
    # log_dir = get_log_dir(args, scenario)
    env_wrapper = load_wrapper(args, scenario)
    # load_path = get_load_path(args)

    # load policy
    ppo_runner = OnPolicyRunner(
        env=env_wrapper,
        train_cfg=env_wrapper.train_cfg,
        device=device,
        # log_dir=log_dir,
        use_vision=task_cfg.use_vision,
    )
    ppo_runner.load(load_path)
    policy = ppo_runner.get_inference_policy(device=env_wrapper.device)

    # export policy as a jit module (used to run it from C++)
    if args.export_policy:
        export_jit_path = get_export_jit_path(args, scenario)
        export_policy_as_jit(ppo_runner.alg.actor_critic, export_jit_path)
        log.info(f"Exported policy as jit script to: {export_jit_path}")

    # env.init_states.objects["cube"].root_state[0, :1] = 0.2
    env_wrapper.init_states.objects["cube"].root_state[0, 1] = 0.0
    # breakpoint()
    env_wrapper.cfg.max_episode_length_s = 100000
    env_wrapper.env.set_states(env_wrapper.init_states)
    env_wrapper.enable_opencv_display = True
    env_wrapper.env._render_viewport = True
    obs, _ = env_wrapper.get_observations()

    reset_interval = 75
    yaw = torch.tensor(0.0, device=env_wrapper.device)
    # set fixed command
    yaw = (random.random()-0.5) * 0.3 - 3.14/3
    yaw = torch.tensor(yaw, device=env_wrapper.device)


    for i in range(10000):

        if i % reset_interval == 0:
            yaw += 0.3
            radius = task_cfg.randomize_cube_radius -0.1
            # radius_bias = 2 * random.random() * 0.8
            radius_bias = 0.0
            
            cube_x = torch.cos(yaw) * (radius + radius_bias)
            cube_y = torch.sin(yaw) * (radius + radius_bias)
            cube_state = env_wrapper.init_states.objects["cube"].root_state
            cube_state[0, 0] = cube_x
            cube_state[0, 1] = cube_y
            env_wrapper.env._set_object_pose(env_wrapper.cfg.objects[1], cube_state[:, :3], cube_state[:, 3:7], env_ids=[0])
            env_wrapper._compute_observations()
            # ppo_runner.alg.policy.reset([0])
        
        if task_cfg.use_vision:
            actions = policy(obs)
        else:
            actions = policy(obs.detach())
        obs, _, _, _ = env_wrapper.step(actions.detach())

    env_wrapper.env.close()


if __name__ == "__main__":
    EXPORT_POLICY = True
    args = get_args()
    play(args)
