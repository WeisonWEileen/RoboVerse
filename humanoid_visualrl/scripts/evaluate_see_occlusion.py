"""This script is used to evaluate the see_flag metric.
Randomly set the object position in range of randomize_object_yaw_range, and evaluate the see_flag metric.

When there are more than 60 percentage of frames the cube are withn 25 pixel distance from the center of the fov in 7 seconds, the evaluation is successful.
"""

import rootutils

rootutils.setup_root(__file__, pythonpath=True)


import os
import random

import torch
from loguru import logger as log
from metasim.constants import PhysicStateType

from humanoid_visualrl.actor_critic.on_policy_runner import OnPolicyRunner
from humanoid_visualrl.utils.evaluator import SeeFlagEvaluator
from humanoid_visualrl.utils.utils import (
    export_policy_as_jit,
    get_args,
    get_export_jit_path,
    get_load_path,
    load_task_cfg,
    load_wrapper,
)
from humanoid_visualrl.utils.video_saver import VideoSaver
from metasim.scenario.lights import DomeLightCfg
from metasim.scenario.scenario import ScenarioCfg
from metasim.scenario.objects import PrimitiveCubeCfg

IN_DISTRIBUTION_RAW_RANGE = 1.8
OUT_DISTRIBUTION_RAW_RANGE = 2.8


def play(args):
    """Run the trained policy in the environment.

    Args:
        args: Command line arguments containing configuration
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    load_path = get_load_path(args)

    # get task cfg from cfg.py in load_path
    # breakpoint()
    task_cfg_cls = load_task_cfg(args)
    task_cfg = task_cfg_cls(actor_critic_class=args.actor_critic_class)
    scenario = ScenarioCfg(
        robots=[task_cfg.robot],
        num_envs=args.num_envs,
        simulator=args.sim,
        headless=args.headless,
        cameras=[],
    )
    scenario.num_envs = 1
    scenario.device = args.device
    # scenario.task
    scenario.lights = [
        DomeLightCfg(
            intensity=1000.0,
            color=(0.85, 0.9, 1.0),
        ),
    ]

    task_cfg.commands.curriculum = False
    task_cfg.ppo_cfg.resume = True
    task_cfg.ppo_cfg.finetune = True
    task_cfg.ppo_cfg.policy.finetune = True
    task_cfg.objects.append(
        PrimitiveCubeCfg(
            name="occlusion_cube",
            size=(0.04, 0.04, 0.2),
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
    from humanoid_visualrl.wrapper.active_vision_cube_wrapper import ActiveVisionWrapper

    env_wrapper: ActiveVisionWrapper = load_wrapper(args, scenario)
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

    # env.init_states.objects["object"].root_state[0, :1] = 0.2
    env_wrapper.init_states.objects["object"].root_state[0, 1] = 0.0
    # breakpoint()
    # filter occlusion cube collision
    env_wrapper.env.filter_collisions(env_wrapper.robot.name, "occlusion_cube")
    env_wrapper.env.filter_collisions("object", "occlusion_cube")

    # env_wrapper.cfg.max_episode_length_s = 100000
    env_wrapper.env.set_states(env_wrapper.init_states)
    # env_wrapper.enable_opencv_display = True
    env_wrapper.env._render_viewport = True
    env_wrapper.env.init_marker_viz()
    env_wrapper._update_camera_pose = True
    obs, _ = env_wrapper.get_observations()

    evalation_save_dir = os.path.join(os.path.dirname(load_path), "evaluation")
    os.makedirs(evalation_save_dir, exist_ok=True)
    # Initialize evaluator
    evaluator = SeeFlagEvaluator(
        num_envs=env_wrapper.num_envs,
        window_size=100,  # Rolling average window
        save_dir=evalation_save_dir,
        plot_interval=5,  # Update plot every 5 steps
        enable_plot=True,
    )
    log.info(f"Evaluator initialized, saving results to: {os.path.join(os.path.dirname(load_path), 'evaluation')}")

    reset_interval = int(7 / 0.025)
    yaw = torch.tensor(0.0, device=env_wrapper.device)
    # set fixed command
    yaw = (random.random() - 0.5) * 2 * 3.14
    yaw = torch.tensor(yaw, device=env_wrapper.device)

    evaluation_round = 50

    video_saver = VideoSaver(os.path.join(evalation_save_dir, "see_video.mp4"))
    success_list = []

    total_step_count = int(7 / 0.025)  # 7s
    for i in range(evaluation_round):
        # reset and generate new object position
        # randomize occlusion cube color
        occlusion_cube_color = [random.random(), random.random(), random.random()]
        env_wrapper.cfg.objects[3].color = occlusion_cube_color

        # if i % reset_interval == 0:
        yaw = (random.random() - 0.5) * 2 * task_cfg.randomize_object_yaw_range * 0.5

        yaw = torch.tensor(yaw, device=env_wrapper.device)
        radius = task_cfg.randomize_object_radius
        radius_bias = 2 * (random.random() - 0.5) * 0.1
        # radius_bias = 0.0
        env_wrapper._reset(list(range(env_wrapper.num_envs)))
        # reset policy
        ppo_runner.alg.policy.reset(list(range(env_wrapper.num_envs)))
        object_x = torch.cos(yaw) * (radius +0.1 )
        object_y = torch.sin(yaw) * (radius +0.1 )
        object_state = env_wrapper.init_states.objects["object"].root_state
        object_state[0, 0] = object_x
        object_state[0, 1] = object_y
        env_wrapper.env._set_object_pose(
            env_wrapper.cfg.objects[2], object_state[:, :3], object_state[:, 3:7], env_ids=[0]
        )

        # set occlusion cube position in the front of the object
        occlusion_cube_radius = radius - 0.1
        occlusion_cube_x =  torch.cos(yaw) * occlusion_cube_radius
        occlusion_cube_y =  torch.sin(yaw) * occlusion_cube_radius
        occlusion_cube_state = env_wrapper.init_states.objects["occlusion_cube"].root_state
        occlusion_cube_state[0, 0] = occlusion_cube_x
        occlusion_cube_state[0, 1] = occlusion_cube_y
        env_wrapper.env._set_object_pose(
            env_wrapper.cfg.objects[3], occlusion_cube_state[:, :3], occlusion_cube_state[:, 3:7], env_ids=[0]
        )
        env_wrapper._compute_observations()
        # see_flag = env_wrapper.see_flag[0]

        # Print reset info
        # log.info(
        #     f"Step {i}: Reset - Object at ({object_x:.3f}, {object_y:.3f}), "
        #     f"yaw={yaw:.3f}, see_flag={see_flag.item()}"
        # )

        # reset texture and material
        if task_cfg.randomize_obj_material:
            env_wrapper.env.randomize_obj_material(list(range(env_wrapper.num_envs)), env_wrapper.obj)

        success_flag_acc = 0

        for step in range(total_step_count):
            if task_cfg.use_vision:
                actions = policy(obs)
            else:
                actions = policy(obs.detach())
            obs, _, _, infos = env_wrapper.step_evaluate(actions.detach())
            state = env_wrapper.env.get_states()
            env_wrapper._refreshed_tensors(state)
            reward = env_wrapper._reward_pixel_norm_at_object(state, env_wrapper.robot.name, env_wrapper.cfg)

            # if reward > env_wrapper.cfg.ema_reward_threshold:
            #     success_flag_acc += 1
            # break
            if env_wrapper.see_flag[0]:
                success_flag_acc += 1

            camera_pos = env_wrapper.camera_pos_w[:, :3]
            camera_quat = env_wrapper.camera_quat_w[:, :4]
            camera_direction = env_wrapper.object_pose_buf[:, :3] - camera_pos

            env_wrapper._update_marker_viz(
                camera_pos,
                camera_quat,
                camera_direction,
            )

            if i < 2:
                video_saver.add(env_wrapper)

        success_flag_average = success_flag_acc / total_step_count
        # if 75 percentage of frames the cube are withn 25 pixel distance from the center of the fov in 7 seconds, the evaluation is successful.
        if success_flag_average > 0.8:
            success_flag = True
        else:
            success_flag = False
        # if success_flag:
        log.info(
            f"success_flag: {success_flag} for round {i}, object_yaw: {yaw.item()}, success_flag_average: {success_flag_average}"
        )
        success_list.append(success_flag)

        if i == 2:
            video_saver.save()
            log.info(f"Saved {i} round video at: {video_saver.video_path}")

    # compute average success rate
    average_success_rate = sum(success_list) / len(success_list)
    log.info(f"average success rate in {evaluation_round} rounds: {average_success_rate}")

    # add episode reward for logging
    # if env_wrapper.reset_buf[0] > 0:
    #     # devide reward scale to get unscaled reward. [4:] is to remove 'rew_' prefix
    #     unscaled_reward = {k: v / env_wrapper.reward_scales[k[4:]] for k, v in infos["episode"].items()}
    #     evaluator.update_episode_reward(unscaled_reward)

    # Update evaluator with current see_flag
    # evaluator.update(env_wrapper.see_flag)

    # env_wrapper.

    # Close evaluator and save results
    evaluator.close()
    env_wrapper.env.close()


if __name__ == "__main__":
    EXPORT_POLICY = True
    args = get_args()
    play(args)
