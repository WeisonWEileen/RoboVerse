"""This script is for batch evaluation of the see_flag and success rate.

When there are more than 60 percentage of frames the cube are withn 25 pixel distance from the center of the fov in 7 seconds, the evaluation is successful.

divide the (-task_cfg.randomize_object_yaw_range, task_cfg.randomize_object_yaw_range) into 10 parts, and batchrize evalute.
"""

import rootutils

rootutils.setup_root(__file__, pythonpath=True)

from metasim.scenario.objects import PrimitiveCubeCfg

from metasim.constants import PhysicStateType
import os
import random
import torch
from loguru import logger as log

from humanoid_visualrl.actor_critic.on_policy_runner import OnPolicyRunner
from humanoid_visualrl.utils.utils import (
    export_policy_as_jit,
    get_args,
    get_export_jit_path,
    get_load_path,
    load_task_cfg,
    load_wrapper,
)
from metasim.scenario.lights import DomeLightCfg
from metasim.scenario.scenario import ScenarioCfg
from humanoid_visualrl.utils.video_saver import VideoSaver

N_DIVIDE = 10
SUCCESS_FLAG_THRESHOLD = 0.8
SUCCESS_REACHING_FRAMES_THRESHOLD = 0.3

def sample_yaw_ranges(R, N_divide, N_env_total, device):
    """
    返回 (N_env_total,) 的 yaw，每 N_interval_envs 来自一个子区间
    """
    N_interval = N_env_total // N_divide
    yaws = []

    for i in range(N_divide):
        # 子区间 [a, b]
        a = -R + i * (2 * R / N_divide)
        b = -R + (i + 1) * (2 * R / N_divide)

        # 在子区间内均匀随机采样 N_interval_envs 个
        y_i = torch.rand(N_interval, device=device) * (b - a) + a
        yaws.append(y_i)

    return torch.cat(yaws, dim=0)

def play(args):
    """Run the trained policy in the environment.

    Args:
        args: Command line arguments containing configuration
    """
    device = args.device
    load_path = get_load_path(args)

    # get task cfg from cfg.py in load_path
    # breakpoint()
    task_cfg_cls = load_task_cfg(args)
    task_cfg = task_cfg_cls(actor_critic_class = args.actor_critic_class, occlude_cube=args.eval_occlu)

    assert args.num_envs % N_DIVIDE == 0, f"num_envs must be divisible by {N_DIVIDE} for batch evaluation, but got {args.num_envs}"
    N_interval_envs = args.num_envs // N_DIVIDE

    scenario = ScenarioCfg(
        robots=[task_cfg.robot],
        num_envs=args.num_envs,
        simulator=args.sim,
        headless=args.headless,
        cameras=[],
    )
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
    if args.eval_occlu:
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
        task_cfg.filter_pairs.append(
            (task_cfg.robot, "occlusion_cube")
        )
        task_cfg.filter_pairs.append(
            ("object", "occlusion_cube")
        )
        

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
    scenario.filter_pairs = task_cfg.filter_pairs
    
    # log_dir = get_log_dir(args, scenario)
    from humanoid_visualrl.wrapper.active_vision_cube_wrapper import ActiveVisionWrapper
    assert not (args.eval_randomize_material_test and args.eval_randomize_material_train), "Must evaluate material in test or train mode, not both"
    if args.eval_randomize_material_test:
        task_cfg.mode = "test"
        task_cfg.randomize_material = True
        log.info(f"Evaluating material in test mode")
    elif args.eval_randomize_material_train:
        task_cfg.mode = "train"
        task_cfg.randomize_material = True

        log.info(f"Evaluating material in train mode")
    else:
        log.info(f"Not evaluating material")


    if args.eval_reaching:
        task_cfg.mask_joint_names = [
            "right_elbow_joint",
            "right_shoulder_pitch_joint",
            "right_shoulder_roll_joint",
            "right_shoulder_yaw_joint",
        ]

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
    ppo_runner.load(load_path, device=device)
    policy = ppo_runner.get_inference_policy(device=env_wrapper.device)

    # export policy as a jit module (used to run it from C++)
    if args.export_policy:
        export_jit_path = get_export_jit_path(args, scenario)
        export_policy_as_jit(ppo_runner.alg.actor_critic, export_jit_path)
        log.info(f"Exported policy as jit script to: {export_jit_path}")

    if args.eval_occlu:
        env_wrapper.env.filter_collisions(env_wrapper.robot.name, "occlusion_cube")
        env_wrapper.env.filter_collisions("object", "occlusion_cube")

    # env.init_states.objects["object"].root_state[0, :1] = 0.2
    env_wrapper.init_states.objects["object"].root_state[0, 1] = 0.0
    # breakpoint()
    # env_wrapper.cfg.max_episode_length_s = 100000
    env_wrapper.env.set_states(env_wrapper.init_states)
    # env_wrapper.enable_opencv_display = True
    env_wrapper.env._render_viewport = True
    env_wrapper.env.init_marker_viz()
    env_wrapper._update_camera_pose = True
    obs, _ = env_wrapper.get_observations()

    evalation_save_dir = os.path.join(os.path.dirname(load_path), "evaluation")
    os.makedirs(evalation_save_dir, exist_ok=True)
    log.info(f"Evaluator initialized, saving results to: {os.path.join(os.path.dirname(load_path), 'evaluation')}")

    yaw = torch.tensor(0.0, device=env_wrapper.device)
    yaw = (random.random() - 0.5) * 2 * 3.14
    yaw = torch.tensor(yaw, device=env_wrapper.device)
    evaluation_round = args.evaluation_round
    
    video_saver = VideoSaver(os.path.join(evalation_save_dir, "see_video.mp4"))
    success_flag_average_acc = torch.zeros(env_wrapper.num_envs, device=env_wrapper.device, dtype=torch.float32) # for accumulate and then print each interval
    success_flag_acc = torch.zeros(env_wrapper.num_envs, device=env_wrapper.device, dtype=torch.int8)
    if args.eval_reaching:
        success_reaching_flag_acc = torch.zeros(env_wrapper.num_envs, device=env_wrapper.device, dtype=torch.int8)
    obj_rand_range = task_cfg.randomize_object_yaw_range 

    # total_step_count = int(7 / 0.025)  # 7s
    total_step_count = int(7 / 0.025) # 7s
    for i in range(evaluation_round):
        ppo_runner.alg.policy.reset(list(range(env_wrapper.num_envs)))
        env_wrapper._reset(list(range(env_wrapper.num_envs)))
        object_state = env_wrapper.init_states.objects["object"].root_state
        
        yaw = sample_yaw_ranges(obj_rand_range, N_DIVIDE, env_wrapper.num_envs, env_wrapper.device)
            # yaw = 2.3 * 0.5
            # yaw = torch.tensor(yaw, device=env_wrapper.device)
        radius = task_cfg.randomize_object_radius
        radius_bias = 2 * (torch.rand(env_wrapper.num_envs, device=env_wrapper.device) - 0.5) * 0.1
        # radius_bias = 0.0
        
        # reset policy
        object_x = torch.cos(yaw) * (radius + radius_bias)
        object_y = torch.sin(yaw) * (radius + radius_bias)


        object_state[:, 0] = object_x
        object_state[:, 1] = object_y

        env_wrapper.env._set_object_pose(
            env_wrapper.cfg.objects[2], object_state[:, :3], object_state[:, 3:7], env_ids=list(range(env_wrapper.num_envs))
        )
        env_wrapper._compute_observations()
        # reset texture and material
        if task_cfg.randomize_obj_material:
            env_wrapper.env.randomize_obj_material(list(range(env_wrapper.num_envs)), env_wrapper.obj)

        if args.eval_occlu:
            occlusion_cube_radius = torch.ones(env_wrapper.num_envs, device=env_wrapper.device) * (radius - 0.13)
            occlusion_cube_x = torch.cos(yaw) * occlusion_cube_radius
            occlusion_cube_y = torch.sin(yaw) * occlusion_cube_radius
            occlusion_cube_state = env_wrapper.init_states.objects["occlusion_cube"].root_state
            occlusion_cube_state[:, 0] = occlusion_cube_x
            occlusion_cube_state[:, 1] = occlusion_cube_y
            env_wrapper.env._set_object_pose(
                env_wrapper.cfg.objects[3],
                occlusion_cube_state[:, :3],
                occlusion_cube_state[:, 3:7],
                env_ids=list(range(env_wrapper.num_envs)),
            )
        if args.eval_randomize_material_train or args.eval_randomize_material_test:
            env_wrapper.domain_randomization_helper.randomization(env_ids=list(range(env_wrapper.num_envs)), force_randomize=True)

        obs, _ = env_wrapper.get_observations()

        success_flag_acc_single_count = torch.zeros(env_wrapper.num_envs, device=env_wrapper.device)
        if args.eval_reaching:
            success_reaching_flag_acc_single_count = torch.zeros(env_wrapper.num_envs, device=env_wrapper.device)

        for _ in range(total_step_count):

            if task_cfg.use_vision:
                actions = policy(obs)
            else:
                actions = policy(obs.detach())
            
            obs, _, _, infos = env_wrapper.step_evaluate(actions.detach())
            state = env_wrapper.env.get_states()
            env_wrapper._refreshed_tensors(state)
            reward = env_wrapper._reward_pixel_norm_at_object(state, env_wrapper.robot.name, env_wrapper.cfg)


            success_flag_acc_single_count += env_wrapper.see_flag

            if args.eval_reaching:
                wrist_pos = state.robots[env_wrapper.robot.name].body_state[:, env_wrapper.left_index_intermediate_link_indices, :7
                ]
                dist = torch.norm(wrist_pos[:, 0, :3] - state.objects["object"].root_state[:, :3]-env_wrapper.env.scene.env_origins, dim=1)
                dis_reaching_flag = dist < 0.3
                success_reaching_flag = (dis_reaching_flag & env_wrapper.see_flag).float()
                success_reaching_flag_acc_single_count += success_reaching_flag



            camera_pos = env_wrapper.camera_pos_w[:, :3]
            camera_quat = env_wrapper.camera_quat_w[:, :4]
            camera_direction = env_wrapper.object_pose_buf[:, :3] - camera_pos


            env_wrapper._update_marker_viz(
                camera_pos,
                camera_quat,
                camera_direction,
            )

            if i < 2 :
                video_saver.add(env_wrapper)

        success_flag_average = success_flag_acc_single_count / total_step_count
        success_flag_average_acc += success_flag_average
        success_flag = success_flag_average > SUCCESS_FLAG_THRESHOLD
        success_flag_acc += success_flag.to(torch.int8)

        if args.eval_reaching:
            success_reaching_flag_ = (success_reaching_flag_acc_single_count / total_step_count) > SUCCESS_REACHING_FRAMES_THRESHOLD
            success_reaching_flag_acc += success_reaching_flag_
            log.info(
                f"reaching success_flag: {success_reaching_flag} for evaluation round {i}, success_flag_average: {success_reaching_flag_}"
            )
            
        # success_flag = success_flag_average > SUCCESS_FLAG_THRESHOLD
        # if success_flag:
        if not args.eval_reaching:
            log.info(f"success_flag: {success_flag} for evaluation round {i}, success_flag_average: {success_flag_average}")


        # generation different interval success rate


        if i == 2 :
            video_saver.save()
            log.info(f"Saved {i} round video at: {video_saver.video_path}")
    
    # # generate success rate for each interval
    success_flag_average_acc = success_flag_average_acc / evaluation_round
    success_flag_acc = success_flag_acc / evaluation_round
    if args.eval_reaching:
        success_reaching_flag_acc = success_reaching_flag_acc / evaluation_round
    else:
        success_reaching_flag_acc = torch.zeros(env_wrapper.num_envs, device=env_wrapper.device, dtype=torch.int8)

    # for i in range(N_DIVIDE):
    #     success_flag_average = success_flag_average_acc[i*N_interval_envs:(i+1)*N_interval_envs].mean()
    #     log.info(
    #         f"success_flag: {success_flag_average} for interval {-obj_rand_range + i * 2 * obj_rand_range / N_DIVIDE:.2f} to {-obj_rand_range + (i + 1) * 2 * obj_rand_range / N_DIVIDE:.2f}, success_flag_average: {success_flag_average}"
    #     )
    # log.info(f"success_flag_acc: {success_flag_acc.float().mean()}")
    # log.info(f"success_flag_average_acc: {success_flag_average_acc.float().mean()}")



    # # generate success rate for each interval
    # success_flag_average_acc = success_flag_average_acc / evaluation_round
    # success_flag_acc = success_flag_acc / evaluation_round

    # # -------- Format table header --------
    # table_lines = []
    # table_lines.append("| Interval (Yaw Range) | Success Rate |")
    # table_lines.append("|----------------------|--------------|")

    # # -------- Fill each interval row --------
    # for i in range(N_DIVIDE):
    #     start = -obj_rand_range + i * 2 * obj_rand_range / N_DIVIDE
    #     end = -obj_rand_range + (i + 1) * 2 * obj_rand_range / N_DIVIDE

    #     success_flag_average = success_flag_average_acc[i * N_interval_envs : (i + 1) * N_interval_envs].mean()

    #     table_lines.append(f"| [{start:.2f}, {end:.2f}] | {100 * success_flag_average.item():.2f}% |")

    # # -------- Add overall average --------
    # overall_success = success_flag_average_acc.float().mean().item()
    # table_lines.append(f"| **Overall Avg** | **{100 * overall_success:.2f}%** |")

    # # Print table nicely
    # log.info("\n" + "\n".join(table_lines))

    interval_success_rates = []

    for i in range(N_DIVIDE):
        rate = success_flag_acc[i * N_interval_envs : (i + 1) * N_interval_envs].mean().item()
        interval_success_rates.append(rate)

    # print ASCII bar chart like wandb
    log.info("\nSuccess Rate Bar Chart (ASCII)")
    max_bar_len = 50

    for i, rate in enumerate(interval_success_rates):
        bar = "█" * int(rate * max_bar_len)
        pct = f"{rate * 100:.1f}%"
        start = -obj_rand_range + i * 2 * obj_rand_range / N_DIVIDE
        end = -obj_rand_range + (i + 1) * 2 * obj_rand_range / N_DIVIDE
        log.info(f"[{start:.2f}, {end:.2f}] | {bar:<50} | {pct}")

    # ---------------- SAVE PNG PLOT ---------------- #
    # ---------------- SAVE PNG PLOT WITH RANGE TICKS (non-overlap version) ---------------- #
    try:
        import matplotlib.pyplot as plt
        import numpy as np

        x = np.arange(N_DIVIDE)
        y = np.array(interval_success_rates)

        tick_labels = []
        for i in range(N_DIVIDE):
            start = -obj_rand_range + i * 2 * obj_rand_range / N_DIVIDE
            end = -obj_rand_range + (i + 1) * 2 * obj_rand_range / N_DIVIDE
            tick_labels.append(f"[{start:.2f},{end:.2f}]")

        plt.figure(figsize=(12, 4))

        # add extra top margin so labels don't hit the title
        plt.subplots_adjust(top=0.80, bottom=0.25)

        plt.bar(x, y, color="skyblue")
        plt.ylim(0, 1.05)  # allow room above bars

        plt.ylabel("Success Rate")

        # Title moved upward
        plt.title("Success Rate by Yaw Interval", y=1.12)

        plt.xticks(x, tick_labels, rotation=45, ha="right")

        # annotate %
        for i, v in enumerate(y):
            plt.text(i, v + 0.015, f"{v * 100:.1f}%", ha="center", fontsize=9)
        if args.eval_occlu:
            png_path = os.path.join(evalation_save_dir, f"success_rate_ckpt_occlu_{args.checkpoint}.png")
        else:
            png_path = os.path.join(evalation_save_dir, f"success_rate_ckpt_{args.checkpoint}.png")
        plt.savefig(png_path, dpi=200)
        plt.close()

        log.info(f"Saved success rate bar chart to {png_path}")

    except Exception as e:
        log.error(f"Failed to save PNG bar chart: {e}")





    # Close evaluator and save results
    # evaluator.close()
    env_wrapper.env.close()


if __name__ == "__main__":
    EXPORT_POLICY = True
    args = get_args()
    play(args)
