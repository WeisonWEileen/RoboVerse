import rootutils

rootutils.setup_root(__file__, pythonpath=True)


import torch
from metasim.scenario.scenario import ScenarioCfg
from humanoid_visualrl.actor_critic.on_policy_runner import OnPolicyRunner
from humanoid_visualrl.utils.utils import (
    export_policy_as_jit,
    get_cfg_cls,
    get_args,
    get_export_jit_path,
    get_load_path,
    get_log_dir,
    get_env_wrapper_cls,
)

from loguru import logger as log
from humanoid_visualrl.actor_critic.on_policy_runner import OnPolicyRunner


def play(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    scenario = ScenarioCfg(
        robots=[args.robot],
        num_envs=args.num_envs,
        simulator=args.sim,
        headless=args.headless,
        cameras=[],
    )
    scenario.num_envs = 1

     
    task_cfg, _ = get_cfg_cls(args)
    task_cfg.commands.curriculum = False
    task_cfg.ppo_cfg.resume = True
    # add objects
    scenario.objects = task_cfg.objects
    if args.use_vision or args.use_resnet or args.use_fixed_gazing:
        scenario.cameras = [task_cfg.camera]
    else:
        scenario.cameras = []

    # task assign and override
    scenario.sim_params = task_cfg.sim_params
    scenario.decimation = task_cfg.decimation
    scenario.render_interval = scenario.decimation
    scenario.task = task_cfg
    scenario.env_spacing = task_cfg.env_spacing

    # log_dir = get_log_dir(args, scenario)
    env, _ = get_env_wrapper_cls(args, scenario)
    load_path = get_load_path(args, scenario)

    obs, _ = env.get_observations()
    # load policy
    ppo_runner = OnPolicyRunner(
        env=env,
        train_cfg=env.train_cfg,
        device=device,
        # log_dir=log_dir,
        use_vision=args.use_vision,
    )
    ppo_runner.load(load_path)
    policy = ppo_runner.get_inference_policy(device=env.device)

    # export policy as a jit module (used to run it from C++)
    if args.export_policy:
        export_jit_path = get_export_jit_path(args, scenario)
        export_policy_as_jit(ppo_runner.alg.actor_critic, export_jit_path)
        print("Exported policy as jit script to: ", export_jit_path)

    # env.init_states.objects["cube"].root_state[0, :1] = 0.2
    env.init_states.objects["cube"].root_state[0, 1] = 0.0
    # breakpoint()
    env.cfg.max_episode_length_s = 100000
    env.env.set_states(env.init_states)

    reset_interval = 75
    for i in range(10000):
        # set fixed command
        if i % reset_interval == 0:
            if i == 0:
                env.init_states.objects["cube"].root_state[0, 1] = 0.075
                # env.init_states.objects["cube"].root_state[0, 1] = 0.15
            # if i == reset_interval:
            #     env.init_states.objects["cube"].root_state[0, 1] = 0.075
            # if i == 2 * reset_interval:
            #     env.init_states.objects["cube"].root_state[0, 1] = 0.0
            # if i == 3 * reset_interval:
            #     env.init_states.objects["cube"].root_state[0, 1] = -0.075
            # if i == 4 * reset_interval:
            #     env.init_states.objects["cube"].root_state[0, 1] = -0.15
            # env.init_states.objects["cube"].root_state[0, 1] *= -1
                env._reset([0])
                env._compute_observations()
        # if i == 200:
        #     env.init_states.objects["cube"].root_state[0, 1] = 0.15
        #     env.env.set_states(env.init_states)
        env.commands[:, 0] = 0.0
        env.commands[:, 1] = 0.0
        env.commands[:, 2] = 0.0
        env.commands[:, 3] = 0.0

        if args.use_vision:
            actions = policy(obs)
        else:
            actions = policy(obs.detach())
        # print(actions)
        # breakpoint()
        # for i in task_cfg.decimation:
        obs, rewards, dones, infos = env.step(actions.detach())
        log.info(f"step: {i}")

    env.env.close()


if __name__ == "__main__":
    EXPORT_POLICY = True
    args = get_args()
    play(args)
