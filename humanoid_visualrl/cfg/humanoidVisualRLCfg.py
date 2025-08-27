from __future__ import annotations

"""Base class for legged-gym style legged-robot tasks."""

from dataclasses import MISSING
from typing import Callable

import torch

from metasim.scenario.robot import RobotCfg
from metasim.scenario.simulator_params import SimParamCfg
from metasim.types import TensorState
from metasim.utils import configclass


@configclass
class LeggedRobotRunnerCfg:
    """Configuration for PPO."""

    seed = 1
    runner_class_name = "OnPolicyRunner"

    @configclass
    class Policy:
        """Network config class for PPO."""

        class_name = "ActorCritic"

        init_noise_std = 1.0
        """Initial noise std for actor network."""
        actor_hidden_dims = [512, 256, 128]
        """Hidden dimensions for actor network."""
        critic_hidden_dims = [768, 256, 128]
        """Hidden dimensions for critic network."""

    @configclass
    class Algorithm:
        """Training config class for PPO."""

        value_loss_coef = 1.0
        """Value loss coefficient."""
        use_clipped_value_loss = True
        """Use clipped value loss."""
        clip_param = 0.2
        """Clipping parameter for PPO."""
        entropy_coef = 0.001
        """Entropy coefficient."""
        num_learning_epochs = 5
        """Number of learning epochs."""
        num_mini_batches = 4
        """mini batch size = num_envs*n_steps / num_mini_batches"""
        learning_rate = 1.0e-3
        schedule = "adaptive"
        gamma = 0.99
        lam = 0.95
        desired_kl = 0.01
        max_grad_norm = 1.0
        class_name = "PPO"

    class_name = "ActorCritic"
    """Policy class name."""
    algorithm_class_name = "PPO"
    """Algorithm class name."""
    num_steps_per_env = 24
    """per iteration"""
    max_iterations = 1500
    """max number of iterations"""

    # logging
    logger: str = "wandb"
    wandb_project: str = "humanoid_visualrl"

    save_interval = 1000
    """save interval for checkpoints"""
    experiment_name = "test"
    """experiment name"""
    run_name = ""
    resume = False
    """resume from checkpoint"""
    load_run = -1
    """load run number"""
    checkpoint = -1
    """checkpoint name"""
    resume_path = None

    policy: Policy = Policy()
    algorithm: Algorithm = Algorithm()


@configclass
class BaseTableHumanoidTaskCfg:
    """Base class for legged-gym style humanoid tasks.

    Attributes:
    robotname: name of the robot
    feet_indices: indices of the feet joints
    penalised_contact_indices: indices of the contact joints
    """

    decimation: int = 10  # for isaacgym
    decimation: int = 4  # for isaacsim
    episode_length: int = MISSING
    reward_functions: list[callable[[list[TensorState], str | None], torch.FloatTensor]] = MISSING
    reward_weights: list[float] = MISSING
    sim_params: SimParamCfg = SimParamCfg()

    @configclass
    class RewardCfg:
        """Constants for reward computation."""

        # base_height_target: float = 0.728  # for g1
        base_height_target: float = 0.78  # follow asap
        """target height of the base"""
        min_dist: float = 0.05
        """minimum distance between feet"""
        max_dist: float = 0.25
        """maximum distance between feet"""

        target_joint_pos_scale: float = 0.17
        """target joint position scale"""
        target_feet_height: float = 0.06
        """target feet height"""
        cycle_time: float = 0.64
        """cycle time"""

        only_positive_rewards: bool = True
        """whether to use only positive rewards"""
        tracking_sigma: float = 5.0
        """tracking reward = exp(error*sigma)"""
        max_contact_force: float = 700.0
        """maximum contact force"""
        soft_torque_limit: float = 0.001
        """soft torque limit"""

    reward_cfg: RewardCfg = RewardCfg()

    @configclass
    class CommandsConfig:
        """Configuration for command generation.

        Attributes:
            curriculum: whether to start curriculum training
            max_curriculum.
            num_commands: number of commands.
            resampling_time: time before command are changed[s].
            heading_command: whether to compute ang vel command from heading error.
            ranges: upperbound and lowerbound of sampling ranges.
        """

        curriculum: bool = False
        """whether to start curriculum training"""
        max_curriculum: float = 1.0
        """maximum value of curriculum"""
        num_commands: int = 4
        """number of commands. linear x, linear y, angular velocity, heading"""
        resampling_time: float = 10.0
        """time before command are changed[s]."""
        heading_command: bool = True
        """whether to compute ang vel command from heading error."""

    @configclass
    class Normalization:
        """Normalization constants for observations and actions."""

        class obs_scales:
            lin_vel = 2.0
            ang_vel = 1.0
            dof_pos = 1.0
            dof_vel = 0.05
            quat = 1.0
            height_measurements = 5.0

        clip_observations = 18.0
        clip_actions = 18.0

    @configclass
    class CommandRanges:
        """Command Ranges for random command sampling when training."""

        lin_vel_x: list[float] = [0.0, 1.0]
        lin_vel_y: list[float] = [-0.5, 0.5]
        ang_vel_yaw: list[float] = [-1.0, 1.0]
        heading: list[float] = [-3.14, 3.14]

    reward_functions: list[Callable] = MISSING
    reward_weights: dict[str, float] = MISSING

    robots: list[RobotCfg] | None = None
    """List of robots in the environment."""
    command_ranges: CommandRanges = CommandRanges()
    """Command Ranges for random command sampling when training."""
    commands = CommandsConfig()
    """Configuration for command generation."""

    use_vision: bool = False
    """Whether to use vision observations."""
    ppo_cfg: LeggedRobotRunnerCfg = LeggedRobotRunnerCfg()
    """PPO config."""
    normalization = Normalization()
    """Normalization config."""
    decimation: int = 4
    """Decimation pd control loop."""
    num_obs: int = 124
    """Number of observations."""
    num_privileged_obs: int = None
    """Number of privileged observations. If not None a priviledge_obs_buf will be returned by step() (critic obs for assymetric training). None is returned """

    env_spacing: float = 2.0
    """Environment spacing."""
    send_timeouts: bool = True
    """Whether to send time out information to the algorithm"""
    episode_length_s: float = 20.0
    """episode length in seconds"""
    feet_indices: torch.Tensor = MISSING
    """feet indices"""
    penalised_contact_indices: torch.Tensor = MISSING
    """penalised contact indices for reward computation"""
    termination_contact_indices: torch.Tensor = MISSING
    """termination contact indices for reward computation"""
    sim_params = SimParamCfg(
        # dt=0.001,  # for isaacgym
        dt=0.005,  # for isaacsim
        contact_offset=0.01,
        num_position_iterations=4,
        num_velocity_iterations=0,
        bounce_threshold_velocity=0.5,
        replace_cylinder_with_capsule=True,
        friction_offset_threshold=0.04,
        num_threads=10,
    )
    """Simulation parameters with physics engine settings."""
    dt = decimation * sim_params.dt
    """simulation time step in s"""
    objects = []
    """objects in the environment"""
    traj_filepath = None
    """path to the trajectory file"""
    # TODO read form max_episode_length_s and divide s
    max_episode_length_s: int = 24
    """maximum episode length in seconds"""
    episode_length: int = 2400
    """episode length in steps"""
    max_episode_length: int = 2400
    """episode length in steps"""

    frame_stack = 1
    c_frame_stack = 3

    command_dim = 3
    num_actions: int = 29
    """Number of actions."""
    num_single_obs: int = 3 * num_actions + 6 + command_dim  #
    num_observations: int = int(frame_stack * num_single_obs)
    single_num_privileged_obs: int = 4 * num_actions + 23
    num_privileged_obs = int(c_frame_stack * single_num_privileged_obs)
    task_name = "walking"

    @configclass
    class HumanoidExtraCfg:
        """An Extension of cfg.

        Attributes:
        delay: delay in seconds
        freq: frequency for controlling sample waypoint
        resample_on_env_reset: resample waypoints on env reset
        """

        delay: float = 0.0
        freq: int = 10
        resample_on_env_reset: bool = True

    humanoid_extra_cfg: HumanoidExtraCfg = HumanoidExtraCfg()

    init_states = [
        {
            "objects": {},
            "robots": {
                "g1": {
                    # "pos": torch.tensor([0.0, 0.0, 0.737]),  # 0.78 is taken from unitree rl_gym
                    "pos": torch.tensor([0.0, 0.0, 0.8]),  # 0.8 is for g1 29 dof
                    "rot": torch.tensor([1.0, 0.0, 0.0, 0.0]),
                    "dof_pos": {
                        "left_hip_pitch_joint": -0.4,
                        "left_hip_roll_joint": 0,
                        "left_hip_yaw_joint": 0.0,
                        "left_knee_joint": 0.8,
                        "left_ankle_pitch_joint": -0.4,
                        "left_ankle_roll_joint": 0,
                        "right_hip_pitch_joint": -0.4,
                        "right_hip_roll_joint": 0,
                        "right_hip_yaw_joint": 0.0,
                        "right_knee_joint": 0.8,
                        "right_ankle_pitch_joint": -0.4,
                        "right_ankle_roll_joint": 0,
                        "left_wrist_roll_joint": 0,
                        "right_wrist_roll_joint": 0,
                        "waist_yaw_joint": 0.0,
                        "left_shoulder_pitch_joint": 0.0,
                        "left_shoulder_roll_joint": 0.0,
                        "left_shoulder_yaw_joint": 0.0,
                        "left_elbow_joint": 0.0,
                        "right_shoulder_pitch_joint": 0.0,
                        "right_shoulder_roll_joint": 0.0,
                        "right_shoulder_yaw_joint": 0.0,
                        "right_elbow_joint": 0.0,
                    },
                },
            },
        }
    ]

    torque_limit_scale = 1.0

    reward_weights: dict[str, float] = {
        "termination": -0.0,
        "lin_vel_z": -0.0,
        # "ang_vel_xy": -0.05,
        "base_height": -10.0,
        "feet_air_time": 1.0,
        "collision": -1.0,
        "feet_stumble": -0.0,
        "stand_still": -0.0,
        "joint_pos": 1.6,
        "feet_clearance": 2.0,  # 1. * 2
        "feet_contact_number": 2.4,  # 1.2 * 2
        # gait
        "foot_slip": -0.05,
        "feet_distance": 0.2,
        "knee_distance": 0.2,
        # contact
        "feet_contact_forces": -0.01,
        # vel tracking
        "tracking_lin_vel": 2.4,  # 1.2 * 2
        "tracking_ang_vel": 2.2,  # 1.1 * 2
        "vel_mismatch_exp": 0.5,
        "low_speed": 0.2,
        "track_vel_hard": 1.0,  # 0.5 * 2
        # base pos
        "default_joint_pos": 0.5,  # 从1.0改为0.5
        "upper_body_pos": 2.0,  # 0.5 * 4
        "orientation": 1.0,
        "base_acc": 0.2,
        # energy
        "action_smoothness": -0.002,
        "torques": -1e-5,
        "dof_vel": -5e-4,
        "dof_acc": -1e-7,
        "torque_limits": 0.001,
        # optional
        "action_rate": -0.0,
    }

    # control
    action_scale = 0.25

    # push robot
    @configclass
    class PushRandomCfg:
        """Configuration for random push forces."""

        enabled: bool = False
        """Whether to enable random push forces."""
        max_push_vel_xy: float = 0.2
        """Maximum push velocity in xy plane."""
        max_push_ang_vel: float = 0.4
        """Maximum push angular velocity."""
        push_interval: int = 4
        """Interval in steps for applying random push forces and torques."""

    random_push = PushRandomCfg(enabled=True)
