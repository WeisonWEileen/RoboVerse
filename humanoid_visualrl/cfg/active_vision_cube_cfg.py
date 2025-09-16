from __future__ import annotations

"""Base class for legged-gym style legged-robot tasks."""

from dataclasses import MISSING
from typing import Callable, Literal

import torch

from metasim.constants import PhysicStateType
from metasim.scenario.objects import PrimitiveCubeCfg
from metasim.scenario.robot import RobotCfg
from metasim.scenario.simulator_params import SimParamCfg
from metasim.types import TensorState
from metasim.utils import configclass
from loguru import logger as log

from metasim.scenario.objects import RigidObjCfg


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
        # action_masking = False
        # masks_ids = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]

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
        num_mini_batches = 8
        """mini batch size = num_envs*n_steps / num_mini_batches"""
        learning_rate = 1.0e-3
        schedule = "adaptive"
        gamma = 0.99
        lam = 0.95
        desired_kl = 0.01
        max_grad_norm = 1.0
        class_name = "PPO"

        # mask = True
        # masks_ids = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]

    # class_name = "ActorCritic"
    """Policy class name."""
    algorithm_class_name = "PPO"
    """Algorithm class name."""
    num_steps_per_env = 24
    """per iteration"""
    max_iterations = 1500
    """max number of iterations"""

    # logging
    # logger: str = "wandb"
    wandb_project: str = "active_vision"

    save_interval = 100
    """save interval for checkpoints"""
    experiment_name = "test"
    """experiment name"""
    run_name = ""
    resume = True
    """resume from checkpoint"""
    load_run = -1
    """load run number"""
    checkpoint = -1
    """checkpoint name"""
    resume_path = None
    """resume path"""
    wandb = False
    """Whether to use wandb."""

    policy: Policy = Policy()
    algorithm: Algorithm = Algorithm()

    empirical_normalization = False


# @register_task("g1_static_dex1_fixed_gazing")
@configclass(name="active_vision")
class BaseTableHumanoidTaskCfg:
    """Base class for legged-gym style humanoid tasks.

    Attributes:
    robotname: name of the robot
    feet_indices: indices of the feet joints
    penalised_contact_indices: indices of the contact joints
    """

    decimation: int = 3
    episode_length: int = MISSING
    reward_functions: list[callable[[list[TensorState], str | None], torch.FloatTensor]] = MISSING
    reward_weights: list[float] = MISSING
    sim_params: SimParamCfg = SimParamCfg()

    @configclass
    class RewardCfg:
        """Constants for reward computation."""

        base_height_target: float = 0.89
        """target height of the base"""
        min_dist: float = 0.2
        """minimum distance between feet"""
        max_dist: float = 0.5
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
        heading_command = False

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

        lin_vel_x: list[float] = [-1.0, 2.0]
        lin_vel_y: list[float] = [-1.0, 1.0]
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
    # whether to use vision observation inside policy
    use_vision: bool = True
    use_rnn: bool = True
    actor_critic_class: Literal["use_vision", "use_rnn", "use_resnet"] = "use_resnet"
    """Whether to use vision observations."""
    ppo_cfg: LeggedRobotRunnerCfg = LeggedRobotRunnerCfg()
    """PPO config."""
    normalization = Normalization()
    """Normalization config."""
    decimation: int = 5
    """Decimation pd control loop."""
    num_obs: int = 124
    """Number of observations."""
    num_privileged_obs: int = None
    """Number of privileged observations. If not None a priviledge_obs_buf will be returned by step() (critic obs for assymetric training). None is returned """
    num_actions: int = 12
    """Number of actions."""
    env_spacing: float = 20.0
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
        dt=0.005,
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
    objects = [
        RigidObjCfg(
            name="table",
            scale=(0.5, 0.2, 0.5),
            physics=PhysicStateType.GEOM,
            usd_path="roboverse_data/ring_table.usd",
            fix_base_link=True,
            default_position=(0.0, 0.0, 0.65),
            default_orientation=(0.7071, 0.7071, 0.0000, 0.0000),
            collision_enabled=True,
            # urdf_path="metasim/example/example_assets/bbq_sauce/urdf/bbq_sauce.urdf",
            # mjcf_path="metasim/example/example_assets/bbq_sauce/mjcf/bbq_sauce.xml",
        ),
        PrimitiveCubeCfg(
            name="cube",
            size=(0.07, 0.07, 0.07),
            color=[1.0, 0.0, 0.0],
            physics=PhysicStateType.RIGIDBODY,
            collision_enabled=True,
            fix_base_link=False,
            default_position=(0.3, 0.1, 0.851),
        ),
    ]
    # cameras
    """objects in the environment"""
    traj_filepath = None
    """path to the trajectory file"""
    # TODO read form max_episode_length_s and divide s
    # max_episode_length_s: int = 6
    max_episode_length_s: int = 4
    """maximum episode length in seconds"""
    episode_length: int = 2400
    """episode length in steps"""
    max_episode_length: int = 2400
    """episode length in steps"""

    num_observations: int = 9
    num_actions: int = 9
    num_privileged_obs: int = 9
    max_episode_length: int = 2400

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
            "objects": {
                "cube": {
                    "pos": torch.tensor([0.7, 0.0, 0.875]),
                    "rot": torch.tensor([1.0, 0.0, 0.0, 0.0]),
                },
            },
            "robots": {
                # "g1_static": {
                #     "pos": torch.tensor([0.0, 0.0, 0.78]),
                #     "rot": torch.tensor([1.0, 0.0, 0.0, 0.0]),
                #     "dof_pos": {
                #         "waist_yaw_joint": 0.0,
                #         "left_shoulder_pitch_joint": 0.0,
                #         "left_shoulder_roll_joint": 0.0,
                #         "left_shoulder_yaw_joint": 0.0,
                #         "left_elbow_joint": 1.45,
                #         "right_shoulder_pitch_joint": 0.0,
                #         "right_shoulder_roll_joint": 0.0,
                #         "right_shoulder_yaw_joint": 0.0,
                #         "right_elbow_joint": 1.45,
                #     },
                # },
                "g1_static_dex1": {
                    "pos": torch.tensor([0.0, 0.0, 0.60]),
                    "rot": torch.tensor([0.8, 0.0, 0.0, 0.0]),
                    "dof_pos": {
                        "waist_yaw_joint": 0.0,
                        "waist_roll_joint": 0.0,
                        "waist_pitch_joint": 0.0,
                        "left_shoulder_pitch_joint": 0.0,
                        "left_shoulder_roll_joint": 0.0,
                        "left_shoulder_yaw_joint": 0.0,
                        "left_elbow_joint": 0.0,
                        "left_wrist_roll_joint": 0.0,
                        "left_wrist_pitch_joint": 0.0,
                        "left_wrist_yaw_joint": 0.0,
                        "right_shoulder_pitch_joint": 0.0,
                        "right_shoulder_roll_joint": 0.0,
                        "right_shoulder_yaw_joint": 0.0,
                        "right_elbow_joint": 0.0,
                        "right_wrist_roll_joint": 0.0,
                        "right_wrist_pitch_joint": 0.0,
                        "right_wrist_yaw_joint": 0.0,
                    },
                },
            },
        }
    ]

    command_dim = 14
    num_actions = 17
    torque_limit_scale = 1.0
    reward_weights: dict[str, float] = {
        # "upper_body_pos": 0.1,
        # "look_at_cube": 0.4,
        "pixel_norm_at_cube": 0.4,
        "cube_showup": 0.1,
    }

    frame_stack = 1
    c_frame_stack = 1

    # obs
    visual_dim: int = 512

    if use_vision:
        # s
        num_single_obs = num_actions * 3
        num_observations: int = int(frame_stack * num_single_obs)
        single_num_privileged_obs: int = num_actions * 3
        num_privileged_obs = int(c_frame_stack * single_num_privileged_obs)
    else:
        num_single_obs = num_actions * 3 + visual_dim
        num_observations = int(frame_stack * num_single_obs)
        # single_num_observations = 3 * num_actions + 6 + visual_dim

        # privileged obs
        single_num_privileged_obs = num_actions * 3 + 7 + visual_dim
        num_privileged_obs = int(c_frame_stack * single_num_privileged_obs)

    # control
    action_scale = 0.25

    task_name = "active_vision"

    from metasim.scenario.cameras import PinholeCameraCfg

    cameras = [
        PinholeCameraCfg(
            name="camera_first_person",
            data_types=["rgb"],
            # data_types=["rgb", "instance_id_seg"],
            # data_types=["rgb", "semantic_seg"],
            width=128,
            height=96,
            pos=(1.5, -1.5, 1.5),
            look_at=(0.0, 0.0, 0.0),
            # mount_to="g1_static/pelvis",
            # mount_to="g1_static",
            mount_to="g1_static_dex1",
            # mount_link="torso_link",
            # mount_link="torso_link/d435_link",
            mount_link="torso_link/d435_link",
            # though camera visulization maybe wrong , it's correct for fov
            # mount_pos=(0.05, 0.0, 0.0),
            mount_pos=(0.0, 0.0, 0.0),
            #     quat_xyzw = R.from_euler("xyz", [0, 60, 0], degrees=True).as_quat()
            # quat = (quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2])  #
            # mount_quat=(0.8660254037844387, 0.0, 0.49999999999999994, 0.0),
            # mount_quat=(0.5, -0.5, 0.5, -0.5),
            mount_quat=(1.0, 0.0, 0.0, 0.0),
        )
    ]

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

    random_push = PushRandomCfg(enabled=False)

    randomization = True

    

    def __post_init__(self):
        self.command_ranges.wrist_max_radius = 0.15
        self.command_ranges.l_wrist_pos_x = [-0.05, 0.15]
        self.command_ranges.l_wrist_pos_y = [-0.05, 0.15]
        self.command_ranges.l_wrist_pos_z = [-0.15, 0.15]
        self.command_ranges.r_wrist_pos_x = [-0.05, 0.15]
        self.command_ranges.r_wrist_pos_y = [-0.15, 0.05]
        self.command_ranges.r_wrist_pos_z = [-0.15, 0.15]

        # self.randomize_cube_y_offset = 0.1
        self.randomize_cube_curriculum = True
        self.randomize_cube_yaw_range = 2.3
        self.randomize_cube_radius = self.init_states[0]["objects"]["cube"]["pos"][0]
        
        # Curriculum learning parameters for cube yaw range
        self.curriculum_cube_yaw = True
        self.curriculum_cube_yaw_stages = [0.2, 0.5, 1.0]  # Multipliers for randomize_cube_yaw_range
        self.curriculum_cube_yaw_thresholds = [0.8, 0.8]  # Success rate thresholds to advance stages
        self.curriculum_cube_yaw_min_episodes = [500, 500]  # Minimum episodes before considering advancement

        self.actor_critic_class = "use_rnn"

        if self.actor_critic_class == "use_vision":
            self.ppo_cfg.policy.class_name = "ActorCriticCNN"
        if self.actor_critic_class == "use_resnet":
            self.ppo_cfg.policy.class_name = "ActorCriticResnet"
        if self.actor_critic_class == "use_rnn":
            self.ppo_cfg.policy.class_name = "ActorCriticCNNRecurrent"

        log.info("================================================")
        log.info(f"USING {self.actor_critic_class} ACTOR CRITIC CLASS")
        log.info("================================================")

        # training runtime highly relevant
        self.robot = "g1_static_dex1"
        self.num_envs = 64
        self.enable_opencv_display = True
        self.use_vision = True
        self.use_fixed_gazing = True


        if "pixel_norm_at_cube" in self.reward_weights:
            self.cameras[0].data_types.append("semantic_seg")