from __future__ import annotations

"""Base class for legged-gym style legged-robot tasks."""

from dataclasses import MISSING
from typing import Callable, Literal

import torch

from metasim.constants import PhysicStateType
from metasim.scenario.objects import PrimitiveCubeCfg, PrimitiveCylinderCfg
from metasim.scenario.robot import RobotCfg
from metasim.scenario.simulator_params import SimParamCfg
from metasim.types import TensorState
from metasim.utils import configclass
from loguru import logger as log

from metasim.scenario.objects import RigidObjCfg, ArticulationObjCfg


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
        rnn_hidden_dim = 256
        # vision_height = 240
        # vision_width = 320
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
        entropy_coef = 0.0001
        """Entropy coefficient."""
        num_learning_epochs = 5
        """Number of learning epochs."""
        num_mini_batches = 4  # batch size = 128 // 4 = 32  batch size = 64 // 2 = 32
        """mini batch size = num_envs*n_steps / num_mini_batches"""
        learning_rate = 1.0e-3
        # schedule = "adaptive"
        schedule = "fixed"
        # schedule = "momentum"
        gamma = 0.99
        lam = 0.95
        desired_kl = 0.01
        max_grad_norm = 1.0
        kl_clip_thres = 0.2
        class_name = "PPO"
        # good schime for touching
        # learning_rate_mlp = 1e-3
        # learning_rate_rnn = 1e-4
        # learning_rate_cnn = 5e-5

        # good schime for lifting
        learning_rate_mlp = 1e-4
        learning_rate_rnn = 1e-4
        learning_rate_cnn = 4e-5
        # normalize_advantage_per_mini_batch = False

        # mask = True
        # masks_ids = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]

    # class_name = "ActorCritic"
    """Policy class name."""
    algorithm_class_name = "PPO"
    """Algorithm class name."""
    # num_steps_per_env = 96  # *0.005*5*48 = 6 s
    num_steps_per_env = 48  # *0.005*5*48 = 6 s
    """per iteration"""
    max_iterations = 4000
    """max number of iterations"""

    # logging
    # logger: str = "wandb"
    wandb_project: str = "active_vision_vega"

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

    # decimation: int = 3
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
    active_contact_sensor: bool = True

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
    robot: str = "g1_static_inpire_left_fixed"
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
    env_spacing: float = 5
    """Environment spacing."""
    send_timeouts: bool = True
    """Whether to send time out information to the algorithm"""
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
        # replace_cylinder_with_capsule=True,
        friction_offset_threshold=0.04,
        num_threads=10,
    )
    """Simulation parameters with physics engine settings."""
    dt = decimation * sim_params.dt
    """simulation time step in s"""

    objects = ["33o1zhw3", "cube", "270o9y3w"]
    objects = [
        RigidObjCfg(
            name="table",
            scale=(1, 1, 1),  # work with g1
            # scale=(0.60, 0.2, 0.60), # work with vega
            physics=PhysicStateType.GEOM,
            usd_path="roboverse_data/scenes/tritable.usd",
            fix_base_link=True,
            # default_position=(0.0, 0.0, 0.65),
            default_position=(0.20, 0.0, 0.50),
            # default_orientation=(0.7071, 0.7071, 0.0000, 0.0000),
            default_orientation=(1.0, 0.0, 0.0, 0.0),
            # default_orientation=(0.7071, 0.0, 0.0, -0.7071),
            collision_enabled=True,
            # urdf_path="metasim/example/example_assets/bbq_sauce/urdf/bbq_sauce.urdf",
            # mjcf_path="metasim/example/example_assets/bbq_sauce/mjcf/bbq_sauce.xml",
        ),
        # RigidObjCfg(
        #     name="wall",
        #     scale=(4.0, 4.0, 1.8),
        #     physics=PhysicStateType.GEOM,
        #     usd_path="roboverse_data/wall.usd",
        #     fix_base_link=True,
        #     default_position=(0.0, 0.0, 0.3),
        #     collision_enabled=False,
        #     enable_gyroscopic_forces=False,
        # ),
        PrimitiveCubeCfg(
            name="object",
            size=(0.05, 0.05, 0.05),
            color=[1.0, 0.0, 0.0],
            physics=PhysicStateType.RIGIDBODY,
            collision_enabled=True,
            fix_base_link=False,
            default_position=(0.55, 0.1, 0.9 + 0.06 / 2 + 0.01),
            mass=0.1,  # 增加质量以确保更好的物理行为
        ),
        # PrimitiveCylinderCfg(
        #     name="object",
        #     radius=0.025,
        #     height=0.05,
        #     color=[1.0, 0.0, 0.0],
        #     collision_enabled=True,
        #     default_position=(0.55, 0.1, 0.9 + 0.05 / 2 + 0.01),
        #     mass=20.0,
        # ),
        # ArticulationObjCfg(
        #     name="box_base",
        #     fix_base_link=True,
        #     usd_path="roboverse_data/assets/rlbench/close_box/box_base/usd/box_base.usd",
        #     urdf_path="get_started/example_assets/box_base/urdf/box_base_unique.urdf",
        #     mjcf_path="get_started/example_assets/box_base/mjcf/box_base_unique.mjcf",
        #     # default_position=(0.3, 0.1, 0.951),
        #     default_position=(0.4, 0.2, 0.951),
        #     # rotate -90 degrees around y axis
        #     default_orientation=(0.7071, 0.0, -0.7071, 0.0),
        #     # default_orientation=(1.0, 0.0, 0.0, 0.0),
        # ),
        # RigidObjCfg(
        #     name="object",
        #     # size=(0.09, 0.09, 0.09),
        #     # color=[1.0, 0.0, 0.0],
        #     scale=(1.2, 1.2, 1.2),
        #     physics=PhysicStateType.RIGIDBODY,
        #     usd_path="roboverse_data/objects/visdex_objects/USD/2h0dnrqc/2h0dnrqc.usd",
        #     collision_enabled=True,
        #     fix_base_link=False,
        #     default_position=(0.3, 0.1, 0.951),
        #     enable_gyroscopic_forces=True,
        #     mass_density=500.0,
        #     randomize_material=True,
        # ),
    ]

    # cameras
    """objects in the environment"""
    traj_filepath = None
    """path to the trajectory file"""
    # TODO read form max_episode_length_s and divide s
    # max_episode_length_s: int = 6
    max_episode_length_s: int = 7
    """maximum episode length in seconds"""
    episode_length: int = 2400
    """episode length in steps"""
    max_episode_length: int = 2400
    """episode length in steps"""

    num_observations: int = 9
    num_actions: int = 9
    num_privileged_obs: int = 9
    max_episode_length: int = 2400
    randomize_obj_material: bool = False
    update_obj_material_step_interval: int = 96 * 100
    reset_fall_down_threshold = 0.54 - 0.04 / 2 # 0.54 is for height 0.8

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
#  0.9 + 0.06 / 2 + 0.01
    init_states = [
        {
            "objects": {
                # "cube": {
                "object": {
                    "pos": torch.tensor([0.52, 0.0, 0.50 +  0.06 / 2 + 0.01]),
                    "rot": torch.tensor([1.0, 0.0, 0.0, 0.0]),
                },
            },
            "robots": {},
        }
    ]

    command_dim = 14
    num_actions = 17 - 6
    torque_limit_scale = 1.0
    scale = 0.5
    reward_weights: dict[str, float] = {
        "pixel_norm_at_object": 0.4 * scale,
        "finger_close_to_object": 1.3 * scale,
        "lift_object": 50.0,
        # "right_arm_default_joint_pos": 0.1 * scale,
        "wrist_lower_than_table": 0.1 * scale,
        "contact_force": 1.0 * scale,
        # note that this should be behind contact_force, otherwise it will be not update
        "contact_force_upward": 0.25 * scale,
        # "stage": 1.0 * scale,
        "action_smoothness": -0.1 * scale,
        # "contact_force_two_much_penalty": -1.0,
        # "energy_consumption": -1e-7,
        # "see_object": 0.20,
        # "hand_to_object_dist": 1.0,
        # "wrist_close_to_object_and_grasp": 1.0,
        # "lift_object": 2.0,
        # "fuse_wrist_close_to_object_and_grasp": 0.5,
    }

    frame_stack = 1
    c_frame_stack = 1

    # obs
    visual_dim: int = 64

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
        # same as unitree isaacsim
        PinholeCameraCfg(
            name="camera_first_person",
            data_types=["rgb", "semantic_seg"],
            width=128,
            height=96,
            pos=(1.5, -1.5, 1.5),
            look_at=(0.0, 0.0, 0.0),
            mount_to="g1_static_dex1",
            mount_link="torso_link/d435_link",
            mount_pos=(0.0, 0.0, 0.0),
            mount_quat=(0.5, -0.5, 0.5, -0.5),
            focal_length=7.6,
            horizontal_aperture=20.0,
            clipping_range=(0.05, 3.),
        )
    ]

    randomization = True
    finetune = False
    enable_grasp = False

    mask_joint_names = [
        "left_elbow_joint",
        "left_shoulder_pitch_joint",
        "left_shoulder_roll_joint",
        "left_shoulder_yaw_joint",
        # "left_wrist_pitch_joint",
        # "left_wrist_roll_joint",
        # "left_wrist_yaw_joint",
        "right_elbow_joint",
        "right_shoulder_pitch_joint",
        "right_shoulder_roll_joint",
        "right_shoulder_yaw_joint",
        # "right_wrist_pitch_joint",
        # "right_wrist_roll_joint",
        # "right_wrist_yaw_joint",
    ]

    reward_hand_object_dist_exp_sharpness = 10.0
    reward_lift_offset = 0.2
    reward_lift_object_z = init_states[0]["objects"]["object"]["pos"][2] + reward_lift_offset
    reward_lift_object_exp_shapeness = 10.0
    reward_object2goal_exp_shapeness = 15
    reward_wrist_close_to_object_exp_sharpness = 4.0
    reward_pixel_norm_at_object_exp_sharpness = 50.0
    reward_improvement_ratio_threshold = 0.15

    randomize_material = False
    randomize_cfg = {
        "enable_floor": True,
        "enable_walls": True,
        "enable_ceiling": False,
        "floor_materials": ["roboverse_data/materials/arnold/Wood/Oak_Planks.mdl"],
        "light_randomize_freq": 100,
        "material_cfg": {
            "table": {
                "material_path": ["roboverse_data/materials/arnold/Wood/Walnut.mdl"],
            },
        },
        "randomize_cfg": {
            "light": {
                "intensity_range": (0.25, 2.0),
                "randomize_orientation": True,
                "randomize_position": True,
                "position_range": ((-100.0, 100.0), (-100.0, 100.0), (-0.0, 0.0)),
            }
        },
        "env_setting_randomize_freq": 1,
    }

    mode: Literal["train", "test"] = "train"
    occlude_cube = False
    occlude_cube_yaw_range = 0.8
    

    # initially give large mass to encourage contact, and then linearly decrease to 0.05
    curriculum_object_mass_flag = True
    curriculum_object_mass_range = (0.1, objects[1].mass)
    curriculum_object_mass_begin_iter = 0
    curriculum_object_mass_end_iter = 200
    curriculum_object_mass_update_interval = 200
    stage_finger_close_to_object_change_thres = 0.045



    def __post_init__(self):
        self.command_ranges.wrist_max_radius = 0.15
        # self.randomize_object_y_offset = 0.1
        self.randomize_object_curriculum = True
        self.randomize_add_scale = 0.005

        self.see_flag_his_win_length = 1000
        self.time_range_increase_curriculum = 0.01
        self.randomize_object_radius = self.init_states[0]["objects"]["object"]["pos"][0]
        self.randomize_object_radius_range = 0.0

        # if self.finetune:
        #     # for finetuning, use less frequent curriculum update and less yaw range
        #     # self.update_curriculum_iteration = 100
        #     # self.randomize_object_yaw_range = 1.8
        #     # self.randomize_object_yaw_range = 1.8
        #     self.randomize_object_yaw_range = 0.57
        #     self.curriculum_object_yaw = False

        #     self.reward_weights = {
        #         "pixel_norm_at_object": 1.4,
        #         "finger_close_to_object": 5.0,
        #         "grasp_binary": 50.0,
        #         "right_arm_default_joint_pos": 0.17,
        #     }
        #     if self.enable_grasp:
        #         self.reward_weights["grasp_binary"] = 5.0
        #         # reduce randomize
        #         self.randomize_object_radius_range = 0.05
        #         self.randomize_object_radius = self.randomize_object_radius - 0.04

        # else:
        # self.update_curriculum_iteration = 400
        # self.randomize_object_yaw_range = 2.3
        # self.randomize_object_yaw_range = 3.14
        # self.randomize_object_yaw_range = 3.06
        # self.randomize_object_yaw_range = 2.14
        self.curriculum_object_yaw = True
        self.curriculum_initial_object_yaw_range = 0.3
        self.randomize_object_yaw_range = 0.7
        self.warm_up_beforecurriculum = 1000  #  10000 / 96 =  104 iteration
        self.curriculum_avg_thres_higher = 0.93
        self.curriculum_avg_thres_lower = 0.85
        self.curriculum_randomize_iteration_interval = 100

        # self.randomize_object_radius = 0.85  # max
        # self.randomize_object_radius = 0.55
        # self.randomize_object_radius -= 0.07
        # if self.finetune:
        #     self.randomize_object_radius -= 0.07

        self.curriculum_object_yaw_stages = [0.2, 0.5, 1.0]  # Multipliers for randomize_object_yaw_range
        self.curriculum_object_yaw_thresholds = [0.8, 0.8]  # Success rate thresholds to advance stages
        self.curriculum_object_yaw_min_episodes = [500, 500]  # Minimum episodes before considering advancement

        # self.actor_critic_class = "use_rnn_foveated"
        # self.actor_critic_class = "use_rnn_foveated"

        if self.actor_critic_class == "use_vision":
            self.ppo_cfg.policy.class_name = "ActorCriticCNN"
        if self.actor_critic_class == "use_resnet":
            self.ppo_cfg.policy.class_name = "ActorCriticResnet"
        if self.actor_critic_class == "use_rnn":
            self.ppo_cfg.policy.class_name = "ActorCriticCNNRecurrent"
        if self.actor_critic_class == "use_rnn_foveated":
            self.ppo_cfg.policy.class_name = "ActorCriticCNNRAM"
        if self.actor_critic_class == "use_vit_rnn":
            self.ppo_cfg.policy.class_name = "ActorCriticViTRecurrent"

        log.info("================================================")
        log.info(f"USING {self.actor_critic_class} ACTOR CRITIC CLASS")
        log.info("================================================")

        # training runtime highly relevant
        # self.robot = "g1_static_dex1"
        self.robot = "vega"
        # self.robot = "g1_static_dex1_comp"
        # self.robot = "g1_static_inpire_left_fixed"
        self.num_envs = 96
        self.enable_opencv_display = True
        self.use_vision = True
        self.use_fixed_gazing = True
        # breakpoint()
        if "finger_close_to_object" in self.reward_weights:
            self.ppo_cfg.policy.masking_all = False

        else:
            self.ppo_cfg.policy.masking_all = True

        if self.robot == "vega":
            if "finger_close_to_object" in self.reward_weights and self.enable_grasp:
                self.mask_joint_names = []
            elif "finger_close_to_object" in self.reward_weights and not self.enable_grasp:
                # mask hand
                self.mask_joint_names = [
                    "R_th_j0",
                    "R_th_j1",
                    "R_ff_j1",
                    "R_mf_j1",
                    "R_rf_j1",
                    "R_lf_j1",
                ]
            else:
                self.mask_joint_names = [
                    "R_arm_j1",
                    "R_arm_j2",
                    "R_arm_j3",
                    "R_arm_j4",
                    "R_arm_j5",
                    "R_arm_j6",
                    "R_arm_j7",
                    "R_th_j0",
                    "R_th_j1",
                    "R_ff_j1",
                    "R_mf_j1",
                    "R_rf_j1",
                    "R_lf_j1",
                ]

        if self.robot in ["g1_static_dex1"]:
            self.init_states[0]["robots"] = {
                "g1_static_dex1": {
                    "pos": torch.tensor([0.0, 0.0, 0.60]),
                    "rot": torch.tensor([0.8, 0.0, 0.0, 0.0]),
                    "dof_pos": {
                        "waist_yaw_joint": 0.0,
                        "waist_roll_joint": 0.0,
                        "waist_pitch_joint": 0.0,
                        # "left_shoulder_pitch_joint": 0.0,
                        # "left_shoulder_roll_joint": 0.0,
                        # "left_shoulder_yaw_joint": 0.0,
                        # "left_elbow_joint": 0.0,
                        # "left_wrist_roll_joint": 0.0,
                        # "left_wrist_pitch_joint": 0.0,
                        # "left_wrist_yaw_joint": 0.0,
                        "right_shoulder_pitch_joint": 0.0,
                        "right_shoulder_roll_joint": 0.0,
                        "right_shoulder_yaw_joint": 0.0,
                        "right_elbow_joint": 0.0,
                        # "right_wrist_roll_joint": 0.0,
                        # "right_wrist_pitch_joint": 0.0,
                        # "right_wrist_yaw_joint": 0.0,
                    },
                },
            }

            self.cameras[0].mount_to = "g1_static_dex1"
            self.cameras[0].mount_link = "d435_link"
            # self.num_joints = 17 - 7
            self.num_joints = 11
        elif self.robot == "g1_static_dex1_comp":
            self.init_states[0]["robots"] = {
                "g1_static_dex1_comp": {
                    "pos": torch.tensor([0.0, 0.0, 0.60]),
                    "rot": torch.tensor([0.8, 0.0, 0.0, 0.0]),
                    "dof_pos": {
                        "waist_yaw_joint": 0.0,
                        "waist_roll_joint": 0.0,
                        "waist_pitch_joint": 0.0,
                        "right_shoulder_pitch_joint": 0.0,
                        "right_shoulder_roll_joint": 0.0,
                        "right_shoulder_yaw_joint": 0.0,
                        "right_elbow_joint": 0.0,
                    },
                },
            }
            self.cameras[0].mount_to = "g1_static_dex1_comp"
            self.cameras[0].mount_link = "d455_link"
            self.num_joints = 11
        elif self.robot == "g1_static_inpire_left_fixed":
            self.init_states[0]["robots"] = {
                "g1_static_inpire_left_fixed": {
                    "pos": torch.tensor([0.0, 0.0, 0.00]),
                    "rot": torch.tensor([0.8, 0.0, 0.0, 0.0]),
                    "dof_pos": {
                        "waist_yaw_joint": 0.0,
                        "waist_roll_joint": 0.0,
                        "waist_pitch_joint": 0.0,
                        # "left_shoulder_pitch_joint": 0.0,
                        # "left_shoulder_roll_joint": 0.0,
                        # "left_shoulder_yaw_joint": 0.0,
                        # "left_elbow_joint": 0.0,
                        # "left_wrist_roll_joint": 0.0,
                        # "left_wrist_pitch_joint": 0.0,
                        # "left_wrist_yaw_joint": 0.0,
                        "right_shoulder_pitch_joint": 0.0,
                        "right_shoulder_roll_joint": 0.0,
                        "right_shoulder_yaw_joint": 0.0,
                        "right_elbow_joint": 0.0,
                        "right_wrist_roll_joint": 0.0,
                        "right_wrist_pitch_joint": 0.0,
                        "right_wrist_yaw_joint": 0.0,
                        "R_index_proximal_joint": 0.0,
                        "R_index_intermediate_joint": 0.0,
                        "R_middle_proximal_joint": 0.0,
                        "R_middle_intermediate_joint": 0.0,
                        "R_pinky_proximal_joint": 0.0,
                        "R_pinky_intermediate_joint": 0.0,
                        "R_ring_proximal_joint": 0.0,
                        "R_ring_intermediate_joint": 0.0,
                        "R_thumb_proximal_yaw_joint": 0.0,
                        "R_thumb_proximal_pitch_joint": 0.0,
                        "R_thumb_intermediate_joint": 0.0,
                        "R_thumb_distal_joint": 0.0,
                    },
                },
            }
            self.cameras[0].mount_to = "g1_static_inpire_left_fixed"
            self.cameras[0].mount_link = "d435_link"
            self.num_joints = 29 - 7  # -
        elif self.robot in ["vega"]:
            self.cameras[0].mount_quat = (1.0, 0.0, 0.0, 0.0)
            self.cameras[0].mount_to = "vega"
            self.cameras[0].mount_link = "head_l3/zed_left_camera"
            SCALE_FACTOR = 6
            self.cameras[0].width = int(960 / SCALE_FACTOR)
            self.cameras[0].height = int(600 / SCALE_FACTOR)
            self.cameras[0].focal_length = 0.2112011909484863
            self.cameras[0].intrinsics = [
                365.5782165527344 / (SCALE_FACTOR * 0.5),
                0.0,
                494.15985107421875 / (SCALE_FACTOR * 0.5),
                0.0,
                365.5782165527344 / (SCALE_FACTOR * 0.5),
                301.70770263671875 / (SCALE_FACTOR * 0.5),
                0.0,
                0.0,
                1.0,
            ]
            self.init_states[0]["robots"] = {
                "vega": {
                    "pos": torch.tensor([0.0, 0.0, 0.06]),
                    # "rot": torch.tensor([1.0, 0.0, 0.0, 0.0]),
                    # same the torso 27 degress
                    # "rot": torch.tensor([0.9701373249726354 0.0, 0.24255632478857206, 0.0]),
                    "rot": torch.tensor([1.0, 0.0, 0.0, 0.0]),
                    "dof_pos": {
                        # "head_j1": 0.0,
                        "base_yaw_joint": 0.0,
                        "head_j2": 0.0,
                        "head_j3": 0.0,  # pitch\
                        # "L_arm_j2": 0.0,
                        # "L_arm_j3": 0.307,
                        # "L_arm_j4": -0.305,
                        # "L_arm_j5": -1.69,
                        # "L_arm_j6": 0.0,
                        # "L_arm_j7": -0.84,
                        # "torso_j1": 0.38,
                        # "torso_j2": 0.95,
                        # "torso_j3": 0.00,
                        # Right arm - neutral pose
                        "R_arm_j1": -2.06,
                        "R_arm_j2": -0.21,
                        "R_arm_j3": -0.13,
                        "R_arm_j4": -2.59,
                        "R_arm_j5": -0.3,
                        "R_arm_j6": 0.05,
                        "R_arm_j7": 0.0,
                        # Right hand - open
                        "R_th_j0": 1.47,
                        "R_th_j1": -0.10,
                        "R_ff_j1": 0.0,
                        "R_mf_j1": 0.0,
                        "R_rf_j1": 0.0,
                        "R_lf_j1": 0.0,
                        "R_ff_j2": 0.0,
                        "R_lf_j2": 0.0,
                        "R_mf_j2": 0.0,
                        "R_rf_j2": 0.0,
                        "R_th_j2": 0.0,
                    },
                },
            }
            self.num_joints = len(self.init_states[0]["robots"]["vega"]["dof_pos"])
            self.num_actions = 16  # 2 head, 7 arm, 6 hand, 1 wheel(yaw orientation)
            self.num_extra_actions = 1

        else:
            raise ValueError(f"Robot {self.robot} not supported")

        self.num_single_obs = self.num_joints * 2 + self.num_actions  # q. dq. actions
        self.num_observations: int = int(self.frame_stack * self.num_single_obs)
        self.single_num_privileged_obs: int = self.num_single_obs
        self.num_privileged_obs = int(self.c_frame_stack * self.single_num_privileged_obs)

        self.randomize_robot_yaw_range = 1.6

        self.ema_alpha = 0.03
        self.thres_radius = 28
        self.pixel_reward_offset = torch.exp(
            -torch.sqrt(torch.tensor([self.cameras[0].width ** 2 + self.cameras[0].height ** 2]))
            / 2.0
            / self.reward_pixel_norm_at_object_exp_sharpness
        )
        self.ema_reward_threshold = (
            torch.exp(torch.tensor([-self.thres_radius / self.reward_pixel_norm_at_object_exp_sharpness]))
            - self.pixel_reward_offset
        ).item()
        log.info(f"reward_threshold: {self.ema_reward_threshold}")

        self.seed = self.ppo_cfg.seed

        # self.filter_pairs = [
        #     (self.robot, "object"),
        # ]

        # randomize occlude cube material
        from metasim.randomization.presets.scene_presets import SceneMaterialCollections

        if self.occlude_cube:
            self.randomize_cfg["material_cfg"]["occlusion_cube"] = {
                "material_path": SceneMaterialCollections.wall_materials(),
            }
