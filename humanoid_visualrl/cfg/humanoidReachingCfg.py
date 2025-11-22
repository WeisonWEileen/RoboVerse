from humanoid_visualrl.cfg.humanoidVisualRLCfg import BaseTableHumanoidTaskCfg
from metasim.utils import configclass
import torch

@configclass(name="reaching")
class HumanoidReachingCfg(BaseTableHumanoidTaskCfg):
    """Configuration for reaching tasks."""

    command_ranges = BaseTableHumanoidTaskCfg.CommandRanges(
        lin_vel_x=[-0, 0], lin_vel_y=[-0, 0], ang_vel_yaw=[-0, 0], heading=[-0, 0]
    )
    task_name = "reaching"
    reward_weights = {
        "wrist_pos": 5,
        "feet_distance": 0.5,
        "upper_body_pos": 0.5,
        "default_joint_pos": 0.5,
        "orientation": 1.0,
        "torques": -1e-5,
        "dof_vel": -5e-4,
        "dof_acc": -1e-7,
    }
    command_dim = 14
    actor_critic_class = "ActorCritic"
    env_spacing = 5
    robot = 'g1_pp_comp'
    wandb_project = "humanoid_juggling"



    init_states = [
        {
            "robots": {
                "g1_pp_comp": {
                    "pos": torch.tensor([0.0, 0.0, 0.78]),
                    "rot": torch.tensor([0.8, 0.0, 0.0, 0.0]),
                    "dof_pos": {
                        "left_hip_pitch_joint": -0.1,
                        "left_hip_roll_joint": 0,
                        "left_hip_yaw_joint": 0.0,
                        "left_knee_joint": 0.4,
                        "left_ankle_pitch_joint": -0.2,
                        "left_ankle_roll_joint": 0,
                        "right_hip_pitch_joint": -0.4,
                        "right_hip_roll_joint": 0,
                        "right_hip_yaw_joint": 0.0,
                        "right_knee_joint": 0.4,
                        "right_ankle_pitch_joint": -0.2,
                        "right_ankle_roll_joint": 0,
                        "waist_yaw_joint": 0.0,
                        "waist_roll_joint": 0.0,
                        "waist_pitch_joint": 0.0,
                        "right_shoulder_pitch_joint": 0.0,
                        "right_shoulder_roll_joint": 0.0,
                        "right_shoulder_yaw_joint": 0.0,
                        "right_elbow_joint": 0.0,
                        "xl330_joint": 0.0,
                        "d455_joint": 0.0,

                    },
                },
            },
            "objects": {},
        },
    ]

    mask_joint_names = ["xl330_joint", "d455_joint"]

    active_contact_sensor = True

    def __post_init__(self):
        super().__post_init__()
        self.num_actions = 21  #
        self.num_single_obs: int = 3 * self.num_actions + 6 + self.command_dim  #
        self.num_observations: int = int(self.frame_stack * self.num_single_obs)
        self.single_num_privileged_obs: int = 3 * self.num_actions + 58
        self.num_privileged_obs = int(self.c_frame_stack * self.single_num_privileged_obs)
        self.command_ranges.wrist_max_radius = 0.15
        self.command_ranges.l_wrist_pos_x = [-0.05, 0.15]
        self.command_ranges.l_wrist_pos_y = [-0.05, 0.15]
        self.command_ranges.l_wrist_pos_z = [-0.15, 0.15]
        self.command_ranges.r_wrist_pos_x = [-0.05, 0.15]
        self.command_ranges.r_wrist_pos_y = [-0.15, 0.05]
        self.command_ranges.r_wrist_pos_z = [-0.15, 0.15]

        # self.ppo_cfg.logger = None

        self.num_single_obs: int = 3 * self.num_actions + 6 + self.command_dim  #
        self.num_observations: int = int(self.frame_stack * self.num_single_obs)