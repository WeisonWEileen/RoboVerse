from humanoid_visualrl.cfg.humanoidVisualRLCfg import BaseTableHumanoidTaskCfg
from metasim.utils import configclass
import torch

@configclass(name="booster_reaching")
class HumanoidReachingCfg(BaseTableHumanoidTaskCfg):
    """Configuration for reaching tasks."""

    command_ranges = BaseTableHumanoidTaskCfg.CommandRanges(
        lin_vel_x=[-0, 0], lin_vel_y=[-0, 0], ang_vel_yaw=[-0, 0], heading=[-0, 0]
    )
    task_name = "reaching"
    robot = 't1'

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


        # self.ppo_cfg.logger = None

    init_states = [
        {
            "objects": {},
            "robots": {
                "t1": {
                    "pos": torch.tensor([0.0, 0.0, 0.50]),
                    "rot": torch.tensor([1.0, 0.0, 0.0, 0.0]),
                    "dof_pos": {
                        "AAHead_yaw": 0.0,
                        "Head_pitch": 0.0,
                        "Left_Shoulder_Pitch": 0.20,
                        "Left_Shoulder_Roll": -1.35,
                        "Left_Elbow_Pitch": 0.0,
                        "Left_Elbow_Yaw": -0.50,
                        "Right_Shoulder_Pitch": 0.435,
                        "Right_Shoulder_Roll": 0.95,
                        "Right_Elbow_Pitch": 0.0,
                        "Right_Elbow_Yaw": 0.55,
                        "Waist": 0.0,
                        "Left_Hip_Pitch": -0.20,
                        "Left_Hip_Roll": 0.0,
                        "Left_Hip_Yaw": 0.0,
                        "Left_Knee_Pitch": 0.42,
                        "Left_Ankle_Pitch": -0.23,
                        "Left_Ankle_Roll": 0.0,
                        "Right_Hip_Pitch": -0.20,
                        "Right_Hip_Roll": 0.0,
                        "Right_Hip_Yaw": 0.0,
                        "Right_Knee_Pitch": 0.42,
                        "Right_Ankle_Pitch": -0.23,
                        "Right_Ankle_Roll": 0.0,
                    },
                },
            },
        }
    ]

    def __post_init__(self):
        super().__post_init__()
        self.num_single_obs: int = 3 * self.num_actions + 6 + self.command_dim  #
        self.num_observations: int = int(self.frame_stack * self.num_single_obs)
        self.single_num_privileged_obs: int = 3 * self.num_actions + 58
        self.num_privileged_obs = int(self.c_frame_stack * self.single_num_privileged_obs)
        self.command_ranges.wrist_max_radius = 0.12
        self.command_ranges.l_wrist_pos_x = [-0.05, 0.15]
        self.command_ranges.l_wrist_pos_y = [-0.05, 0.15]
        self.command_ranges.l_wrist_pos_z = [-0.15, 0.15]
        self.command_ranges.r_wrist_pos_x = [-0.05, 0.15]
        self.command_ranges.r_wrist_pos_y = [-0.15, 0.05]
        self.command_ranges.r_wrist_pos_z = [-0.15, 0.15]