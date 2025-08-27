from humanoid_visualrl.cfg.humanoidVisualRLCfg import BaseTableHumanoidTaskCfg
from metasim.utils import configclass


@configclass
class HumanoidReachingCfg(BaseTableHumanoidTaskCfg):
    """Configuration for reaching tasks."""

    command_ranges = BaseTableHumanoidTaskCfg.CommandRanges(
        lin_vel_x=[-0, 0], lin_vel_y=[-0, 0], ang_vel_yaw=[-0, 0], heading=[-0, 0]
    )
    task_name = "reaching"
    logger = None

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

    def __post_init__(self):
        super().__post_init__()
        self.num_single_obs: int = 3 * self.num_actions + 6 + self.command_dim  #
        self.num_observations: int = int(self.frame_stack * self.num_single_obs)
        self.single_num_privileged_obs: int = 3 * self.num_actions + 60
        self.num_privileged_obs = int(self.c_frame_stack * self.single_num_privileged_obs)
        self.command_ranges.wrist_max_radius = 0.15
        self.command_ranges.l_wrist_pos_x = [-0.05, 0.15]
        self.command_ranges.l_wrist_pos_y = [-0.05, 0.15]
        self.command_ranges.l_wrist_pos_z = [-0.15, 0.15]
        self.command_ranges.r_wrist_pos_x = [-0.05, 0.15]
        self.command_ranges.r_wrist_pos_y = [-0.15, 0.05]
        self.command_ranges.r_wrist_pos_z = [-0.15, 0.15]