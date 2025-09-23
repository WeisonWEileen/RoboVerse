# roboverse_data/robots/T1/cfgs/t1_booster_cfg.py
from __future__ import annotations

from dataclasses import MISSING
from typing import Literal

from metasim.scenario.robot import BaseActuatorCfg, RobotCfg
from metasim.utils import configclass


@configclass
class T1Cfg(RobotCfg):
    # --------------------------------------------------------------------- #
    # 基本信息
    # --------------------------------------------------------------------- #
    name: str = "t1"
    num_joints: int = 23

    usd_path: str = "roboverse_data/robots/T1/T1_test_1.usd"
    xml_path: str = MISSING
    urdf_path: str = MISSING

    enabled_gravity: bool = True
    fix_base_link: bool = False
    enabled_self_collisions: bool = False
    collapse_fixed_joints: bool = True
    isaacgym_flip_visual_attachments: bool = False

    # --------------------------------------------------------------------- #
    # 执行器与控制参数（已对齐 isaaclab 配置）
    # --------------------------------------------------------------------- #
    actuators: dict[str, BaseActuatorCfg] = {
        # 头部（isaaclab未定义，设为0）
        "AAHead_yaw": BaseActuatorCfg(stiffness=0, damping=0),
        "Head_pitch": BaseActuatorCfg(stiffness=0, damping=0),
        # 左臂
        "Left_Shoulder_Pitch": BaseActuatorCfg(stiffness=40, damping=10),
        "Left_Shoulder_Roll": BaseActuatorCfg(stiffness=40, damping=10),
        "Left_Elbow_Pitch": BaseActuatorCfg(stiffness=40, damping=10),
        "Left_Elbow_Yaw": BaseActuatorCfg(stiffness=40, damping=10),
        # 右臂
        "Right_Shoulder_Pitch": BaseActuatorCfg(stiffness=40, damping=10),
        "Right_Shoulder_Roll": BaseActuatorCfg(stiffness=40, damping=10),
        "Right_Elbow_Pitch": BaseActuatorCfg(stiffness=40, damping=10),
        "Right_Elbow_Yaw": BaseActuatorCfg(stiffness=40, damping=10),
        # 躯干
        "Waist": BaseActuatorCfg(stiffness=200, damping=5),
        # 左腿
        "Left_Hip_Pitch": BaseActuatorCfg(stiffness=200, damping=5),
        "Left_Hip_Roll": BaseActuatorCfg(stiffness=200, damping=5),
        "Left_Hip_Yaw": BaseActuatorCfg(stiffness=200, damping=5),
        "Left_Knee_Pitch": BaseActuatorCfg(stiffness=200, damping=5),
        "Left_Ankle_Pitch": BaseActuatorCfg(stiffness=50, damping=1),
        "Left_Ankle_Roll": BaseActuatorCfg(stiffness=50, damping=1),
        # 右腿
        "Right_Hip_Pitch": BaseActuatorCfg(stiffness=200, damping=5),
        "Right_Hip_Roll": BaseActuatorCfg(stiffness=200, damping=5),
        "Right_Hip_Yaw": BaseActuatorCfg(stiffness=200, damping=5),
        "Right_Knee_Pitch": BaseActuatorCfg(stiffness=200, damping=5),
        "Right_Ankle_Pitch": BaseActuatorCfg(stiffness=50, damping=1),
        "Right_Ankle_Roll": BaseActuatorCfg(stiffness=50, damping=1),
    }

    # --------------------------------------------------------------------- #
    # 关节角度限制（不变，保持 MJCF 对齐）
    # --------------------------------------------------------------------- #
    joint_limits: dict[str, tuple[float, float]] = {
        "AAHead_yaw": (-1.57, 1.57),
        "Head_pitch": (-0.35, 1.22),
        "Left_Shoulder_Pitch": (-3.31, 1.22),
        "Left_Shoulder_Roll": (-1.74, 1.57),
        "Left_Elbow_Pitch": (-2.27, 2.27),
        "Left_Elbow_Yaw": (-2.44, 0.0),
        "Right_Shoulder_Pitch": (-3.31, 1.22),
        "Right_Shoulder_Roll": (-1.57, 1.74),
        "Right_Elbow_Pitch": (-2.27, 2.27),
        "Right_Elbow_Yaw": (0.0, 2.44),
        "Waist": (-1.57, 1.57),
        "Left_Hip_Pitch": (-1.8, 1.57),
        "Left_Hip_Roll": (-0.2, 1.57),
        "Left_Hip_Yaw": (-1.0, 1.0),
        "Left_Knee_Pitch": (0.0, 2.34),
        "Left_Ankle_Pitch": (-0.87, 0.35),
        "Left_Ankle_Roll": (-0.44, 0.44),
        "Right_Hip_Pitch": (-1.8, 1.57),
        "Right_Hip_Roll": (-1.57, 0.2),
        "Right_Hip_Yaw": (-1.0, 1.0),
        "Right_Knee_Pitch": (0.0, 2.34),
        "Right_Ankle_Pitch": (-0.87, 0.35),
        "Right_Ankle_Roll": (-0.44, 0.44),
    }

    # --------------------------------------------------------------------- #
    # 力矩限制（已对齐 isaaclab effort_limit_sim）
    # --------------------------------------------------------------------- #
    torque_limits: dict[str, float] = {
        "AAHead_yaw": 0,
        "Head_pitch": 0,
        "Waist": 30,
        "Left_Shoulder_Pitch": 18,
        "Left_Shoulder_Roll": 18,
        "Left_Elbow_Pitch": 18,
        "Left_Elbow_Yaw": 18,
        "Right_Shoulder_Pitch": 18,
        "Right_Shoulder_Roll": 18,
        "Right_Elbow_Pitch": 18,
        "Right_Elbow_Yaw": 18,
        "Left_Hip_Pitch": 45,
        "Left_Hip_Roll": 30,
        "Left_Hip_Yaw": 30,
        "Left_Knee_Pitch": 60,
        "Left_Ankle_Pitch": 24,
        "Left_Ankle_Roll": 15,
        "Right_Hip_Pitch": 45,
        "Right_Hip_Roll": 30,
        "Right_Hip_Yaw": 30,
        "Right_Knee_Pitch": 60,
        "Right_Ankle_Pitch": 24,
        "Right_Ankle_Roll": 15,
    }

    # --------------------------------------------------------------------- #
    # 默认关节姿态（已对齐 isaaclab init_state）
    # --------------------------------------------------------------------- #
    default_joint_positions: dict[str, float] = {
        "AAHead_yaw": 0.0,
        "Head_pitch": 0.0,
        "Left_Shoulder_Pitch": 0.20,
        "Left_Shoulder_Roll": -1.35,
        "Left_Elbow_Pitch": 0.0,
        "Left_Elbow_Yaw": -0.50,
        # "Right_Shoulder_Pitch": 0.435,
        # "Right_Shoulder_Roll": 0.95,
        "Right_Shoulder_Pitch": -0.68,
        "Right_Shoulder_Roll": 1.49,
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
    }

    control_type: dict[str, Literal["position", "effort"]] = {k: "effort" for k in joint_limits.keys()}

    feet_links: list[str] = ["foot_link"]
    knee_links: list[str] = ["knee"]
    elbow_links: list[str] = ["elbow"]
    wrist_links: list[str] = ["hand_link", "wrist"]
    torso_links: list[str] = ["Trunk", "Waist"]
    terminate_contacts_links: list[str] = ["pelvis", "torso", "waist", "shoulder", "elbow", "wrist"]
    penalized_contacts_links: list[str] = ["hip", "knee"]

    upper_body_joints = ["Shoulder", "Elbow", "AAHead", "Head_pitch", "Waist", "wrist"]
