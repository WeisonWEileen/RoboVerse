from __future__ import annotations

from dataclasses import MISSING
from typing import Literal

from metasim.scenario.robot import BaseActuatorCfg, RobotCfg
from metasim.utils import configclass


@configclass
class G1StaticCfg(RobotCfg):
    name: str = "g1_static"
    num_joints: int = 9
    usd_path: str = "roboverse_data/robots/g1/xml/g1_29dof_lock_waist_rev_1_0_modified_lower_fixed.usd"
    xml_path: str = MISSING
    urdf_path: str = "roboverse_data/robots/g1/xml/g1_29dof_lock_waist_rev_1_0_modified_lower_fixed.usd"
    enabled_gravity: bool = True
    fix_base_link: bool = True
    # fix_base_link: bool = False
    enabled_self_collisions: bool = False
    isaacgym_flip_visual_attachments: bool = False
    collapse_fixed_joints: bool = True

    actuators: dict[str, BaseActuatorCfg] = {
        "waist_yaw_joint": BaseActuatorCfg(stiffness=400, damping=5),
        "left_shoulder_pitch_joint": BaseActuatorCfg(stiffness=40, damping=10),
        "left_shoulder_roll_joint": BaseActuatorCfg(stiffness=40, damping=10),
        "left_shoulder_yaw_joint": BaseActuatorCfg(stiffness=40, damping=10),
        "left_elbow_joint": BaseActuatorCfg(stiffness=40, damping=10),
        "right_shoulder_pitch_joint": BaseActuatorCfg(stiffness=40, damping=10),
        "right_shoulder_roll_joint": BaseActuatorCfg(stiffness=40, damping=10),
        "right_shoulder_yaw_joint": BaseActuatorCfg(stiffness=40, damping=10),
        "right_elbow_joint": BaseActuatorCfg(stiffness=40, damping=10),
    }
    joint_limits: dict[str, tuple[float, float]] = {
        "waist_yaw_joint": (-2.618, 2.618),
        "left_shoulder_pitch_joint": (-3.0892, 2.6704),
        "left_shoulder_roll_joint": (-1.5882, 2.2515),
        "left_shoulder_yaw_joint": (-2.618, 2.618),
        "left_elbow_joint": (-1.0472, 2.0944),
        "right_shoulder_pitch_joint": (-3.0892, 2.6704),
        "right_shoulder_roll_joint": (-2.2515, 1.5882),
        "right_shoulder_yaw_joint": (-2.618, 2.618),
        "right_elbow_joint": (-1.0472, 2.0944),
    }

    torque_limits: dict[str, float] = {  # = target angles [rad] when action = 0.0
        "waist_yaw_joint": 88,
        "left_shoulder_pitch_joint": 25,
        "left_shoulder_roll_joint": 25,
        "left_shoulder_yaw_joint": 25,
        "left_elbow_joint": 25,
        "right_shoulder_pitch_joint": 25,
        "right_shoulder_roll_joint": 25,
        "right_shoulder_yaw_joint": 25,
        "right_elbow_joint": 25,
    }

    default_joint_positions: dict[str, float] = {  # = target angles [rad] when action = 0.0
        "waist_yaw_joint": 0.0,
        "left_shoulder_pitch_joint": 0.0,
        "left_shoulder_roll_joint": 0.0,
        "left_shoulder_yaw_joint": 0.0,
        "left_elbow_joint": 0.0,
        "right_shoulder_pitch_joint": 0.0,
        "right_shoulder_roll_joint": 0.0,
        "right_shoulder_yaw_joint": 0.0,
        "right_elbow_joint": 0.0,
    }

    control_type: dict[str, Literal["position", "effort"]] = {
        "waist_yaw_joint": "effort",
        "left_shoulder_pitch_joint": "effort",
        "left_shoulder_roll_joint": "effort",
        "left_shoulder_yaw_joint": "effort",
        "left_elbow_joint": "effort",
        "right_shoulder_pitch_joint": "effort",
        "right_shoulder_roll_joint": "effort",
        "right_shoulder_yaw_joint": "effort",
        "right_elbow_joint": "effort",
    }

    # rigid body name substrings, to find indices of different rigid bodies.
    feet_links: list[str] = ["ankle_roll"]
    knee_links: list[str] = ["knee"]
    elbow_links: list[str] = ["elbow"]
    wrist_links: list[str] = ["wrist_yaw_link"]
    torso_links: list[str] = ["torso_link"]
    terminate_contacts_links = ["pelvis", "torso", "waist", "shoulder", "elbow", "wrist"]
    penalized_contacts_links: list[str] = ["hip", "knee"]

    # joint substrings, to find indices of joints.

    upper_body_joints = ["shoulder", "elbow", "torso"]
