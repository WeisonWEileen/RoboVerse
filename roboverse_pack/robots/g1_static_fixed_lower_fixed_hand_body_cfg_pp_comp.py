"""table tennis robot"""
from __future__ import annotations

from dataclasses import MISSING
from typing import Literal

from metasim.scenario.robot import BaseActuatorCfg, RobotCfg
from metasim.utils import configclass


@configclass
class G1PpCompCfg(RobotCfg):
    name: str = "g1_pp_comp"
    # usd_path: str = "roboverse_data/robots/g1/xml/g1_29dof_lock_waist_rev_1_0_modified_lower_fixed.usd"
    # usd_path: str = "roboverse_data/robots/g1_inspire/g1_29dof_with_inspire_rev_1_0.usd"
    usd_path: str = "roboverse_data/robots/g1/g1_pp_comp.usd"
    xml_path: str = MISSING
    urdf_path: str = "roboverse_data/robots/g1/test_8_31.usd"
    enabled_gravity: bool = True
    fix_base_link: bool = False
    # fix_base_link: bool = False
    enabled_self_collisions: bool = True
    isaacgym_flip_visual_attachments: bool = False
    collapse_fixed_joints: bool = True

    actuators: dict[str, BaseActuatorCfg] = {
        "left_hip_pitch_joint": BaseActuatorCfg(stiffness=200, damping=5),
        "left_hip_roll_joint": BaseActuatorCfg(stiffness=150, damping=5),
        "left_hip_yaw_joint": BaseActuatorCfg(stiffness=150, damping=5),
        "left_knee_joint": BaseActuatorCfg(stiffness=200, damping=5),
        "left_ankle_pitch_joint": BaseActuatorCfg(stiffness=20, damping=4),
        "left_ankle_roll_joint": BaseActuatorCfg(stiffness=20, damping=4),
        "right_hip_pitch_joint": BaseActuatorCfg(stiffness=200, damping=5),
        "right_hip_roll_joint": BaseActuatorCfg(stiffness=150, damping=5),
        "right_hip_yaw_joint": BaseActuatorCfg(stiffness=150, damping=5),
        "right_knee_joint": BaseActuatorCfg(stiffness=200, damping=5),
        "right_ankle_pitch_joint": BaseActuatorCfg(stiffness=20, damping=4),
        "right_ankle_roll_joint": BaseActuatorCfg(stiffness=20, damping=4),
        # "waist_yaw_joint": BaseActuatorCfg(stiffness=400, damping=5),
        # "waist_roll_joint": BaseActuatorCfg(stiffness=400, damping=5),
        # "waist_pitch_joint": BaseActuatorCfg(stiffness=400, damping=5),
        "waist_yaw_joint": BaseActuatorCfg(stiffness=400, damping=5),
        # todo: make it even smaller
        "waist_roll_joint": BaseActuatorCfg(stiffness=400, damping=5),
        "waist_pitch_joint": BaseActuatorCfg(stiffness=400, damping=5),
        # "left_shoulder_pitch_joint": BaseActuatorCfg(stiffness=40, damping=10),
        # "left_shoulder_roll_joint": BaseActuatorCfg(stiffness=40, damping=10),
        # "left_shoulder_yaw_joint": BaseActuatorCfg(stiffness=40, damping=10),
        # "left_elbow_joint": BaseActuatorCfg(stiffness=4, damping=0.2),
        # "left_wrist_roll_joint": BaseActuatorCfg(stiffness=4, damping=0.2),
        # "left_wrist_pitch_joint": BaseActuatorCfg(stiffness=4, damping=0.2),
        # "left_wrist_yaw_joint": BaseActuatorCfg(stiffness=4, damping=0.2),
        "right_shoulder_pitch_joint": BaseActuatorCfg(stiffness=40, damping=10),
        "right_shoulder_roll_joint": BaseActuatorCfg(stiffness=40, damping=10),
        "right_shoulder_yaw_joint": BaseActuatorCfg(stiffness=40, damping=10),
        "right_elbow_joint": BaseActuatorCfg(stiffness=4, damping=0.2),
        # "right_wrist_roll_joint": BaseActuatorCfg(stiffness=4, damping=0.2),
        # "right_wrist_yaw_joint": BaseActuatorCfg(stiffness=4, damping=0.2),
        # "right_wrist_pitch_joint": BaseActuatorCfg(stiffness=4,
        # damping=0.2),
        # leg
        "xl330_joint": BaseActuatorCfg(stiffness=40, damping=10),
        "d455_joint": BaseActuatorCfg(stiffness=40, damping=10),
    }


    joint_limits: dict[str, tuple[float, float]] = {
        "left_hip_pitch_joint": (-2.5307, 2.8798),
        "left_hip_roll_joint": (-0.5236, 2.9671),
        "left_hip_yaw_joint": (-2.7576, 2.7576),
        "left_knee_joint": (-0.087267, 2.8798),
        "left_ankle_pitch_joint": (-0.87267, 0.5236),
        "left_ankle_roll_joint": (-0.2618, 0.2618),
        "right_hip_pitch_joint": (-2.5307, 2.8798),
        "right_hip_roll_joint": (-2.9671, 0.5236),
        "right_hip_yaw_joint": (-2.7576, 2.7576),
        "right_knee_joint": (-0.087267, 2.8798),
        "right_ankle_pitch_joint": (-0.87267, 0.5236),
        "right_ankle_roll_joint": (-0.2618, 0.2618),
        "waist_yaw_joint": (-2.618, 2.618),
        "waist_roll_joint": (-0.52, 0.52),
        "waist_pitch_joint": (-0.52, 0.52),
        # "left_shoulder_pitch_joint": (-3.0892, 2.6704),
        # "left_shoulder_roll_joint": (-1.5882, 2.2515),
        # "left_shoulder_yaw_joint": (-2.618, 2.618),
        # "left_elbow_joint": (-1.0472, 2.0944),
        # # "left_wrist_pitch_joint": (-1.61, 1.61),
        # "left_wrist_roll_joint": (-1.97, 1.97),
        # "left_wrist_yaw_joint": (-1.61, 1.61),
        "right_shoulder_pitch_joint": (-3.0892, 2.6704),
        "right_shoulder_roll_joint": (-2.2515, 1.5882),
        "right_shoulder_yaw_joint": (-2.618, 2.618),
        "right_elbow_joint": (-1.0472, 2.0944),
        # "right_wrist_pitch_joint": (-1.61, 1.61),
        # "right_wrist_roll_joint": (-1.97, 1.97),
        # "right_wrist_yaw_joint": (-1.61, 1.61),
        "xl330_joint":(-0.86, 0.86),
        "d455_joint":(-1.57, 0.34)
    }

    torque_limits: dict[str, float] = {  # = target angles [rad] when action = 0.0
    
        "left_hip_pitch_joint": 88,
        "left_hip_roll_joint": 139,
        "left_hip_yaw_joint": 88,
        "left_knee_joint": 139,
        "left_ankle_pitch_joint": 50,
        "left_ankle_roll_joint": 50,
        "right_hip_pitch_joint": 88,
        "right_hip_roll_joint": 139,
        "right_hip_yaw_joint": 88,
        "right_knee_joint": 139,
        "right_ankle_pitch_joint": 50,
        "right_ankle_roll_joint": 50,
        "waist_yaw_joint": 88,
        "waist_pitch_joint": 55,
        "waist_roll_joint": 55,
        # "left_shoulder_pitch_joint": 25,
        # "left_shoulder_roll_joint": 25,
        # "left_shoulder_yaw_joint": 25,
        # "left_elbow_joint": 25,
        # "left_wrist_pitch_joint": 25,
        # "left_wrist_roll_joint": 25,
        # "left_wrist_yaw_joint": 25,
        "right_shoulder_pitch_joint": 25,
        "right_shoulder_roll_joint": 25,
        "right_shoulder_yaw_joint": 25,
        "right_elbow_joint": 25,
        # "right_wrist_pitch_joint": 25,
        # "right_wrist_roll_joint": 25,
        # "right_wrist_yaw_joint": 25,
        "xl330_joint": 25,
        "d455_joint": 25,
    }

    default_joint_positions: dict[str, float] = {  # = target angles [rad] when action = 0.0
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
        "waist_pitch_joint": 0.0,
        "waist_roll_joint": 0.0,
        # "left_wrist_pitch_joint": 0.0,
        # "left_wrist_roll_joint": 0.0,
        # "left_wrist_yaw_joint": 0.0,
        "right_shoulder_pitch_joint": 0.0,
        "right_shoulder_roll_joint": 0.0,
        "right_shoulder_yaw_joint": 0.0,
        "right_elbow_joint": 0.0,
        # "right_wrist_pitch_joint": 0.0,
        # "right_wrist_roll_joint": 0.0,
        # "right_wrist_yaw_joint": 0.0,
        "xl330_joint": 0.0,
        "d455_joint": 0.0,
    }

    control_type: dict[str, Literal["position", "effort"]] = {
        "left_hip_pitch_joint": "effort",
        "left_hip_roll_joint": "effort",
        "left_hip_yaw_joint": "effort",
        "left_knee_joint": "effort",
        "left_ankle_pitch_joint": "effort",
        "left_ankle_roll_joint": "effort",
        "right_hip_pitch_joint": "effort",
        "right_hip_roll_joint": "effort",
        "right_hip_yaw_joint": "effort",
        "right_knee_joint": "effort",
        "right_ankle_pitch_joint": "effort",
        "right_ankle_roll_joint": "effort",
        "waist_yaw_joint": "effort",
        "waist_pitch_joint": "effort",
        "waist_roll_joint": "effort",
        # "left_wrist_pitch_joint": "effort",
        # "left_wrist_roll_joint": "effort",
        # "left_wrist_yaw_joint": "effort",
        "right_shoulder_pitch_joint": "effort",
        "right_shoulder_roll_joint": "effort",
        "right_shoulder_yaw_joint": "effort",
        "right_elbow_joint": "effort",
        # "right_wrist_pitch_joint": "effort",
        # "right_wrist_roll_joint": "effort",
        # "right_wrist_yaw_joint": "effort",
        "xl330_joint": "effort",
        "d455_joint": "effort",
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

    upper_body_joints = [
        "shoulder",
        "elbow",
        "torso",
        "waist_pitch_joint",
        "waist_roll_joint",
        "wrist",
    ]


    num_joints: int = len(actuators)