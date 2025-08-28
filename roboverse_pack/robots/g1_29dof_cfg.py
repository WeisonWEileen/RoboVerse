from __future__ import annotations

from dataclasses import MISSING
from typing import Literal

from metasim.scenario.robot import BaseActuatorCfg, RobotCfg
from metasim.utils import configclass


@configclass
class G1Cfg(RobotCfg):
    name: str = "g1"
    num_joints: int = 29
    # usd_path: str = "roboverse_data/robots/g1/urdf/g1_29dof_lock_waist_rev_1_0_modified.usd"
    usd_path: str = "roboverse_data/robots/g1/g1_29dof.usd"
    xml_path: str = MISSING
    urdf_path: str = "roboverse_data/robots/g1/urdf/g1_29dof_lock_waist_rev_1_0_modified.urdf"
    enabled_gravity: bool = True
    fix_base_link: bool = False
    enabled_self_collisions: bool = False
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
        "waist_yaw_joint": BaseActuatorCfg(stiffness=400, damping=5),
        "left_shoulder_pitch_joint": BaseActuatorCfg(stiffness=40, damping=10),
        "left_shoulder_roll_joint": BaseActuatorCfg(stiffness=40, damping=10),
        "left_shoulder_yaw_joint": BaseActuatorCfg(stiffness=40, damping=10),
        "left_elbow_joint": BaseActuatorCfg(stiffness=40, damping=10),
        "left_wrist_roll_joint": BaseActuatorCfg(stiffness=40, damping=10),
        "right_shoulder_pitch_joint": BaseActuatorCfg(stiffness=40, damping=10),
        "right_shoulder_roll_joint": BaseActuatorCfg(stiffness=40, damping=10),
        "right_shoulder_yaw_joint": BaseActuatorCfg(stiffness=40, damping=10),
        "right_elbow_joint": BaseActuatorCfg(stiffness=40, damping=10),
        "right_wrist_roll_joint": BaseActuatorCfg(stiffness=40, damping=10),
        "left_wrist_pitch_joint": BaseActuatorCfg(stiffness=40, damping=10),
        "right_wrist_pitch_joint": BaseActuatorCfg(stiffness=40, damping=10),
        "left_wrist_yaw_joint": BaseActuatorCfg(stiffness=40, damping=10),
        "right_wrist_yaw_joint": BaseActuatorCfg(stiffness=40, damping=10),
        "waist_pitch_joint": BaseActuatorCfg(stiffness=400, damping=50),
        "waist_roll_joint": BaseActuatorCfg(stiffness=400, damping=5),
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
        "left_shoulder_pitch_joint": (-3.0892, 2.6704),
        "left_shoulder_roll_joint": (-1.5882, 2.2515),
        "left_shoulder_yaw_joint": (-2.618, 2.618),
        "left_elbow_joint": (-1.0472, 2.0944),
        "right_shoulder_pitch_joint": (-3.0892, 2.6704),
        "right_shoulder_roll_joint": (-2.2515, 1.5882),
        "right_shoulder_yaw_joint": (-2.618, 2.618),
        "right_elbow_joint": (-1.0472, 2.0944),
        "left_wrist_pitch_joint": (-0.87267, 0.5236),
        "right_wrist_pitch_joint": (-0.87267, 0.5236),
        "left_wrist_yaw_joint": (-0.87267, 0.5236),
        "right_wrist_yaw_joint": (-0.87267, 0.5236),
        "waist_pitch_joint": (-0.52267, 0.5236),
        "waist_roll_joint": (-0.52267, 0.5236),
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
        "left_shoulder_pitch_joint": 25,
        "left_shoulder_roll_joint": 25,
        "left_shoulder_yaw_joint": 25,
        "left_elbow_joint": 25,
        "left_wrist_roll_joint": 25,
        "right_shoulder_pitch_joint": 25,
        "right_shoulder_roll_joint": 25,
        "right_shoulder_yaw_joint": 25,
        "right_elbow_joint": 25,
        "right_wrist_roll_joint": 25,
        "left_wrist_pitch_joint": 5,
        "right_wrist_pitch_joint": 5,
        "left_wrist_yaw_joint": 5,
        "right_wrist_yaw_joint": 5,
        "waist_pitch_joint": 50,
        "waist_roll_joint": 50,
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
        "left_shoulder_pitch_joint": 0.0,
        "left_shoulder_roll_joint": 0.0,
        "left_shoulder_yaw_joint": 0.0,
        "left_elbow_joint": 0.0,
        "left_wrist_roll_joint": 0.0,
        "right_shoulder_pitch_joint": 0.0,
        "right_shoulder_roll_joint": 0.0,
        "right_shoulder_yaw_joint": 0.0,
        "right_elbow_joint": 0.0,
        "right_wrist_roll_joint": 0.0,
        "left_wrist_pitch_joint": 0.0,
        "right_wrist_pitch_joint": 0.0,
        "left_wrist_yaw_joint": 0.0,
        "right_wrist_yaw_joint": 0.0,
        "waist_pitch_joint": 0.0,
        "waist_roll_joint": 0.0,
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
        "left_shoulder_pitch_joint": "effort",
        "left_shoulder_roll_joint": "effort",
        "left_shoulder_yaw_joint": "effort",
        "left_elbow_joint": "effort",
        "right_shoulder_pitch_joint": "effort",
        "right_shoulder_roll_joint": "effort",
        "right_shoulder_yaw_joint": "effort",
        "right_elbow_joint": "effort",
        "right_wrist_roll_joint": "effort",
        "left_wrist_pitch_joint": "effort",
        "right_wrist_pitch_joint": "effort",
        "left_wrist_yaw_joint": "effort",
        "right_wrist_yaw_joint": "effort",
        "waist_pitch_joint": "effort",
        "waist_roll_joint": "effort",
    }

    # rigid body name substrings, to find indices of different rigid bodies.
    feet_links: list[str] = ["ankle_roll"]
    knee_links: list[str] = ["knee"]
    elbow_links: list[str] = ["elbow"]
    wrist_links: list[str] = ["left_wrist_yaw_link", "right_wrist_yaw_link"]  # TODO to be checked
    torso_links: list[str] = ["torso_link"]
    terminate_contacts_links = ["pelvis", "torso", "waist", "shoulder", "wrist"]
    penalized_contacts_links: list[str] = ["hip", "knee"]

    # joint substrings, to find indices of joints.

    left_yaw_roll_joints = ["left_hip_yaw_joint", "left_hip_roll_joint"]
    right_yaw_roll_joints = ["right_hip_yaw_joint", "right_hip_roll_joint"]
    upper_body_joints = ["shoulder", "elbow", "torso", "waist_pitch_joint", "waist_roll_joint"]



    body_names = [
            "pelvis",
            "left_hip_pitch_link",
            "left_hip_roll_link",
            "left_hip_yaw_link",
            "left_knee_link",
            "left_ankle_pitch_link",
            "left_ankle_roll_link",
            "right_hip_pitch_link",
            "right_hip_roll_link",
            "right_hip_yaw_link",
            "right_knee_link",
            "right_ankle_pitch_link",
            "right_ankle_roll_link",
            "waist_yaw_link",
            "waist_roll_link",
            "torso_link",
            "left_shoulder_pitch_link",
            "left_shoulder_roll_link",
            "left_shoulder_yaw_link",
            "left_elbow_link",
            "left_wrist_roll_link",
            "left_wrist_pitch_link",
            "left_wrist_yaw_link",
            "right_shoulder_pitch_link",
            "right_shoulder_roll_link",
            "right_shoulder_yaw_link",
            "right_elbow_link",
            "right_wrist_roll_link",
            "right_wrist_pitch_link",
            "right_wrist_yaw_link",
        ]