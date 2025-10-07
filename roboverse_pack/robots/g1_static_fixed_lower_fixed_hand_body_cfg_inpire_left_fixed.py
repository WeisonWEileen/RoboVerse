from __future__ import annotations

from dataclasses import MISSING
from typing import Literal

from metasim.scenario.robot import BaseActuatorCfg, RobotCfg
from metasim.utils import configclass


@configclass
class G1StaticInpireLeftFixedCfg(RobotCfg):
    name: str = "g1_static_inpire_left_fixed"
    usd_path: str = "roboverse_data/robots/g1_inspire/g1_29dof_with_inspire_rev_1_0_1.usd"
    xml_path: str = MISSING
    urdf_path: str = MISSING
    enabled_gravity: bool = True
    fix_base_link: bool = True
    # fix_base_link: bool = False
    enabled_self_collisions: bool = False
    isaacgym_flip_visual_attachments: bool = False
    collapse_fixed_joints: bool = True

    actuators: dict[str, BaseActuatorCfg] = {
        "waist_yaw_joint": BaseActuatorCfg(stiffness=80, damping=8),
        "waist_roll_joint": BaseActuatorCfg(stiffness=60, damping=5),
        "waist_pitch_joint": BaseActuatorCfg(stiffness=60, damping=5),
        "left_shoulder_pitch_joint": BaseActuatorCfg(stiffness=40, damping=10),
        "left_shoulder_roll_joint": BaseActuatorCfg(stiffness=40, damping=10),
        "left_shoulder_yaw_joint": BaseActuatorCfg(stiffness=40, damping=10),
        "left_elbow_joint": BaseActuatorCfg(stiffness=4, damping=0.2),
        "left_wrist_roll_joint": BaseActuatorCfg(stiffness=4, damping=0.2),
        "left_wrist_pitch_joint": BaseActuatorCfg(stiffness=4, damping=0.2),
        "left_wrist_yaw_joint": BaseActuatorCfg(stiffness=4, damping=0.2),
        "right_shoulder_pitch_joint": BaseActuatorCfg(stiffness=40, damping=10),
        "right_shoulder_roll_joint": BaseActuatorCfg(stiffness=40, damping=10),
        "right_shoulder_yaw_joint": BaseActuatorCfg(stiffness=40, damping=10),
        "right_elbow_joint": BaseActuatorCfg(stiffness=40, damping=10),
        "right_wrist_roll_joint": BaseActuatorCfg(stiffness=4, damping=0.2),
        "right_wrist_yaw_joint": BaseActuatorCfg(stiffness=4, damping=0.2),
        "right_wrist_pitch_joint": BaseActuatorCfg(stiffness=4, damping=0.2),
        "R_index_proximal_joint": BaseActuatorCfg(stiffness=2, damping=0.1),
        "R_index_intermediate_joint": BaseActuatorCfg(stiffness=2, damping=0.1),
        "R_middle_proximal_joint": BaseActuatorCfg(stiffness=2, damping=0.1),
        "R_middle_intermediate_joint": BaseActuatorCfg(stiffness=2, damping=0.1),
        "R_pinky_proximal_joint": BaseActuatorCfg(stiffness=2, damping=0.1),
        "R_pinky_intermediate_joint": BaseActuatorCfg(stiffness=2, damping=0.1),
        "R_ring_proximal_joint": BaseActuatorCfg(stiffness=2, damping=0.1),
        "R_ring_intermediate_joint": BaseActuatorCfg(stiffness=2, damping=0.1),
        "R_thumb_proximal_yaw_joint": BaseActuatorCfg(stiffness=3, damping=0.15),
        "R_thumb_proximal_pitch_joint": BaseActuatorCfg(stiffness=3, damping=0.15),
        "R_thumb_intermediate_joint": BaseActuatorCfg(stiffness=3, damping=0.15),
        "R_thumb_distal_joint": BaseActuatorCfg(stiffness=3, damping=0.15),
    }
    joint_limits: dict[str, tuple[float, float]] = {
        "waist_yaw_joint": (-2.618, 2.618),
        "waist_roll_joint": (-0.52, 0.52),
        "waist_pitch_joint": (-0.52, 0.52),
        "left_shoulder_pitch_joint": (-3.0892, 2.6704),
        "left_shoulder_roll_joint": (-1.5882, 2.2515),
        "left_shoulder_yaw_joint": (-2.618, 2.618),
        "left_elbow_joint": (-1.0472, 2.0944),
        "left_wrist_pitch_joint": (-1.61, 1.61),
        "left_wrist_roll_joint": (-1.97, 1.97),
        "left_wrist_yaw_joint": (-1.61, 1.61),
        "right_shoulder_pitch_joint": (-3.0892, 2.6704),
        "right_shoulder_roll_joint": (-2.2515, 1.5882),
        "right_shoulder_yaw_joint": (-2.618, 2.618),
        "right_elbow_joint": (-1.0472, 2.0944),
        "right_wrist_pitch_joint": (-1.61, 1.61),
        "right_wrist_roll_joint": (-1.97, 1.97),
        "right_wrist_yaw_joint": (-1.61, 1.61),
    }

    torque_limits: dict[str, float] = {  # = target angles [rad] when action = 0.0
        "waist_yaw_joint": 88,
        "waist_pitch_joint": 55,
        "waist_roll_joint": 55,
        "left_shoulder_pitch_joint": 25,
        "left_shoulder_roll_joint": 25,
        "left_shoulder_yaw_joint": 25,
        "left_elbow_joint": 25,
        "left_wrist_pitch_joint": 25,
        "left_wrist_roll_joint": 25,
        "left_wrist_yaw_joint": 25,
        "right_shoulder_pitch_joint": 25,
        "right_shoulder_roll_joint": 25,
        "right_shoulder_yaw_joint": 25,
        "right_elbow_joint": 25,
        "right_wrist_pitch_joint": 25,
        "right_wrist_roll_joint": 25,
        "right_wrist_yaw_joint": 25,
        "R_index_proximal_joint": 2,
        "R_index_intermediate_joint": 2,
        "R_middle_proximal_joint": 2,
        "R_middle_intermediate_joint": 2,
        "R_pinky_proximal_joint": 2,
        "R_pinky_intermediate_joint": 2,
        "R_ring_proximal_joint": 2,
        "R_ring_intermediate_joint": 2,
        "R_thumb_proximal_yaw_joint": 3,
        "R_thumb_proximal_pitch_joint": 3,
        "R_thumb_intermediate_joint": 3,
        "R_thumb_distal_joint": 3,
    }

    default_joint_positions: dict[str, float] = {  # = target angles [rad] when action = 0.0
        "waist_yaw_joint": 0.0,
        "waist_pitch_joint": 0.0,
        "waist_roll_joint": 0.0,
        "left_shoulder_pitch_joint": 0.0,
        "left_shoulder_roll_joint": 0.0,
        "left_shoulder_yaw_joint": 0.0,
        "left_elbow_joint": 0.0,
        "left_wrist_pitch_joint": 0.0,
        "left_wrist_roll_joint": 0.0,
        "left_wrist_yaw_joint": 0.0,
        "right_shoulder_pitch_joint": 0.0,
        "right_shoulder_roll_joint": 0.0,
        "right_shoulder_yaw_joint": 0.0,
        "right_elbow_joint": 0.0,
        "right_wrist_pitch_joint": 0.0,
        "right_wrist_roll_joint": 0.0,
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
    }

    control_type: dict[str, Literal["position", "effort"]] = {
        "waist_yaw_joint": "effort",
        "waist_pitch_joint": "effort",
        "waist_roll_joint": "effort",
        "left_shoulder_pitch_joint": "effort",
        "left_shoulder_roll_joint": "effort",
        "left_shoulder_yaw_joint": "effort",
        "left_elbow_joint": "effort",
        "left_wrist_pitch_joint": "effort",
        "left_wrist_roll_joint": "effort",
        "left_wrist_yaw_joint": "effort",
        "right_shoulder_pitch_joint": "effort",
        "right_shoulder_roll_joint": "effort",
        "right_shoulder_yaw_joint": "effort",
        "right_elbow_joint": "effort",
        "right_wrist_pitch_joint": "effort",
        "right_wrist_roll_joint": "effort",
        "right_wrist_yaw_joint": "effort",
        "R_index_proximal_joint": "effort",
        "R_index_intermediate_joint": "effort",
        "R_middle_proximal_joint": "effort",
        "R_middle_intermediate_joint": "effort",
        "R_pinky_proximal_joint": "effort",
        "R_pinky_intermediate_joint": "effort",
        "R_ring_proximal_joint": "effort",
        "R_ring_intermediate_joint": "effort",
        "R_thumb_proximal_yaw_joint": "effort",
        "R_thumb_proximal_pitch_joint": "effort",
        "R_thumb_intermediate_joint": "effort",
        "R_thumb_distal_joint": "effort",
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


    right_hand_palm_links: list[str] = ["R_index_proximal","R_middle_proximal", "R_ring_proximal", "R_pinky_proximal"]