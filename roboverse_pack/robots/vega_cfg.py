from __future__ import annotations

from typing import Literal

from metasim.scenario.robot import BaseActuatorCfg, RobotCfg
from metasim.utils import configclass


@configclass
class VegaCfg(RobotCfg):
    """Configuration for the Vega Humanoid Robot (vega-1).

    The Vega is a full-bdy humanoid robot with:
    - Mobile base with wheels
    - Torso with 3 DOF
    - Head (fixed)
    - Two 7-DOF arms (left and right)
    - Two 5-finger dexterous hands (left and right)
    - Various sensors (cameras, lidar, IMU, ultrasonic)
    """

    name: str = "vega"

    fix_base_link: bool = True  # Humanoid robots typically have fixed base in simulation

    # Asset paths
    # urdf_path: str = "roboverse_pack/robots/robots_vega/humanoid/vega_1/vega.urdf"
    # usd_path: str = "roboverse_data/robots/vega/vega.usd"
    usd_path: str = "roboverse_data/robots/vega/vega_root_rot_finger_tip_flattened_mimic_enhanced.usd"
    # Physical properties
    enabled_gravity: bool = True  # Disable gravity for default setup

    # ==================== Actuator Configuration ====================
    # head = 2 + hand = 6 + arm = 7 = 15
    actuators: dict[str, BaseActuatorCfg] = {
        # "L_arm_j7": BaseActuatorCfg(velocity_limit=2.7, torque_limit=25.0, stiffness=5e3, damping=500),
        # Right arm - progressive stiffness from base to tip
        # "head_j1": BaseActuatorCfg(velocity_limit=2.4, torque_limit=150.0, stiffness=6, damping=3.2), # we do not need roll
        "base_yaw_joint": BaseActuatorCfg(velocity_limit=2.4, torque_limit=1000.0, stiffness=10000.0, damping=2000.0),
        "head_j2": BaseActuatorCfg(velocity_limit=2.4, torque_limit=150.0, stiffness=2.5, damping=3.2),
        "head_j3": BaseActuatorCfg(velocity_limit=2.4, torque_limit=150.0, stiffness=6, damping=3.2),
        "R_arm_j1": BaseActuatorCfg(
            velocity_limit=2.4, torque_limit=150.0, stiffness=5e4, damping=5e3
        ),  # Increased for stability
        "R_arm_j2": BaseActuatorCfg(
            velocity_limit=2.4, torque_limit=150.0, stiffness=5e4, damping=5e3
        ),  # Increased for stability
        "R_arm_j3": BaseActuatorCfg(velocity_limit=2.7, torque_limit=80.0, stiffness=2e4, damping=2e3),
        "R_arm_j4": BaseActuatorCfg(velocity_limit=2.7, torque_limit=80.0, stiffness=1e4, damping=1e3),
        "R_arm_j5": BaseActuatorCfg(velocity_limit=2.7, torque_limit=25.0, stiffness=5e3, damping=500),
        "R_arm_j6": BaseActuatorCfg(velocity_limit=2.7, torque_limit=25.0, stiffness=5e3, damping=500),
        "R_arm_j7": BaseActuatorCfg(velocity_limit=2.7, torque_limit=25.0, stiffness=5e3, damping=500),
        # Right hand - Thumb
        "R_ff_j1": BaseActuatorCfg(velocity_limit=6.28, torque_limit=0.9, stiffness=320, damping=22),
        # "R_ff_j2": BaseActuatorCfg(velocity_limit=6.28, torque_limit=0.9, stiffness=320, damping=22), follower joint
        "R_lf_j1": BaseActuatorCfg(velocity_limit=6.28, torque_limit=0.9, stiffness=320, damping=22),
        # "R_lf_j2": BaseActuatorCfg(velocity_limit=6.28, torque_limit=0.9, stiffness=320, damping=22),
        "R_mf_j1": BaseActuatorCfg(velocity_limit=6.28, torque_limit=0.9, stiffness=320, damping=22),
        # "R_mf_j2": BaseActuatorCfg(velocity_limit=6.28, torque_limit=0.9, stiffness=320, damping=22),
        "R_rf_j1": BaseActuatorCfg(velocity_limit=6.28, torque_limit=0.9, stiffness=320, damping=22),
        "R_th_j0": BaseActuatorCfg(velocity_limit=6.28, torque_limit=1.4, stiffness=300, damping=22),
        "R_th_j1": BaseActuatorCfg(velocity_limit=6.28, torque_limit=1.4, stiffness=300, damping=22),
        # "R_th_j2": BaseActuatorCfg(velocity_limit=6.28, torque_limit=1.1, stiffness=260, damping=20),
    }
    # action_scale = {
    #     "base_yaw_joint": 0.35,
    #     "head_j2": 0.45,
    #     "head_j3": 0.25,
    #     "R_arm_j1": 0.25,
    #     "R_arm_j2": 0.25,
    #     "R_arm_j3": 0.05,
    #     "R_arm_j4": 0.25,
    #     "R_arm_j5": 0.25,
    #     "R_arm_j6": 0.05,
    #     "R_arm_j7": 0.20,
    #     # "R_arm_j1": 0.0,
    #     # "R_arm_j2": 0.0,
    #     # "R_arm_j3": 0.0,
    #     # "R_arm_j4": 0.0,
    #     # "R_arm_j5": 0.0,
    #     # "R_arm_j6": 0.0,
    #     # "R_arm_j7": 0.0,

    #     "R_ff_j1": 0.35,
    #     "R_lf_j1": 0.35,
    #     "R_mf_j1": 0.35,
    #     "R_rf_j1": 0.35,
    #     "R_th_j0": 0.35,
    #     "R_th_j1": 0.80,
    # }
    scale_factor = 0.25  # for all scale tuning
    action_scale = {
        "base_yaw_joint": 0.25 * scale_factor,
        "head_j2": 0.05 * scale_factor,
        "head_j3": 0.05 * scale_factor,
        "R_arm_j1": 0.15 * 0.5 * scale_factor,
        "R_arm_j2": 0.15 * 0.5 * scale_factor,
        "R_arm_j3": 0.05 * 0.5 * scale_factor,
        "R_arm_j4": 0.18 * 0.5 * scale_factor,
        "R_arm_j5": 0.15 * 0.5 * scale_factor,
        "R_arm_j6": 0.05 * 0.5 * scale_factor,
        "R_arm_j7": 0.10 * 0.5 * scale_factor,
        # "R_arm_j1": 0.0,
        # "R_arm_j2": 0.0,
        # "R_arm_j3": 0.0,
        # "R_arm_j4": 0.0,
        # "R_arm_j5": 0.0,
        # "R_arm_j6": 0.0,
        # "R_arm_j7": 0.0,
        "R_ff_j1": 0.10 * scale_factor,
        "R_lf_j1": 0.10 * scale_factor,
        "R_mf_j1": 0.10 * scale_factor,
        "R_rf_j1": 0.10 * scale_factor,
        "R_th_j0": 0.10 * scale_factor,
        "R_th_j1": 0.10 * scale_factor,
    }
    assert len(action_scale) == len(actuators), f"action_scale: {action_scale} != len(actuators): {len(actuators)}"

    # five mimic joints in the finger
    mimic_joints: dict[str] = {"R_ff_j2", "R_lf_j·2", "R_mf_j2", "R_rf_j2", "R_th_j2"}

    modified_joint_limits: dict[str, tuple[float, float]] = {
        "head_j3": (-0.52, 0.0),
        "head_j2": (-0.71, 0.71),
        "R_arm_j1": (-2.50, -1.25),
        "R_arm_j3": (-0.45, 0.45),
    }

    # ==================== Joint Limits ====================
    # Joint angle limits from URDF (in radians)
    joint_limits: dict[str, tuple[float, float]] = {
        "head_j3": (-0.52, 0.0),
        "head_j2": (-0.71, 0.71),
        "base_yaw_joint": (-1.5708, 1.5708),
        "R_arm_j1": (-2.50, -1.25),  # (-3.071, 3.071),
        "R_arm_j2": (-1.553, 0.453),
        "R_arm_j3": (-0.45, 0.45),  # (-3.071, 3.071),
        "R_arm_j4": (-3.071, 0.244),
        "R_arm_j5": (-3.071, 3.071),
        "R_arm_j6": (-1.396, 1.396),
        "R_arm_j7": (-1.117, 1.378),
        # Right hand - Thumb
        "R_ff_j1": (-1.0946, 0.2891),
        "R_lf_j1": (-1.0118, 0.2811),
        "R_mf_j1": (-1.0844, 0.2801),
        "R_rf_j1": (-1.0154, 0.2840),
        "R_th_j0": (-0.0158, 1.605),
        "R_th_j1": (-0.3468, 0.1834),
        "torso_j2": (0.00, 0.60),
        # "R_th_j2": (-0.95, 0.95),
    }

    # ==================== Control Types ====================
    # Default to position control for all joints
    control_type: dict[str, Literal["position", "effort"]] = {
        "base_yaw_joint": "position",
        # Base wheels
        # "B_wheel_j1": "effort",
        "B_wheel_j2": "position",
        # "R_wheel_j1": "effort",
        "R_wheel_j2": "position",
        "L_wheel_j1": "position",
        "L_wheel_j2": "position",
        # Torso
        # "torso_j1": "position",
        # "torso_j2": "position",
        # "torso_j3": "position",
        # Head
        "head_j1": "position",
        "head_j2": "position",
        "head_j3": "position",
        # Left arm
        "L_arm_j1": "position",
        "L_arm_j2": "position",
        "L_arm_j3": "position",
        "L_arm_j4": "position",
        "L_arm_j5": "position",
        "L_arm_j6": "position",
        "L_arm_j7": "position",
        # Right arm
        "R_arm_j1": "position",
        "R_arm_j2": "position",
        "R_arm_j3": "position",
        "R_arm_j4": "position",
        "R_arm_j5": "position",
        "R_arm_j6": "position",
        "R_arm_j7": "position",
        # Left hand
        "L_th_j0": "position",
        "L_th_j1": "position",
        "L_th_j2": "position",
        "L_ff_j1": "position",
        "L_ff_j2": "position",
        "L_mf_j1": "position",
        "L_mf_j2": "position",
        "L_rf_j1": "position",
        "L_rf_j2": "position",
        "L_lf_j1": "position",
        "L_lf_j2": "position",
        # Right hand
        "R_th_j0": "position",
        "R_th_j1": "position",
        "R_th_j2": "position",
        "R_ff_j1": "position",
        "R_ff_j2": "position",
        "R_mf_j1": "position",
        "R_mf_j2": "position",
        "R_rf_j1": "position",
        "R_rf_j2": "position",
        "R_lf_j1": "position",
        "R_lf_j2": "position",
    }

    ee_body_name: str = "L_arm_l7"  # Last arm link with geometry (fixed joint, but
    gripper_close_q: list[float] = [
        1.20,  # L_th_j0: thumb abduction (close towards palm)
        -0.30,  # L_th_j1: thumb flexion (negative closes)
        -0.40,  # L_th_j2: thumb tip flexion
        -0.95,  # L_ff_j1: index finger proximal (negative closes)
        -1.05,  # L_ff_j2: index finger distal
        -0.95,  # L_mf_j1: middle finger proximal
        -1.05,  # L_mf_j2: middle finger distal
        -0.90,  # L_rf_j1: ring finger proximal
        -1.00,  # L_rf_j2: ring finger distal
        -0.90,  # L_lf_j1: little finger proximal
        -1.00,  # L_lf_j2: little finger distal
    ]  # Closed hand (approx. 85-90% of lower joint limits)
    gripper_open_q: list[float] = [
        0.20,  # L_th_j0: relaxed abduction
        0.0,  # L_th_j1: thumb flexion open
        0.0,  # L_th_j2: thumb tip open
        0.0,  # L_ff_j1: index finger proximal open
        0.0,  # L_ff_j2: index finger distal open
        0.0,  # L_mf_j1: middle finger proximal open
        0.0,  # L_mf_j2: middle finger distal open
        0.0,  # L_rf_j1: ring finger proximal open
        0.0,  # L_rf_j2: ring finger distal open
        0.0,  # L_lf_j1: little finger proximal open
        0.0,  # L_lf_j2: little finger distal open
    ]  # Open hand (neutral positions)

    # ==================== Default Joint Positions ====================
    # Default home positions (can be customized based on use case)
    default_joint_positions: dict[str, float] = {
        "base_yaw_joint": 0.0,
        "head_j2": 0.0,
        "head_j3": 0.0,
        "R_arm_j1": -2.06,
        "R_arm_j2": -0.21,
        "R_arm_j3": -0.13,
        "R_arm_j4": -2.59,
        "R_arm_j5": -0.3,
        "R_arm_j6": 0.05,
        "R_arm_j7": 0.0,
        "R_ff_j1": 0.0,
        "R_lf_j1": 0.0,
        "R_mf_j1": 0.0,
        "R_rf_j1": 0.0,
        "R_th_j0": 1.47,
        "R_th_j1": 0.15,
        # tight initial pose
        "L_arm_j1": 3.06,
        "L_arm_j2": 0.00,
        "L_arm_j3": 0.0,
        "L_arm_j4": -2.2,
        "L_arm_j5": 0.0,
        "L_arm_j6": -0.71,
        "L_arm_j7": -0.13,
        # "torso_j1": 0.38,
        # open initial pose
        "torso_j1": 0.15,
        "torso_j2": 0.00,
        "torso_j3": -0.88,
        "R_ff_j2": 0.0,
        "R_lf_j2": 0.0,
        "R_mf_j2": 0.0,
        "R_rf_j2": 0.0,
        "R_th_j2": 0.0,
        "R_wheel_j1": -0.63783,
        "L_wheel_j1": 0.63783,
        # "R_wheel_j2": 0.0,
        # "L_wheel_j2": 0.0,
        # "B_wheel_j1": 0.0,
        # "B_wheel_j2": 0.0,
    }
    # joints that need to change default joint positions but not be actuated and fixed
    default_fixed_joints: list[str] = {
        "L_arm_j1",
        "L_arm_j2",
        "L_arm_j3",
        "L_arm_j4",
        "L_arm_j5",
        "L_arm_j6",
        "L_arm_j7",
        "torso_j1",
        "torso_j2",
        "torso_j3",
        "R_wheel_j1",
        "L_wheel_j1",
    }

    # origial_config_joint: list[str] = {"B_wheel_j1", "B_wheel_j2"}

    # velocity_joints: list[str] = ["R_wheel_j2", "L_wheel_j2"]

    torque_limits: dict[str, float] = {
        "base_yaw_joint": 1000.0,
        "head_j1": 150.0,
        "head_j2": 150.0,
        "head_j3": 150.0,
        "R_arm_j1": 150.0,
        "R_arm_j2": 150.0,
        "R_arm_j3": 150.0,
        "R_arm_j4": 150.0,
        "R_arm_j5": 150.0,
        "R_arm_j6": 150.0,
        "R_arm_j7": 150.0,
        "R_th_j0": 100.0,
        "R_th_j1": 100.0,
        "R_th_j2": 100.0,
        "R_ff_j1": 100.0,
        "R_ff_j2": 100.0,
        "R_mf_j1": 100.0,
        "R_mf_j2": 100.0,
        "R_rf_j1": 100.0,
        "R_rf_j2": 100.0,
        "R_lf_j1": 100.0,
        "R_lf_j2": 100.0,
    }
    num_joints: int = len(actuators) + len(mimic_joints)

    feet_links: list[str] = []
    knee_links: list[str] = []
    elbow_links: list[str] = []
    wrist_links: list[str] = ["L_arm_l7", "R_arm_l7"]
    torso_links: list[str] = []
    terminate_contacts_links = []
    penalized_contacts_links: list[str] = []

    # joint substrings, to find indices of joints.

    upper_body_joints = []

    mimic_joints = {
        "R_th_j2",
        "R_ff_j2",
        "R_lf_j2",
        "R_mf_j2",
        "R_rf_j2",
    }

    num_joints_all: int = (
        len(actuators) + len(mimic_joints) + len(default_fixed_joints)
        # + len(velocity_joints)
        # + len(origial_config_joint)
    )

    assert num_joints_all == len(default_joint_positions), (
        f"num_joints_all: {num_joints_all} != len(default_joint_positions): {len(default_joint_positions)}"
    )

    right_arm_joints = {
        "R_arm_j3",
        "R_arm_j4",
        "R_arm_j5",
        "R_arm_j6",
        "R_arm_j7",
    }

    tip_link_names = {"R_ff_tip", "R_lf_tip", "R_mf_tip", "R_rf_tip", "R_th_tip"}

    right_palm_link = ["R_mf_l1"]

    # exploration joint with smaller penalty
    actuators_energy_penalty_coffe: dict[str, BaseActuatorCfg] = {
        "base_yaw_joint": 0.5,
        "head_j2": 0.01,
        "head_j3": 0.01,
        "R_arm_j1": 1,  # Increased for stability
        "R_arm_j2": 1,  # Increased for stability
        "R_arm_j3": 1,
        "R_arm_j4": 1,
        "R_arm_j5": 1,
        "R_arm_j6": 1,
        "R_arm_j7": 1,
        # Right hand - Thumb
        "R_ff_j1": 0.35,
        # "R_ff_j2": , follower joint
        "R_lf_j1": 0.35,
        # "R_lf_j2": BaseActuatorCfg(velocity_limit=6.28, torque_limit=0.9, stiffness=320, damping=22),
        "R_mf_j1": 0.35,
        # "R_mf_j2": BaseActuatorCfg(velocity_limit=6.28, torque_limit=0.9, stiffness=320, damping=22),
        "R_rf_j1": 0.35,
        "R_th_j0": 0.35,
        "R_th_j1": 0.80,
        # "R_th_j2": BaseActuatorCfg(velocity_limit=6.28, torque_limit=1.1, stiffness=260, damping=20),
    }
