import argparse
import os
import sys

# Add workspace root to Python path for roboverse_pack imports
workspace_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if workspace_root not in sys.path:
    sys.path.insert(0, workspace_root)

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Tutorial on using the differential IK controller.")
parser.add_argument("--robot", type=str, default="franka_panda", help="Name of the robot.")
parser.add_argument("--num_envs", type=int, default=128, help="Number of environments to spawn.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch
import numpy as np

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.math import subtract_frame_transforms
# from roboverse_pack.robots.vega_cfg import VegaCfg
from typing import Literal
##
# Pre-defined configs
##
from isaaclab_assets import FRANKA_PANDA_HIGH_PD_CFG, UR10_CFG  # isort:skip
from metasim.scenario.robot import BaseActuatorCfg, RobotCfg

@configclass
class VegaCfg(RobotCfg):
    """Configuration for the Vega Humanoid Robot (vega-1).

    The Vega is a full-body humanoid robot with:
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
        "base_yaw_joint": -0.3,
        "head_j2": 0.0,
        "head_j3": 0.0,
        "R_arm_j1": -2.06,
        "R_arm_j2": -0.21,
        "R_arm_j3": -0.13,
        "R_arm_j4": -2.59,
        "R_arm_j5": -0.3,
        "R_arm_j6": 0.51,
        "R_arm_j7": 0.0,
        "R_ff_j1": 0.0,
        "R_lf_j1": 0.0,
        "R_mf_j1": 0.0,
        "R_rf_j1": 0.0,
        "R_th_j0": 1.47,
        "R_th_j1": -0.10,
        # tight initial pose
        "L_arm_j1": 3.06,
        "L_arm_j2": 0.00,
        "L_arm_j3": 0.0,
        "L_arm_j4": -2.1,
        "L_arm_j5": 0.0,
        "L_arm_j6": -0.71,
        "L_arm_j7": -0.13,
        # "torso_j1": 0.38,
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
    print(
        f"Joints Length INFO: {num_joints_all} != {len(default_joint_positions)}, With actuators length: {len(actuators)} + mimic joints length: {len(mimic_joints)} + default fixed joints length: {len(default_fixed_joints)}"
    )

    assert num_joints_all == len(default_joint_positions), (
        f"num_joints_all: {num_joints_all} != len(default_joint_positions): {len(default_joint_positions)}"
    )

    right_arm_joints = {
        # "R_arm_j1",
        # "R_arm_j2",
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


def load_robot_cfg():
    from isaaclab.assets import ArticulationCfg
    from isaaclab.actuators import ImplicitActuatorCfg

    robot = VegaCfg()
    robot_actuators_names = []
    # include real actuators and default fixed joints
    for jn in robot.actuators.keys():
        robot_actuators_names.append(jn)
    if hasattr(robot, "default_fixed_joints"):
        robot_actuators_names.extend(robot.default_fixed_joints)

    sorted_actuator_names = sorted(robot_actuators_names)
    actuators = {}
    for jn in sorted_actuator_names:
        if jn in robot.actuators.keys():
            actuators[jn] = ImplicitActuatorCfg(
                # prim_path
                joint_names_expr=[jn],
                # TODO fix this with different mode
                stiffness=robot.actuators[jn].stiffness if robot.control_type[jn] == "position" else 0.0,
                damping=robot.actuators[jn].damping if robot.control_type[jn] == "position" else 0.0,
                armature=0.01,
                friction=0.05,
                # TODO armature to be determined
            )
        elif hasattr(robot, "velocity_joints") and jn in robot.velocity_joints:
            actuators[jn] = ImplicitActuatorCfg(
                joint_names_expr=[jn],
                stiffness=0,
                damping=100,
                velocity_limit=100,
                effort_limit=100,
                armature=0.01,
                friction=0.05,
            )
        elif hasattr(robot, "default_fixed_joints") and jn in robot.default_fixed_joints:
            actuators[jn] = ImplicitActuatorCfg(
                joint_names_expr=[jn], stiffness=100000.0, damping=1000.0, friction=300, armature=1000.0
            )
        elif hasattr(robot, "origial_config_joints") and jn in robot.origial_config_joints:
            actuators[jn] = ImplicitActuatorCfg(
                joint_names_expr=[jn], stiffness=0.0, damping=0.00, friction=0.01, armature=0.01
            )
    init_state = ArticulationCfg.InitialStateCfg(
        pos=[0.0, 0.0, 0.0],
        # joint_pos={jn: robot.default_joint_positions[jn] for jn in robot.actuators.keys()},
        joint_pos={jn: robot.default_joint_positions[jn] for jn in robot.default_joint_positions.keys()},
        joint_vel={".*": 0.0},
    )
    robot = ArticulationCfg(
        spawn=sim_utils.UsdFileCfg(
            usd_path="roboverse_data/robots/vega/vega_root_rot_finger_tip_flattened_mimic_enhanced.usd",
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                retain_accelerations=False,
                linear_damping=0.0,
                angular_damping=0.0,
                max_linear_velocity=1000.0,
                max_angular_velocity=1000.0,
                max_depenetration_velocity=1.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                fix_root_link=True,
                enabled_self_collisions=False,
                solver_position_iteration_count=32,
                solver_velocity_iteration_count=0,
                sleep_threshold=0.005,  # 休眠阈值
                stabilization_threshold=0.0005,  # 稳定化阈值
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=True, contact_offset=0.02, rest_offset=0.0
            ),
        ),
        actuators=actuators,
        init_state=init_state,
    )

    return robot


@configclass
class TableTopSceneCfg(InteractiveSceneCfg):
    """Configuration for a cart-pole scene."""

    # ground plane
    ground = AssetBaseCfg(
        prim_path="/World/defaultGroundPlane",
        spawn=sim_utils.GroundPlaneCfg(),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -0.1)),
    )

    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    )

    # articulation
    if args_cli.robot == "franka_panda":
        robot = FRANKA_PANDA_HIGH_PD_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    elif args_cli.robot == "ur10":
        robot = UR10_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    elif args_cli.robot == "vega":
        robot = load_robot_cfg().replace(prim_path="{ENV_REGEX_NS}/Robot")
    else:
        raise ValueError(f"Robot {args_cli.robot} is not supported. Valid: franka_panda, ur10")

    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        spawn=sim_utils.UsdFileCfg(
            usd_path="roboverse_data/scenes/tritable.usd",
            scale=(1.0, 1.0, 1.0),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True,
                kinematic_enabled=True,  # This fixes the rigid body
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(fix_root_link=True),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.20, 0.0, 0.50), rot=(1.0, 0.0, 0.0, 0.0)),
    )


def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    """Runs the simulation loop."""
    # Extract scene entities
    # note: we only do this here for readability.
    robot = scene["robot"]

    # Create controller
    diff_ik_cfg = DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls")
    diff_ik_controller = DifferentialIKController(diff_ik_cfg, num_envs=scene.num_envs, device=sim.device)

    # Markers
    frame_marker_cfg = FRAME_MARKER_CFG.copy()
    frame_marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
    ee_marker = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/ee_current"))
    goal_marker = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/ee_goal"))

    # Define goals for the arm
    ee_goals = [
        [0.5, -0.2, 0.65, 0, 0, -0.7071, 0.7071],
    ]

    ee_goals = torch.tensor(ee_goals, device=sim.device)
    # Track the given command
    current_goal_idx = 0
    # Create buffers to store actions
    ik_commands = torch.zeros(scene.num_envs, diff_ik_controller.action_dim, device=robot.device)
    ik_commands[:] = ee_goals[current_goal_idx]

    # Specify robot-specific parameters
    if args_cli.robot == "franka_panda":
        robot_entity_cfg = SceneEntityCfg("robot", joint_names=["panda_joint.*"], body_names=["panda_hand"])
    elif args_cli.robot == "ur10":
        robot_entity_cfg = SceneEntityCfg("robot", joint_names=[".*"], body_names=["ee_link"])
    elif args_cli.robot == "vega":
        robot_entity_cfg = SceneEntityCfg("robot", joint_names=["R_arm_j.*"], body_names=["R_mf_l1"])
    else:
        raise ValueError(f"Robot {args_cli.robot} is not supported. Valid: franka_panda, ur10")
    # Resolving the scene entities
    robot_entity_cfg.resolve(scene)
    # Obtain the frame index of the end-effector
    # For a fixed base robot, the frame index is one less than the body index. This is because
    # the root body is not included in the returned Jacobians.
    if robot.is_fixed_base:
        ee_jacobi_idx = robot_entity_cfg.body_ids[0] - 1
    else:
        ee_jacobi_idx = robot_entity_cfg.body_ids[0]

    robot_cfg = VegaCfg()

    fixed_joint_ids = []
    fixed_joint_pos = []
    for jn in robot_cfg.default_joint_positions.keys():
        if jn not in robot_entity_cfg.joint_names and "R_arm_" not in jn:
            fixed_joint_ids.append(robot.joint_names.index(jn))
            fixed_joint_pos.append(robot_cfg.default_joint_positions[jn])
    fixed_joint_ids = torch.tensor(fixed_joint_ids, device=robot.device)
    fixed_joint_pos = torch.tensor(fixed_joint_pos, device=robot.device)

    # Initialize data collection for qpos recording
    recorded_qpos = []
    max_recorded_frames = 400
    recording_complete = False

    # Define simulation stepping
    sim_dt = sim.get_physics_dt()
    count = 0
    # Simulation loop
    while simulation_app.is_running():
        # reset
        if count % 150 == 0:
            # reset time
            count = 0
            # reset joint state
            joint_pos = robot.data.default_joint_pos.clone()
            joint_vel = robot.data.default_joint_vel.clone()
            robot.write_joint_state_to_sim(joint_pos, joint_vel)
            robot.reset()
            # reset actions
            ik_commands[:] = ee_goals[current_goal_idx]
            joint_pos_des = joint_pos[:, robot_entity_cfg.joint_ids].clone()
            # reset controller
            diff_ik_controller.reset()
            delta = torch.zeros_like(ik_commands)
            delta[:, 0] += torch.randn_like(ik_commands[:, 0]) * 0.08
            delta[:, 1] += torch.randn_like(ik_commands[:, 1]) * 0.08 - 0.03
            delta[:, 2] += torch.randn_like(ik_commands[:, 2]) * 0.03

            diff_ik_controller.set_command(ik_commands + delta)
            # change goal
            current_goal_idx = (current_goal_idx + 1) % len(ee_goals)
        else:
            # obtain quantities from simulation
            # get_jacobians() returns shape: [num_envs, num_bodies, 6, num_all_joints]
            # After indexing [:, ee_jacobi_idx, :, robot_entity_cfg.joint_ids]:
            # - [:, ...] : all environments (128)
            # - [ee_jacobi_idx, ...] : specific end-effector body (single body)
            # - [:, :, ...] : all 6 DOF (3 position + 3 rotation)
            # - [:, :, :, robot_entity_cfg.joint_ids] : selected joint indices (7 joints for Franka Panda)
            # Final shape: [128, 6, 7] = [num_envs, 6DOF, num_selected_joints]
            jacobian = robot.root_physx_view.get_jacobians()[:, ee_jacobi_idx, :, robot_entity_cfg.joint_ids]
            # import pdb; pdb.set_trace()
            ee_pose_w = robot.data.body_pose_w[:, robot_entity_cfg.body_ids[0]]
            root_pose_w = robot.data.root_pose_w
            joint_pos = robot.data.joint_pos[:, robot_entity_cfg.joint_ids]
            # compute frame in root frame
            ee_pos_b, ee_quat_b = subtract_frame_transforms(
                root_pose_w[:, 0:3], root_pose_w[:, 3:7], ee_pose_w[:, 0:3], ee_pose_w[:, 3:7]
            )
            # compute the joint commands
            joint_pos_des = diff_ik_controller.compute(ee_pos_b, ee_quat_b, jacobian, joint_pos)

        # apply actions
        robot.set_joint_position_target(joint_pos_des, joint_ids=robot_entity_cfg.joint_ids)
        # set other joint positions to default joint positions
        robot.set_joint_position_target(fixed_joint_pos, joint_ids=fixed_joint_ids)

        scene.write_data_to_sim()
        # perform step
        sim.step()
        # update sim-time
        count += 1
        # update buffers
        scene.update(sim_dt)

        # obtain quantities from simulation
        ee_pose_w = robot.data.body_state_w[:, robot_entity_cfg.body_ids[0], 0:7]
        # update marker positions
        ee_marker.visualize(ee_pose_w[:, 0:3], ee_pose_w[:, 3:7])
        goal_marker.visualize(
            ik_commands[:, 0:3] + delta[:, 0:3] + scene.env_origins, ik_commands[:, 3:7] + delta[:, 3:7]
        )

        # Check if ee_pose_w xyz difference is less than 0.05 and record qpos
        if (count + 1) % 150 == 0:
            if not recording_complete:
                # Calculate target position (considering env_origins)
                target_pos = ik_commands[:, 0:3] + delta[:, 0:3] + scene.env_origins
                # Current ee position in world frame
                current_pos = ee_pose_w[:, 0:3]
                # Calculate position difference
                pos_diff = torch.norm(current_pos - target_pos, dim=1)  # [num_envs]

                # Check if any environment has position error < 0.05
                # Record qpos for all environments if at least one meets the condition
                if torch.any(pos_diff < 0.05):
                    # Get all joint positions
                    qpos = robot.data.joint_pos.cpu().numpy()  # [num_envs, num_joints]
                    recorded_qpos.append(qpos)
                    print(f"[INFO]: Recorded {len(recorded_qpos)} frames")

                    if len(recorded_qpos) >= max_recorded_frames:
                        recording_complete = True
                        # Save to npz file
                        output_file = "recorded_qpos.npz"
                        # Stack all recorded frames: [max_recorded_frames, num_envs, num_joints]
                        qpos_array = np.stack(recorded_qpos, axis=0)
                        np.savez(output_file, qpos=qpos_array)
                        print(f"[INFO]: Recorded {len(recorded_qpos)} frames and saved to {output_file}")
                        print(f"[INFO]: qpos shape: {qpos_array.shape}")


def main():
    """Main function."""
    # Load kit helper
    sim_cfg = sim_utils.SimulationCfg(dt=0.01, device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)
    # Set main camera
    sim.set_camera_view([2.5, 2.5, 2.5], [0.0, 0.0, 0.0])
    # Design scene
    scene_cfg = TableTopSceneCfg(num_envs=args_cli.num_envs, env_spacing=5.0)
    scene = InteractiveScene(scene_cfg)
    # Play the simulator
    sim.reset()
    # Now we are ready!
    print("[INFO]: Setup complete...")
    # Run the simulator
    run_simulator(sim, scene)


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
