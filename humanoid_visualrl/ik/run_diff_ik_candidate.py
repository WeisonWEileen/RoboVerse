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

##
# Pre-defined configs
##
from isaaclab.assets import RigidObjectCfg
from isaaclab_assets import FRANKA_PANDA_HIGH_PD_CFG, UR10_CFG  # isort:skip

from humanoid_visualrl.ik.robotCfg import VegaCfg


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
        pos=[0.0, 0.0, 0.06],
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
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
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

    cube = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Cube",
        spawn=sim_utils.MeshCuboidCfg(
            size=(0.05, 0.05, 0.05),
            mass_props=sim_utils.MassPropertiesCfg(mass=20),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=10.0,  # 增加静摩擦力
                dynamic_friction=10.0,  # 增加动摩擦力
            ),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=True),
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=False, contact_offset=0.02, rest_offset=0.0
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.55, 0.1, 0.9 + 0.06 / 2 + 0.01),
        ),
    )


def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    """Runs the simulation loop."""
    # Extract scene entities
    # note: we only do this here for readability.
    robot = scene["robot"]
    cube = scene["cube"]

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
    recorded_cube_pos = []  # Store cube positions (pos + quat)
    max_recorded_frames = 400
    recording_complete = False

    # Define simulation stepping
    sim_dt = sim.get_physics_dt()
    count = 0
    offset = torch.tensor([0.0, 0.0, 0.06], device=robot.device)
    cube_hand_offset = torch.tensor([0.0, 0.05, 0.03], device=robot.device)

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

            # Set cube position to match command position
            cube_pos_w = ik_commands[:, 0:3] + delta[:, 0:3] + scene.env_origins + offset + cube_hand_offset
            cube_quat_w = ik_commands[:, 3:7] + delta[:, 3:7]
            cube_pose = torch.cat([cube_pos_w, cube_quat_w], dim=-1)
            env_ids = torch.arange(scene.num_envs, device=sim.device)
            cube.write_root_pose_to_sim(cube_pose, env_ids=env_ids)
            cube.write_root_velocity_to_sim(
                torch.zeros((scene.num_envs, 6), device=sim.device, dtype=torch.float32),
                env_ids=env_ids,
            )
            cube.write_data_to_sim()

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
            ik_commands[:, 0:3] + delta[:, 0:3] + scene.env_origins + offset, ik_commands[:, 3:7] + delta[:, 3:7]
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
                # Record qpos for the first environment only
                if torch.any(pos_diff < 0.1):
                    # Get joint positions for the first environment only
                    qpos = robot.data.joint_pos[0].cpu().numpy()  # [num_joints]
                    recorded_qpos.append(qpos)
                    # Get cube position and rotation for the first environment only (relative to env_origins)
                    cube_pos_w = cube.data.root_pos_w[0] - scene.env_origins[0]  # [3]
                    cube_quat_w = cube.data.root_quat_w[0]  # [4]
                    cube_pose = torch.cat([cube_pos_w, cube_quat_w], dim=-1).cpu().numpy()  # [7]
                    recorded_cube_pos.append(cube_pose)
                    print(f"[INFO]: Recorded {len(recorded_qpos)} frames")

                    if len(recorded_qpos) >= max_recorded_frames:
                        recording_complete = True
                        # Save to npz file
                        output_file = "recorded_qpos.npz"
                        # Stack all recorded frames: [max_recorded_frames, num_joints]
                        qpos_array = np.stack(recorded_qpos, axis=0)
                        # Stack cube positions: [max_recorded_frames, 7] (3 pos + 4 quat)
                        cube_pos_array = np.stack(recorded_cube_pos, axis=0)
                        np.savez(output_file, qpos=qpos_array, cube_pos=cube_pos_array)
                        print(f"[INFO]: Recorded {len(recorded_qpos)} frames and saved to {output_file}")
                        print(f"[INFO]: qpos shape: {qpos_array.shape}")
                        print(f"[INFO]: cube_pos shape: {cube_pos_array.shape}")


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
