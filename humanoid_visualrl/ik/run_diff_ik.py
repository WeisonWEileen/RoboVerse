import argparse

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
from roboverse_pack.robots.vega_cfg import VegaCfg

##
# Pre-defined configs
##
from isaaclab_assets import FRANKA_PANDA_HIGH_PD_CFG, UR10_CFG  # isort:skip


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
            usd_path="/home/panwei/RoboVerse/roboverse_data/robots/vega/vega_root_rot_finger_tip_flattened_mimic_enhanced.usd",
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
        [0.2, 0.1, 0.7, 1.0, 0.0, 0.0, 0.0],
        [0.2, -0.3, 0.6, 1.0, 0.0, 0.0, 0.0],
        [0.2, 0, 0.5, 1.0, 0.0, 0.0, 0.0],
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
        robot_entity_cfg = SceneEntityCfg("robot", joint_names=["R_arm_j.*"], body_names=["R_arm_l7"])
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
            diff_ik_controller.set_command(ik_commands)
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
        goal_marker.visualize(ik_commands[:, 0:3] + scene.env_origins, ik_commands[:, 3:7])


def main():
    """Main function."""
    # Load kit helper
    sim_cfg = sim_utils.SimulationCfg(dt=0.01, device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)
    # Set main camera
    sim.set_camera_view([2.5, 2.5, 2.5], [0.0, 0.0, 0.0])
    # Design scene
    scene_cfg = TableTopSceneCfg(num_envs=args_cli.num_envs, env_spacing=2.0)
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
