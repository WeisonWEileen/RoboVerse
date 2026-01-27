"""Inverse state generator for IK-based curriculum learning.

This module provides functionality to generate initial states using inverse kinematics,
which can be used for curriculum learning in reinforcement learning tasks.
"""

from __future__ import annotations

import torch
from loguru import logger as log

# IK related imports
try:
    from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
    from isaaclab.managers import SceneEntityCfg
    from isaaclab.markers import VisualizationMarkers
    from isaaclab.markers.config import FRAME_MARKER_CFG
    from isaaclab.utils.math import subtract_frame_transforms

    ISAACLAB_AVAILABLE = True
except ImportError:
    ISAACLAB_AVAILABLE = False
    log.error("IsaacLab not available, IK inverse curriculum will be skipped")

from humanoid_visualrl.wrapper.base_humanoid_wrapper import HumanoidBaseWrapper


# 四元数乘法 (IsaacSim: wxyz)
def quat_mul(q, r):
    # q, r: [..., 4], [w, x, y, z]
    w1, x1, y1, z1 = q.unbind(-1)
    w2, x2, y2, z2 = r.unbind(-1)
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return torch.stack([w, x, y, z], dim=-1)


def generate_ik_curriculum_data(env: HumanoidBaseWrapper, task_cfg, num_samples=10, threshold=0.1, max_iterations=150):
    """Generate IK curriculum data by inverse kinematics to cube positions.

    This function uses differential IK to move the robot's end-effector to various
    positions near the cube, and records successful configurations (qpos, cube_pos)
    that can be used as initial states for curriculum learning.

    Args:
        env: The environment instance (wrapper with env.env.handler.scene or handler.scene)
        task_cfg: The task configuration (must have robot name and object info)
        num_samples: Number of successful IK samples to collect
        threshold: Position error threshold for success (meters)
        max_iterations: Maximum iterations per IK attempt

    Returns:
        List of (qpos, cube_pos) tuples, where:
        - qpos: numpy array of joint positions [num_joints]
        - cube_pos: numpy array of cube pose [7] (3 pos + 4 quat, relative to env_origin)
    """
    if not ISAACLAB_AVAILABLE:
        log.warning("IsaacLab not available, skipping IK curriculum generation")
        return []

    # Handle different environment structures (wrapper vs direct handler)
    handler = env.env

    # Check if handler has scene (IsaacSim/IsaacLab)
    if not hasattr(handler, "scene"):
        log.warning("Handler does not have scene attribute, skipping IK curriculum generation")
        return []

    env._reset(env_ids=list(range(env.num_envs)))

    scene = handler.scene
    sim = handler.sim

    robot_name = "vega"
    object_name = "object"

    robot = scene.articulations[robot_name]
    cube = scene.rigid_objects[object_name]

    # Save original cube state and fix it for IK generation
    # We'll restore it at the end of the function
    # Get original fix_base_link state from task_cfg
    cube_original_fixed = False
    for obj_cfg in task_cfg.objects:
        if obj_cfg.name == object_name and hasattr(obj_cfg, "fix_base_link"):
            cube_original_fixed = obj_cfg.fix_base_link
            break

    # try:
    # Fix the cube for IK generation using UsdPhysics API
    import omni.usd
    from pxr import UsdPhysics

    stage = omni.usd.get_context().get_stage()
    if stage:
        # Fix the cube for IK generation
        # Set kinematic_enabled=True and disable_gravity=True for all environments
        for env_id in range(scene.num_envs):
            env_cube_prim_path = f"/World/envs/env_{env_id}/{object_name}"
            env_cube_prim = stage.GetPrimAtPath(env_cube_prim_path)
            if env_cube_prim.IsValid():
                # Use UsdPhysics.RigidBodyAPI instead of PhysxSchema
                env_rigid_body_api = UsdPhysics.RigidBodyAPI.Apply(env_cube_prim)
                if env_rigid_body_api:
                    # Create or set kinematic enabled attribute
                    # When kinematic is enabled, gravity is automatically disabled
                    kinematic_attr = env_rigid_body_api.GetKinematicEnabledAttr()
                    if not kinematic_attr:
                        kinematic_attr = env_rigid_body_api.CreateKinematicEnabledAttr(True)
                    else:
                        kinematic_attr.Set(True)
        log.info("Fixed cube for IK generation (kinematic=True, gravity disabled)")
    # except Exception as e:
    #     log.warning(f"Could not fix cube via PhysX API: {e}. Cube may move during IK generation.")

    # Create IK controller
    diff_ik_cfg = DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls")
    diff_ik_controller = DifferentialIKController(diff_ik_cfg, num_envs=scene.num_envs, device=sim.device)

    # Markers
    frame_marker_cfg = FRAME_MARKER_CFG.copy()
    frame_marker_cfg.markers["frame"].scale = (0.05, 0.05, 0.05)
    ee_marker = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/ee_current"))
    goal_marker = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/ee_goal"))

    # Robot-specific configuration for Vega
    robot_entity_cfg = SceneEntityCfg(robot_name, joint_names=["R_arm_j.*"], body_names=["R_mf_l1"])

    robot_entity_cfg.resolve(scene)

    # Get end-effector index
    if robot.is_fixed_base:
        ee_jacobi_idx = robot_entity_cfg.body_ids[0] - 1
    else:
        ee_jacobi_idx = robot_entity_cfg.body_ids[0]

    # Get fixed joint positions (for non-arm joints)
    from humanoid_visualrl.ik.robotCfg import VegaCfg

    robot_cfg = VegaCfg()
    fixed_joint_ids = []
    fixed_joint_pos = []

    robot_joint_names = robot.joint_names
    for jn in robot_cfg.default_joint_positions.keys():
        if jn not in robot_entity_cfg.joint_names and "R_arm_" not in jn:
            fixed_joint_ids.append(robot_joint_names.index(jn))
            fixed_joint_pos.append(robot_cfg.default_joint_positions[jn])

    fixed_joint_ids = torch.tensor(fixed_joint_ids, device=robot.device)
    fixed_joint_pos = torch.tensor(fixed_joint_pos, device=robot.device)
    # Broadcast to all environments: [num_fixed_joints] -> [num_envs, num_fixed_joints]
    fixed_joint_pos = fixed_joint_pos.unsqueeze(0).repeat(scene.num_envs, 1)

    # Initialize data collection
    recorded_qpos = []
    recorded_cube_pos_list = []
    # Get physics dt from handler or use default
    if hasattr(handler, "physics_dt"):
        sim_dt = handler.physics_dt
    elif hasattr(sim, "get_physics_dt"):
        sim_dt = sim.get_physics_dt()
    else:
        sim_dt = (
            task_cfg.sim_params.dt if hasattr(task_cfg, "sim_params") and task_cfg.sim_params.dt is not None else 0.01
        )

    # Offset for end-effector relative to cube position
    cube_hand_offset = torch.tensor([0.0, 0.05, 0.03], device=robot.device)

    ik_commands = torch.zeros(scene.num_envs, diff_ik_controller.action_dim, device=robot.device)

    count = 0
    iteration = 0

    log.info(f"Starting IK curriculum generation, collecting {num_samples} samples...")

    angle = torch.tensor(torch.pi / 2, device=robot.device)

    x_axis_quat = torch.tensor([0.0, 0.0, torch.cos(angle / 2), -torch.sin(angle / 2)], device=robot.device)
    while len(recorded_qpos) < num_samples and iteration < num_samples * 10:
        # Reset every max_iterations steps
        if count % max_iterations == 0:
            count = 0
            iteration += 1

            # Reset joint state
            joint_pos = robot.data.default_joint_pos.clone()
            joint_vel = robot.data.default_joint_vel.clone()
            robot.write_joint_state_to_sim(joint_pos, joint_vel)
            robot.reset()

            # Reset IK controller
            joint_pos_des = joint_pos[:, robot_entity_cfg.joint_ids].clone()
            diff_ik_controller.reset()

            # Randomly set cube position
            cube_pos_local = torch.zeros(scene.num_envs, 3, device=robot.device)
            cube_pos_local[:, 0] = torch.rand(scene.num_envs, device=robot.device) * 0.1 + 0.4  # x: 0.4-0.7
            cube_pos_local[:, 1] = torch.rand(scene.num_envs, device=robot.device) * 0.13 - 0.2  # y: -0.2-0.2
            cube_pos_local[:, 2] = torch.rand(scene.num_envs, device=robot.device) * 0.0 + 0.65  # z: 0.6-0.7

            recorded_cube_pos = torch.tensor(
                [0.488, 0.142, 0.5650, 0.9677, 0.0, 0.0, -0.2522], device=robot.device
            ).repeat(scene.num_envs, 1)
            # recorded_cube_pos = torch.tensor(
            #     [0.488, 0.142, 0.5650, 1.0, 0.0, 0.0, 0.0], device=robot.device
            # ).repeat(scene.num_envs, 1)
            recorded_cube_pos[:, :3] += scene.env_origins

            # Random cube orientation
            cube_quat_local = torch.zeros(scene.num_envs, 4, device=robot.device)
            cube_quat_local[:, 0] = 1.0  # Default quaternion (w=1)

            # Set cube position in world coordinates
            cube_pos_w = cube_pos_local + scene.env_origins
            cube_pose = torch.cat([cube_pos_w, cube_quat_local], dim=-1)
            env_ids = torch.arange(scene.num_envs, device=sim.device)
            cube.write_root_pose_to_sim(cube_pose, env_ids=env_ids)
            # cube.write_root_pose_to_sim(recorded_cube_pos, env_ids=env_ids)
            cube.write_root_velocity_to_sim(
                torch.zeros((scene.num_envs, 6), device=sim.device, dtype=torch.float32),
                env_ids=env_ids,
            )
            cube.write_data_to_sim()

        # Get real-time cube position and update IK command
        cube_pos_w = cube.data.root_pos_w  # World coordinates
        cube_quat_w = cube.data.root_quat_w  # World quaternion

        x_axis_quat_expand = x_axis_quat.unsqueeze(0).expand_as(cube_quat_w)

        root_pose_w = robot.data.root_pose_w
        root_pos_w = root_pose_w[:, 0:3]
        root_quat_w = root_pose_w[:, 3:7]
        target_ee_pos_w = cube_pos_w
        target_ee_quat_w = cube_quat_w  # Use cube orientation

        target_ee_pos_b, target_ee_quat_b = subtract_frame_transforms(
            root_pos_w, root_quat_w, target_ee_pos_w, target_ee_quat_w
        )

        # Set IK command (in root frame) based on real-time cube position
        ik_commands[:, 0:3] = target_ee_pos_b
        ik_commands[:, 3:7] = target_ee_quat_b

        diff_ik_controller.set_command(ik_commands)

        # Compute IK
        jacobian = robot.root_physx_view.get_jacobians()[:, ee_jacobi_idx, :, robot_entity_cfg.joint_ids]
        ee_pose_w = robot.data.body_pose_w[:, robot_entity_cfg.body_ids[0]]

        pos = ee_pose_w[:, :3]
        quat = ee_pose_w[:, 3:7]
        x_axis_quat_expand = x_axis_quat.unsqueeze(0).expand_as(quat)
        new_quat = quat_mul(quat, x_axis_quat_expand)
        ee_pose_w_current = torch.cat([pos, new_quat], dim=1)
        ee_pose_w = ee_pose_w_current

        joint_pos = robot.data.joint_pos[:, robot_entity_cfg.joint_ids]

        ee_pos_b, ee_quat_b = subtract_frame_transforms(
            root_pose_w[:, 0:3], root_pose_w[:, 3:7], ee_pose_w[:, 0:3], ee_pose_w[:, 3:7]
        )

        # Compute joint commands
        joint_pos_des = diff_ik_controller.compute(ee_pos_b, ee_quat_b, jacobian, joint_pos)
        # Apply actions
        robot.set_joint_position_target(joint_pos_des, joint_ids=robot_entity_cfg.joint_ids)
        robot.set_joint_position_target(fixed_joint_pos, joint_ids=fixed_joint_ids)

        scene.write_data_to_sim()
        sim.step()
        target_ee_pos_w = cube_pos_w
        goal_marker.visualize(target_ee_pos_w, cube_quat_w)

        # Check for success and record
        if (count + 1) % max_iterations == 0:
            ee_pose_w = robot.data.body_state_w[:, robot_entity_cfg.body_ids[0], 0:7]
            # Target position is cube position + offset
            cube_pos_w_current = cube.data.root_pos_w
            # target_pos = cube_pos_w_current + cube_hand_offset.unsqueeze(0)
            target_pos = cube_pos_w_current
            current_pos = ee_pose_w[:, 0:3]
            pos_diff = torch.norm(current_pos - target_pos, dim=1)

            # Record if position error is below threshold
            if torch.any(pos_diff < threshold):
                # Get joint positions for the first environment
                qpos = robot.data.joint_pos[0].cpu().numpy()
                recorded_qpos.append(qpos)

                # Get cube position relative to env_origin
                cube_pos_w = cube.data.root_pos_w[0] - scene.env_origins[0]
                cube_quat_w = cube.data.root_quat_w[0]
                cube_pose = torch.cat([cube_pos_w, cube_quat_w], dim=-1).cpu().numpy()
                recorded_cube_pos_list.append(cube_pose)

                if len(recorded_qpos) % 10 == 0:
                    log.info(f"Recorded {len(recorded_qpos)}/{num_samples} IK samples")

        count += 1

    log.info(f"Completed IK curriculum generation: {len(recorded_qpos)} samples collected")

    # Restore cube to original state based on task_cfg
    # try:
    import omni.usd
    from pxr import UsdPhysics

    stage = omni.usd.get_context().get_stage()
    if stage:
        # Restore original kinematic state for all environments
        for env_id in range(scene.num_envs):
            env_cube_prim_path = f"/World/envs/env_{env_id}/{object_name}"
            env_cube_prim = stage.GetPrimAtPath(env_cube_prim_path)
            if env_cube_prim.IsValid():
                # Use UsdPhysics.RigidBodyAPI instead of PhysxSchema
                env_rigid_body_api = UsdPhysics.RigidBodyAPI.Apply(env_cube_prim)
                if env_rigid_body_api:
                    # Restore original state from task_cfg
                    # When kinematic is disabled, gravity will be re-enabled automatically
                    kinematic_attr = env_rigid_body_api.GetKinematicEnabledAttr()
                    if kinematic_attr:
                        kinematic_attr.Set(cube_original_fixed)
                    else:
                        env_rigid_body_api.CreateKinematicEnabledAttr(cube_original_fixed)
        log.info(f"Restored cube to original state (kinematic={cube_original_fixed})")
    # except Exception as e:
    #     log.warning(f"Could not restore cube state via PhysX API: {e}")

    # remove visualize markers
    import omni.usd

    stage = omni.usd.get_context().get_stage()
    if stage:
        # Remove marker prims if they exist
        for prim_path in ["/Visuals/ee_current", "/Visuals/ee_goal"]:
            prim = stage.GetPrimAtPath(prim_path)
            if prim.IsValid():
                stage.RemovePrim(prim_path)
    return list(zip(recorded_qpos, recorded_cube_pos_list))


def save_ik_curriculum_data(curriculum_data, output_path):
    """Save IK curriculum data to npz file.

    Args:
        curriculum_data: List of (qpos, cube_pos) tuples
        output_path: Path to save the npz file
    """
    import numpy as np

    if not curriculum_data:
        log.warning("No curriculum data to save")
        return

    # Unpack the data
    qpos_list, cube_pos_list = zip(*curriculum_data)

    # Stack into arrays
    qpos_array = np.stack(qpos_list, axis=0)  # [num_samples, num_joints]
    cube_pos_array = np.stack(cube_pos_list, axis=0)  # [num_samples, 7]

    # Save to npz
    np.savez(output_path, qpos=qpos_array, cube_pos=cube_pos_array)
    log.info(f"Saved {len(curriculum_data)} IK curriculum samples to {output_path}")
    log.info(f"qpos shape: {qpos_array.shape}, cube_pos shape: {cube_pos_array.shape}")


def load_ik_curriculum_data(input_path):
    """Load IK curriculum data from npz file.

    Args:
        input_path: Path to the npz file

    Returns:
        Tuple of (qpos_array, cube_pos_array) where:
        - qpos_array: numpy array [num_samples, num_joints]
        - cube_pos_array: numpy array [num_samples, 7]
    """
    import numpy as np

    data = np.load(input_path)
    qpos_array = data["qpos"]
    cube_pos_array = data["cube_pos"]
    log.info(f"Loaded IK curriculum data from {input_path}")
    log.info(f"qpos shape: {qpos_array.shape}, cube_pos shape: {cube_pos_array.shape}")
    return qpos_array, cube_pos_array
