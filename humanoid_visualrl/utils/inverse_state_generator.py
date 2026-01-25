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
    from isaaclab.utils.math import subtract_frame_transforms

    ISAACLAB_AVAILABLE = True
except ImportError:
    ISAACLAB_AVAILABLE = False
    log.warning("IsaacLab not available, IK inverse curriculum will be skipped")


def generate_ik_curriculum_data(env, task_cfg, num_samples=200, threshold=0.1, max_iterations=150):
    """Generate IK curriculum data by inverse kinematics to cube positions.

    This function uses differential IK to move the robot's end-effector to various
    positions near the cube, and records successful configurations (qpos, cube_pos)
    that can be used as initial states for curriculum learning.

    Args:
        env: The environment instance (must have handler.scene and handler.sim)
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

    # Check if handler has scene (IsaacSim/IsaacLab)
    if not hasattr(env.handler, "scene"):
        log.warning("Handler does not have scene attribute, skipping IK curriculum generation")
        return []

    scene = env.handler.scene
    sim = env.handler.sim

    # Get robot and object names from task_cfg
    robot_name = task_cfg.robot
    # Find object name (usually "object" or "cube")
    object_name = None
    if hasattr(task_cfg, "objects") and len(task_cfg.objects) > 0:
        # Find the first non-robot object
        for obj in task_cfg.objects:
            if hasattr(obj, "name") and obj.name != robot_name:
                object_name = obj.name
                break

    if object_name is None:
        # Try to infer from init_states
        if hasattr(task_cfg, "init_states") and len(task_cfg.init_states) > 0:
            objects_dict = task_cfg.init_states[0].get("objects", {})
            if objects_dict:
                object_name = next(iter(objects_dict.keys()))

    if object_name is None:
        log.warning("Could not find object name, skipping IK curriculum generation")
        return []

    log.info(f"Generating IK curriculum data for robot: {robot_name}, object: {object_name}")

    # Get robot and cube from scene
    try:
        robot = scene.articulations[robot_name]
        cube = scene.rigid_objects[object_name]
    except KeyError as e:
        log.warning(f"Could not find robot or object in scene: {e}, skipping IK curriculum generation")
        return []

    # Create IK controller
    diff_ik_cfg = DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls")
    diff_ik_controller = DifferentialIKController(diff_ik_cfg, num_envs=scene.num_envs, device=sim.device)

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
    for jn in robot_cfg.default_joint_positions.keys():
        if jn not in robot_entity_cfg.joint_names and "R_arm_" not in jn:
            fixed_joint_ids.append(robot.joint_names.index(jn))
            fixed_joint_pos.append(robot_cfg.default_joint_positions[jn])
    fixed_joint_ids = torch.tensor(fixed_joint_ids, device=robot.device)
    fixed_joint_pos = torch.tensor(fixed_joint_pos, device=robot.device)

    # Initialize data collection
    recorded_qpos = []
    recorded_cube_pos = []
    # Get physics dt from handler or use default
    if hasattr(env.handler, "physics_dt"):
        sim_dt = env.handler.physics_dt
    elif hasattr(sim, "get_physics_dt"):
        sim_dt = sim.get_physics_dt()
    else:
        sim_dt = (
            task_cfg.sim_params.dt if hasattr(task_cfg, "sim_params") and task_cfg.sim_params.dt is not None else 0.01
        )

    # Define goals (cube positions to reach)
    ee_goals = [
        [0.5, -0.2, 0.65, 0, 0, -0.7071, 0.7071],
    ]
    ee_goals = torch.tensor(ee_goals, device=sim.device)

    # Offset for cube placement relative to end-effector
    offset = torch.tensor([0.0, 0.0, 0.06], device=robot.device)
    cube_hand_offset = torch.tensor([0.0, 0.05, 0.03], device=robot.device)

    current_goal_idx = 0
    ik_commands = torch.zeros(scene.num_envs, diff_ik_controller.action_dim, device=robot.device)
    ik_commands[:] = ee_goals[current_goal_idx]

    count = 0
    iteration = 0
    delta = torch.zeros_like(ik_commands)  # Initialize delta

    log.info(f"Starting IK curriculum generation, collecting {num_samples} samples...")

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
            ik_commands[:] = ee_goals[current_goal_idx]
            joint_pos_des = joint_pos[:, robot_entity_cfg.joint_ids].clone()
            diff_ik_controller.reset()

            # Add random noise to target position
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

            # Change goal periodically
            current_goal_idx = (current_goal_idx + 1) % len(ee_goals)
        else:
            # Compute IK
            jacobian = robot.root_physx_view.get_jacobians()[:, ee_jacobi_idx, :, robot_entity_cfg.joint_ids]
            ee_pose_w = robot.data.body_pose_w[:, robot_entity_cfg.body_ids[0]]
            root_pose_w = robot.data.root_pose_w
            joint_pos = robot.data.joint_pos[:, robot_entity_cfg.joint_ids]

            # Compute frame in root frame
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
        scene.update(sim_dt)

        # Check for success and record
        if (count + 1) % max_iterations == 0:
            ee_pose_w = robot.data.body_state_w[:, robot_entity_cfg.body_ids[0], 0:7]
            target_pos = ik_commands[:, 0:3] + delta[:, 0:3] + scene.env_origins
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
                recorded_cube_pos.append(cube_pose)

                if len(recorded_qpos) % 10 == 0:
                    log.info(f"Recorded {len(recorded_qpos)}/{num_samples} IK samples")

        count += 1

    log.info(f"Completed IK curriculum generation: {len(recorded_qpos)} samples collected")

    # Return as list of tuples
    return list(zip(recorded_qpos, recorded_cube_pos))
