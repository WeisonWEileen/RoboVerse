"""A humanoid base wrapper for skillBench tasks."""

from __future__ import annotations

import logging
from collections import deque
from copy import deepcopy

import numpy as np
import torch

# Module-level logger
logger = logging.getLogger(__name__)

from humanoid_visualrl.cfg.humanoidVisualRLCfg import BaseTableHumanoidTaskCfg
from humanoid_visualrl.utils.opencv_renderer import OpenCVRenderer
from humanoid_visualrl.utils.utils import (
    get_body_reindexed_indices_from_substring,
    get_euler_xyz_tensor,
    get_joint_reindexed_indices_from_substring,
    torch_rand_float,
)
from metasim.scenario.scenario import ScenarioCfg
from metasim.types import TensorState
from metasim.utils.math import quat_apply, quat_rotate_inverse
from roboverse_learn.rl.rsl_rl.rsl_rl_wrapper import RslRlWrapper


class HumanoidBaseWrapper(RslRlWrapper):
    """Wraps Metasim environments to be compatible with rsl_rl OnPolicyRunner.

    Note that rsl_rl is designed for parallel training fully on GPU, with robust support for Isaac Gym and Isaac Lab.
    """

    def __init__(
        self,
        scenario: ScenarioCfg,
        enable_opencv_display: bool = False,
        opencv_render_env_idx: int = 0,
        opencv_fps: int = 30,
    ):
        super().__init__(scenario)

        self._env_origins = self.env.scene.env_origins.clone()

        self._parse_indices(scenario.robots[0])
        self._parse_actuation_cfg(scenario)
        self._prepare_reward_function(scenario.task)
        self._init_buffers()

        # tensor_state = self.env.get_states()

        # Initialize OpenCV renderer for real-time visualization
        self.enable_opencv_display = enable_opencv_display
        self.opencv_render_env_idx = opencv_render_env_idx
        self.opencv_renderer = None
        if self.enable_opencv_display:
            self.opencv_renderer = OpenCVRenderer(
                window_name="First Person View of Env " + str(opencv_render_env_idx),
                window_size=(640, 480),  # Upscale from 64x48 to 640x480
                fps_limit=opencv_fps,
                enable_recording=True,  # Allow video recording
                recording_path="humanoid_vision_recording.mp4",
                scroll_callback=self._handle_opencv_scroll,
            )
            self._update_opencv_status_text()
        self._load_actuator_indices(scenario.robots[0])
        

    def _load_actuator_indices(self, robot):
        """Load actuator indices from robot cfg."""
        joint_names = self.env.get_joint_names(robot.name)
        self.actuator_indices = [joint_names.index(jn) for jn in joint_names if jn in robot.actuators.keys()]
        return self.actuator_indices

    def _parse_indices(self, robot):
        """Parse rigid body indices from robot cfg."""
        try:
            feet_names = robot.feet_links
            knee_names = robot.knee_links
            elbow_names = robot.elbow_links
            wrist_names = robot.wrist_links
            torso_names = robot.torso_links
            termination_contact_names = robot.terminate_contacts_links
            penalised_contact_names = robot.penalized_contacts_links
        except Exception as e:
            print(f"Error parsing indices for robot {robot.name}: {e}")
            # raise e

        # get sorted indices for specific body links
        self.feet_indices = get_body_reindexed_indices_from_substring(
            self.env, robot.name, feet_names, device=self.device
        )
        self.knee_indices = get_body_reindexed_indices_from_substring(
            self.env, robot.name, knee_names, device=self.device
        )
        self.elbow_indices = get_body_reindexed_indices_from_substring(
            self.env, robot.name, elbow_names, device=self.device
        )
        self.wrist_indices = get_body_reindexed_indices_from_substring(
            self.env, robot.name, wrist_names, device=self.device
        )
        self.torso_indices = get_body_reindexed_indices_from_substring(
            self.env, robot.name, torso_names, device=self.device
        )
        self.termination_contact_indices = get_body_reindexed_indices_from_substring(
            self.env, robot.name, termination_contact_names, device=self.device
        )
        self.penalised_contact_indices = get_body_reindexed_indices_from_substring(
            self.env, robot.name, penalised_contact_names, device=self.device
        )

        # get upper body joint indices
        upper_joint_names = robot.upper_body_joints
        self.upper_body_joint_indices = get_joint_reindexed_indices_from_substring(
            self.env, robot.name, upper_joint_names, device=self.device
        )

        # TODO fix this
        # self.env._load_contact_sensor_idx()
        # get 

    def _parse_cfg(self, scenario):
        super()._parse_cfg(scenario)
        # per step dt
        self.dt = scenario.decimation * scenario.task.sim_params.dt
        self.command_ranges = scenario.task.command_ranges
        self.num_commands = scenario.task.command_dim

    def _parse_actuation_cfg(self, scenario):
        """Parse default joint positions and torque limits from cfg."""
        torque_limits = scenario.robots[0].torque_limits
        # for joint in scenario.robots[0].actuators.keys():
        valid_joint_names =  list(scenario.robots[0].actuators.keys()) 
        if hasattr(scenario.robots[0], "mimic_joints"):
            valid_joint_names.extend(list(scenario.robots[0].mimic_joints))
        sorted_joint_names = sorted(valid_joint_names)
        sorted_limits = [torque_limits[name] for name in sorted_joint_names]
        self.torque_limits = (
            torch.tensor(sorted_limits, device=self.device).unsqueeze(0).repeat(self.num_envs, 1)
            * scenario.task.torque_limit_scale
        )

        all_default_joint_pos = scenario.robots[0].default_joint_positions
        sorted_joint_pos = [all_default_joint_pos[name] for name in sorted_joint_names]

        self.default_joint_pd_target = (
            torch.tensor(sorted_joint_pos, device=self.device).unsqueeze(0).repeat(self.num_envs, 1)
        )
        actuator_keys = scenario.robots[0].actuators.keys()
        self.actuated_index = [sorted_joint_names.index(name) for name in actuator_keys]
        # self.actuated_index = torch.tensor(actuated_index, device=self.device)

    def _init_buffers(self):
        """Init all buffer for reward computation."""
        super()._init_buffers()

        # states
        self.dof_pos = torch.zeros(self.num_envs, self.scenario.robots[0].num_joints, device=self.device, requires_grad=False)
        self.dof_vel = torch.zeros(self.num_envs, self.scenario.robots[0].num_joints, device=self.device, requires_grad=False)
        self.root_state = torch.zeros(self.num_envs, 13, device=self.device, requires_grad=False)
        self.base_quat = torch.zeros(self.num_envs, 4, device=self.device, requires_grad=False)
        self.base_lin_vel = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)
        self.base_ang_vel = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)
        self.projected_gravity = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)
        self.base_euler_xyz = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)
        self.feet_height = torch.zeros((self.num_envs, 2), device=self.device, requires_grad=False)
        self.rand_push_force = torch.zeros(
            (self.num_envs, 3), dtype=torch.float32, device=self.device, requires_grad=False
        )
        self.rand_push_torque = torch.zeros(
            (self.num_envs, 3), dtype=torch.float32, device=self.device, requires_grad=False
        )
        self.contact_forces = torch.zeros(
            (self.num_envs, len(self.env.get_body_names(self.robot.name)), 3), dtype=torch.float32, device=self.device
        )
        self.gravity_vec = torch.tensor(self.get_axis_params(-1.0, 2), device=self.device, dtype=torch.float32).repeat((
            self.num_envs,
            1,
        ))
        self.forward_vec = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32, device=self.device).repeat((
            self.num_envs,
            1,
        ))

        # history buffer
        self.last_contacts = torch.zeros(
            self.num_envs, len(self.feet_indices), dtype=torch.bool, device=self.device, requires_grad=False
        )
        self.feet_air_time = torch.zeros(
            self.num_envs, len(self.feet_indices), dtype=torch.float, device=self.device, requires_grad=False
        )
        self.last_feet_z = 0.05 * torch.ones(
            self.num_envs, len(self.feet_indices), device=self.device, requires_grad=False
        )
        self.last_root_vel = torch.zeros(self.num_envs, 6, device=self.device, requires_grad=False)

        # control
        self._p_gains = torch.zeros(self.num_envs, self.num_actions, device=self.device, requires_grad=False)
        self._d_gains = torch.zeros(self.num_envs, self.num_actions, device=self.device, requires_grad=False)
        self._torque_limits = torch.zeros(self.num_envs, self.num_actions, device=self.device, requires_grad=False)
        self._action_scale = self.scenario.task.action_scale * torch.ones(
            self.num_envs, self.num_actions, device=self.device, requires_grad=False
        )
        dof_names = self.env.get_joint_names(self.robot.name)
        
        i = 0
        for _, dof_name in enumerate(dof_names):
            # HACK
            if dof_name not in self.robot.actuators:
                continue

            if dof_name in self.robot.mimic_joints:
                continue
            i_actuator_cfg = self.robot.actuators[dof_name]
            self._p_gains[:, i] = i_actuator_cfg.stiffness
            self._d_gains[:, i] = i_actuator_cfg.damping
            torque_limit = self.robot.torque_limits[dof_name]
            self._torque_limits[:, i] = self.scenario.task.torque_limit_scale * torque_limit
            i += 1
        # commands
        self.common_step_counter = 0
        self.commands = torch.zeros(
            self.num_envs,
            self.cfg.commands.num_commands,
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )
        self.commands_scale = torch.tensor(
            [
                self.cfg.normalization.obs_scales.lin_vel,
                self.cfg.normalization.obs_scales.lin_vel,
                self.cfg.normalization.obs_scales.ang_vel,
            ],
            device=self.device,
            requires_grad=False,
        )

        # episode length buffer
        self.episode_length_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.int32)
        self.timeout_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self.reset_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)

        # store globally for reset update and pass to obs and privileged_obs
        self.actions = torch.zeros(
            self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False
        )
        # history buffer for reward computation
        self.last_actions = torch.zeros(
            self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False
        )
        self.last_last_actions = torch.zeros(
            self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False
        )
        self.last_dof_vel = torch.zeros(self.num_envs, self.scenario.robots[0].num_joints, device=self.device, requires_grad=False)
        self.obs_history = deque(maxlen=self.cfg.frame_stack)
        self.critic_history = deque(maxlen=self.cfg.c_frame_stack)

        # history buffer for obs and privileged_obs
        for _ in range(self.cfg.frame_stack):
            self.obs_history.append(
                torch.zeros(self.num_envs, self.cfg.num_single_obs, dtype=torch.float, device=self.device)
            )
        for _ in range(self.cfg.c_frame_stack):
            self.critic_history.append(
                torch.zeros(self.num_envs, self.cfg.single_num_privileged_obs, dtype=torch.float, device=self.device)
            )

    def _compute_effort(self, actions):
        """Compute effort from actions."""
        # scale the actions (generally output from policy)
        action_scaled = self._action_scale * actions
        effort = (
            self._p_gains * (action_scaled + self.default_joint_pd_target - self.dof_pos) - self._d_gains * self.dof_vel
        )
        self.effort = torch.clip(effort, -self._torque_limits, self._torque_limits)
        effort = self.effort.to(torch.float32)
        return effort

    def _get_phase(
        self,
    ):
        cycle_time = self.cfg.reward_cfg.cycle_time
        phase = self.episode_length_buf * self.dt / cycle_time
        return phase

    def _update_history(self, tensor_state: TensorState):
        """Update history buffer at the the of the frame, called after reset."""
        self.last_last_actions[:] = self.last_actions[:]
        self.last_actions[:] = self.actions[:]
        self.last_dof_vel[:] = tensor_state.robots[self.robot.name].joint_vel

    def _prepare_reward_function(self, task: BaseTableHumanoidTaskCfg):
        self.reward_scales = task.reward_weights
        for key in list(self.reward_scales.keys()):
            scale = self.reward_scales[key]
            if scale == 0:
                self.reward_scales.pop(key)
            else:
                self.reward_scales[key] *= self.dt

        self.reward_functions = []
        self.reward_names = []
        for name, scale in self.reward_scales.items():
            if name == "termination":
                continue
            self.reward_names.append(name)
            method_name = "_reward_" + name
            fn = getattr(self, method_name, None)
            if fn is None or not callable(fn):
                raise KeyError(f"No reward function named '{method_name}' on {self.__class__.__name__}")
            self.reward_functions.append(fn)

        self.episode_sums = {
            name: torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
            for name in self.reward_scales.keys()
        }
        self.episode_metrics = {name: 0 for name in self.reward_scales.keys()}

    def _compute_reward(self, tensor_state):
        """Compute all the reward from the states provided."""
        self.rew_buf[:] = 0

        for i in range(len(self.reward_functions)):
            name = self.reward_names[i]
            rew_func_return = self.reward_functions[i](tensor_state, self.robot.name, self.cfg)
            if isinstance(rew_func_return, tuple):
                unscaled_rew, metric = rew_func_return
                self.episode_metrics[name] = metric.mean().item()
            else:
                unscaled_rew = rew_func_return
            rew = unscaled_rew * self.reward_scales[name]
            self.rew_buf += rew
            self.episode_sums[name] += rew

        if self.cfg.reward_cfg.only_positive_rewards:
            self.rew_buf[:] = torch.clip(self.rew_buf[:], min=0.0)

    def _get_gait_phase(self):
        """Add phase into states."""
        phase = self._get_phase()
        sin_pos = torch.sin(2 * torch.pi * phase)
        # Add double support phase
        stance_mask = torch.zeros((self.num_envs, 2), device=self.device)
        # left foot stance
        stance_mask[:, 0] = sin_pos >= 0
        # right foot stance
        stance_mask[:, 1] = sin_pos < 0
        # Double support phase
        stance_mask[torch.abs(sin_pos) < 0.1] = 1
        return stance_mask

    def _pre_compute_reward(self):
        """Hook method for subclasses to add custom logic before computing reward. Default implementation does nothing."""
        pass

    def _compute_observations(self, **kwargs):
        """Compute observations and priviledged observation."""
        raise NotImplementedError

    def _refreshed_tensors(self, tensor_state: TensorState):
        """Update tensors from are refreshed tensors after physics step."""
        self.root_state[:] = tensor_state.robots[self.robot.name].root_state
        self.dof_pos[:] = tensor_state.robots[self.robot.name].joint_pos
        self.dof_vel[:] = tensor_state.robots[self.robot.name].joint_vel
        self.base_quat[:] = tensor_state.robots[self.robot.name].root_state[:, 3:7]
        self.base_lin_vel[:] = quat_rotate_inverse(
            self.base_quat, tensor_state.robots[self.robot.name].root_state[:, 7:10]
        )
        self.base_ang_vel[:] = quat_rotate_inverse(
            self.base_quat, tensor_state.robots[self.robot.name].root_state[:, 10:13]
        )
        self.projected_gravity[:] = quat_rotate_inverse(self.base_quat, self.gravity_vec)
        self.base_euler_xyz = get_euler_xyz_tensor(self.base_quat)
        # self.contact_forces[:] = tensor_state.extras["net_contact_force"][:]
        # FIXME debug here
        # self.contact_forces[:] = self.env.contact_sensor.data.net_forces_w[
        #     :, self.env.get_body_reindex(self.robot.name), :
        # ]

    def _check_reset(self):
        # reset_buf = torch.any(
        #     torch.norm(self.contact_forces[:, self.termination_contact_indices, :], dim=-1) > 1.0, dim=1
        # )
        # reindex = self.env.get_body_reindex(self.robot.name)
        # contact_forces = self.env.contact_sensor.data.net_forces_w
        # reset_buf = torch.any(
        #     torch.norm(contact_forces[:, self.env.termination_contact_indices, :], dim=-1) > 1.0,
        #     dim=1,
        # )
        # self.reset_buf = torch.logical_or(self.timeout_buf, reset_buf)
        raise NotImplementedError

    def _post_physics_step(self):
        """After physics step, compute reward, get obs and privileged_obs, resample command."""
        self.common_step_counter += 1
        self.episode_length_buf += 1
        self.timeout_buf = self.episode_length_buf >= self.cfg.max_episode_length_s / self.dt
        self._post_physics_step_callback()
        tensor_state = self.env.get_states()

        self._refreshed_tensors(tensor_state)
        self._check_reset()

        reset_env_idx = self.reset_buf.nonzero(as_tuple=False).flatten().tolist()

        self._pre_compute_reward()
        self._compute_reward(tensor_state)
        self._reset(reset_env_idx)

        # compute obs for actor,  privileged_obs for critic network
        self._compute_observations()
        self._update_history(tensor_state)

    def _post_physics_step_evaluate(self):
        """After physics step, compute reward, get obs and privileged_obs, resample command."""
        self.common_step_counter += 1
        self.episode_length_buf += 1
        # self.timeout_buf = self.episode_length_buf >= self.cfg.max_episode_length_s / self.dt
        # self._post_physics_step_callback()
        tensor_state = self.env.get_states()

        self._refreshed_tensors(tensor_state)
        # self._check_reset()

        # compute obs for actor,  privileged_obs for critic network
        self._compute_observations()
        self._update_history(tensor_state)

    def update_command_curriculum(self, env_ids):
        """Implements a curriculum of increasing commands."""
        # If the tracking reward is above 80% of the maximum, increase the range of commands
        if (
            torch.mean(self.episode_sums["tracking_lin_vel"][env_ids]) / self.max_episode_length
            > 0.8 * self.reward_scales["tracking_lin_vel"]
        ):
            self.command_ranges.lin_vel_x[0] = np.clip(
                self.command_ranges.lin_vel_x[0] - 0.5, -self.cfg.commands.max_curriculum, 0.0
            )
            self.command_ranges.lin_vel_x[1] = np.clip(
                self.command_ranges.lin_vel_x[1] + 0.5, 0.0, self.cfg.commands.max_curriculum
            )

    def clip_actions(self, actions):
        """Clip actions based on cfg."""
        clip_action_limit = self.cfg.normalization.clip_actions
        return torch.clip(actions, -clip_action_limit, clip_action_limit).to(self.device)

    def _pre_physics_step(self, actions):
        """Apply action smoothing and wrap actions as dict before physics step."""
        delay = torch.rand((self.num_envs, 1), device=self.device)
        actions = (1 - delay) * actions.to(self.device) + delay * self.actions
        clipped_actions = self.clip_actions(actions)
        self.actions = clipped_actions
        if not self.env.control_effort_mode:
            self.actions += self._action_scale * (self.default_joint_pd_target[:, self.actuated_index] + self.actions)

        return self.actions

    def _physics_step(self, action) -> None:
        self.env.set_dof_targets(action)
        for _ in range(self.cfg.decimation):
            # refresh dof states
            # tensor_state = self.env.get_states()
            # self.dof_pos = tensor_state.robots[self.robot.name].joint_pos
            # self.dof_vel = tensor_state.robots[self.robot.name].joint_vel
            
            if self.env.control_effort_mode:
                # test more light weight
                reindex = self.env.get_joint_reindex(self.robot.name)
                self.dof_pos = self.env.scene.articulations[self.robot.name].data.joint_pos[:, reindex]
                self.dof_vel = self.env.scene.articulations[self.robot.name].data.joint_vel[:, reindex]
                torques = self._compute_effort(action)
                self.env.set_dof_targets(torques)

            self.env.simulate()

    def step(self, actions):
        """Perform one training step and return observation, reward, and reset buffers."""
        action = self._pre_physics_step(actions)
        # start_step = time.time()
        self._physics_step(action)
        self._post_physics_step()
        # end_step = time.time()
        # print(f"Step time: {end_step - start_step}")
        return self.obs_buf, self.rew_buf, self.reset_buf, self.extra_buf

    def step_evaluate(self, actions):
        """Perform one evaluation step without reward computation side effects."""
        action = self._pre_physics_step(actions)
        self._physics_step(action)
        self._post_physics_step_evaluate()
        return self.obs_buf, self.rew_buf, self.reset_buf, self.extra_buf

    def _pre_reset_hook(self, env_ids):
        """Hook method for subclasses to add custom logic before resetting."""
        pass

    def _post_reset_hook(self, env_ids):
        """Hook method for subclasses to add custom logic after resetting."""
        pass

    def _load_actuator_indices(self, robot):
        """Load actuator indices from robot cfg."""
        joint_names = self.env.get_joint_names(robot.name)
        self.actuator_indices = [joint_names.index(jn) for jn in joint_names if jn in robot.actuators.keys()]
        return self.actuator_indices

    def _reset(self, env_ids=None):
        """Reset the wrapper."""
        if env_ids is None:
            env_ids = list(range(self.num_envs))
        if len(env_ids) == 0:
            return
        self._pre_reset_hook(env_ids)
        self.env.set_states(self.init_states, env_ids)
        self._resample_commands(env_ids)

        # for isaacsim internal data buffer reset
        self.env.scene.reset(env_ids)
        self.env.scene.write_data_to_sim()
        self.env.sim.forward()

        # reset state buffer in the wrapper
        self.actions[env_ids] = self.init_states.robots[self.robot.name].joint_pos[env_ids][:, self.actuator_indices]
        # self.last_actions[env_ids] = 0.0
        # self.last_last_actions[env_ids] = 0.0
        self.last_dof_vel[env_ids] = 0.0
        self.episode_length_buf[env_ids] = 0
        # self.feet_air_time[env_ids] = 0.0
        self.root_state[env_ids] = self.init_states.robots[self.robot.name].root_state[env_ids]

        #
        self.dof_pos[env_ids] = self.init_states.robots[self.robot.name].joint_pos[env_ids]
        self.dof_vel[env_ids] = 0.0

        self._post_reset_hook(env_ids)

        if not self.cfg.task_name == "active_vision":
            self.base_quat[env_ids] = (
                torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device, dtype=torch.float32)
                .unsqueeze(0)
                .repeat(len(env_ids), 1)
            )
            self.base_euler_xyz = get_euler_xyz_tensor(self.base_quat)
            self.projected_gravity[env_ids] = quat_rotate_inverse(self.base_quat[env_ids], self.gravity_vec[env_ids])

        self.extra_buf["episode"] = {}
        for key in self.episode_sums.keys():
            self.extra_buf["episode"]["rew_" + key] = (
                torch.mean(self.episode_sums[key][env_ids]) / self.cfg.max_episode_length_s
            )
            self.episode_sums[key][env_ids] = 0.0

        # log metrics
        self.extra_buf["episode_metrics"] = deepcopy(self.episode_metrics)

        # reset env handler state buffer
        for i in range(self.obs_history.maxlen):
            self.obs_history[i][env_ids] *= 0
        for i in range(self.critic_history.maxlen):
            self.critic_history[i][env_ids] *= 0

    def _post_physics_step_callback(self):
        env_ids = (
            (self.episode_length_buf % int(self.cfg.commands.resampling_time / self.dt) == 0)
            .nonzero(as_tuple=False)
            .flatten()
        )
        if len(env_ids) > 0:
            self._resample_commands(env_ids)

        if self.cfg.commands.heading_command:
            forward = quat_apply(self.base_quat, self.forward_vec)
            heading = torch.atan2(forward[:, 1], forward[:, 0])
            self.commands[:, 2] = torch.clip(0.5 * self.wrap_to_pi(self.commands[:, 3] - heading), -1.0, 1.0)

        # self._randomize()
        self._update_curriculum()

    def _update_curriculum(self):
        pass

    def _resample_commands(self, env_ids):
        self.commands[env_ids, 0] = torch_rand_float(
            self.command_ranges.lin_vel_x[0],
            self.command_ranges.lin_vel_x[1],
            (len(env_ids), 1),
            device=self.device,
        ).squeeze(1)
        self.commands[env_ids, 1] = torch_rand_float(
            self.command_ranges.lin_vel_y[0],
            self.command_ranges.lin_vel_y[1],
            (len(env_ids), 1),
            device=self.device,
        ).squeeze(1)
        if self.cfg.commands.heading_command:
            self.commands[env_ids, 3] = torch_rand_float(
                self.command_ranges.heading[0],
                self.command_ranges.heading[1],
                (len(env_ids), 1),
                device=self.device,
            ).squeeze(1)
        else:
            self.commands[env_ids, 2] = torch_rand_float(
                self.command_ranges.ang_vel_yaw[0],
                self.command_ranges.ang_vel_yaw[1],
                (len(env_ids), 1),
                device=self.device,
            ).squeeze(1)

        # set small commands to zero
        self.commands[env_ids, :2] *= (torch.norm(self.commands[env_ids, :2], dim=1) > 0.2).unsqueeze(1)

    def _randomize(self):
        pass

    def _push_robots(self):
        """Randomly set robot's root velocity to simulate a push."""
        if self.cfg.random_push.enabled and self.common_step_counter % self.cfg.random_push.push_interval == 0:
            pass  # due to performace issue, we generally not use get_states
            # max_vel = self.cfg.random_push.max_push_vel_xy
            # max_push_angular = self.cfg.random_push.max_push_ang_vel
            # tensor_states = self.env.get_states()
            # self.rand_push_force[:, :2] += torch_rand_float(-max_vel, max_vel, (self.num_envs, 2), device=self.device)
            # tensor_states.robots[self.robot.name].root_state[:, 0:2] += self.rand_push_force[:, :2]
            # self.rand_push_torque = torch_rand_float(
            #     -max_push_angular, max_push_angular, (self.num_envs, 3), device=self.device
            # )
            # tensor_states.robots[self.robot.name].root_state[:, 10:13] = self.rand_push_torque

    def _update_marker_viz(self):
        # convert to world frame
        world_pos = self.ref_wrist_pos[:, :, :3] + self._env_origins[:, None, :3]
        pos = world_pos.reshape(-1, 3)
        ori = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).repeat(pos.shape[0], 1)
        idx = torch.zeros(pos.shape[0], dtype=torch.long, device=self.device)
        self.marker_viz.visualize(pos, ori, marker_indices=idx)

    def _get_phase(
        self,
    ):
        cycle_time = self.cfg.reward_cfg.cycle_time
        phase = self.episode_length_buf * self.dt / cycle_time
        return phase

    @staticmethod
    def get_axis_params(value, axis_idx, x_value=0.0, dtype=np.float64, n_dims=3):
        """Construct arguments to `Vec` according to axis index."""
        zs = np.zeros((n_dims,))
        assert axis_idx < n_dims, "the axis dim should be within the vector dimensions"
        zs[axis_idx] = 1.0
        params = np.where(zs == 1.0, value, zs)
        params[0] = x_value
        return list(params.astype(dtype))

    @staticmethod
    def wrap_to_pi(angles: torch.Tensor) -> torch.Tensor:
        """Wrap angles to [-pi, pi]."""
        angles %= 2 * np.pi
        angles -= 2 * np.pi * (angles > np.pi)
        return angles

    def _handle_opencv_scroll(self, direction: int):
        """Handle mouse scroll events to switch the rendered environment index."""
        if self.num_envs == 0 or direction == 0:
            return
        prev_idx = self.opencv_render_env_idx
        self.opencv_render_env_idx = int((self.opencv_render_env_idx + direction) % self.num_envs)
        if prev_idx != self.opencv_render_env_idx:
            logger.info("OpenCVRenderer switched to env %d", self.opencv_render_env_idx)
            self._update_opencv_status_text()

    def _update_opencv_status_text(self):
        """Update status text shown on the OpenCV window."""
        if self.opencv_renderer is not None:
            self.opencv_renderer.set_status_text(f"Env {self.opencv_render_env_idx}")

    def _get_joint_masking_indices(self):
        mask_joint_names = self.cfg.mask_joint_names
        self.mask_joint_indices = get_joint_reindexed_indices_from_substring(
            self.env, self.robot.name, mask_joint_names, device=self.device
        )
        self.action_masking = torch.ones(self.num_actions, device=self.device, dtype=torch.float)
        self.action_masking[self.mask_joint_indices] = 0.0