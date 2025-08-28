"""A humanoid base wrapper for skillBench tasks."""

from __future__ import annotations

from collections import deque
from copy import deepcopy

import numpy as np
import torch

from active_vision.cfg.scenario_cfg import BaseTableHumanoidTaskCfg
from active_vision.utils.utils import (
    get_body_reindexed_indices_from_substring,
    sample_int_from_float,
    sample_wp,
    torch_rand_float,
)
from metasim.scenario.scenario import ScenarioCfg
from metasim.types import TensorState
from roboverse_learn.rl.rsl_rl.rsl_rl_wrapper import RslRlWrapper


class ActiveVisionWrapper(RslRlWrapper):
    """Wraps Metasim environments to be compatible with rsl_rl OnPolicyRunner.

    Note that rsl_rl is designed for parallel training fully on GPU, with robust support for Isaac Gym and Isaac Lab.
    """

    def __init__(self, scenario: ScenarioCfg):
        super().__init__(scenario)
        self.up_axis_idx = 2
        self._env_origins = self.env.scene.env_origins.clone()

        self._parse_rigid_body_indices(scenario.robots[0])
        self._parse_actuation_cfg(scenario)
        self._prepare_reward_function(scenario.task)
        self._init_buffers()

        tensor_state = self.env.get_states()
        self._init_target_wp(tensor_state)
        self.marker_viz = self.env.init_marker_viz()

    def _parse_rigid_body_indices(self, robot):
        """Parse rigid body indices from robot cfg."""
        feet_names = robot.feet_links
        knee_names = robot.knee_links
        elbow_names = robot.elbow_links
        wrist_names = robot.wrist_links

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

    def _parse_cfg(self, scenario):
        super()._parse_cfg(scenario)
        # per step dt
        self.dt = scenario.decimation * scenario.task.sim_params.dt
        self.command_ranges = scenario.task.command_ranges
        self.num_commands = scenario.task.command_dim

    def _parse_actuation_cfg(self, scenario):
        """Parse default joint positions and torque limits from cfg."""
        torque_limits = scenario.robots[0].torque_limits
        sorted_joint_names = sorted(torque_limits.keys())
        sorted_limits = [torque_limits[name] for name in sorted_joint_names]
        self.torque_limits = (
            torch.tensor(sorted_limits, device=self.device).unsqueeze(0).repeat(self.num_envs, 1)
            * scenario.task.torque_limit_scale
        )

        default_joint_pos = scenario.robots[0].default_joint_positions
        sorted_joint_pos = [default_joint_pos[name] for name in sorted_joint_names]
        self.default_joint_pd_target = (
            torch.tensor(sorted_joint_pos, device=self.device).unsqueeze(0).repeat(self.num_envs, 1)
        )

    def _init_buffers(self):
        """Init all buffer for reward computation."""
        super()._init_buffers()

        # states
        self._dof_pos = torch.zeros(self.num_envs, self.num_actions, device=self.device, requires_grad=False)
        self._dof_vel = torch.zeros(self.num_envs, self.num_actions, device=self.device, requires_grad=False)

        # control
        self._p_gains = torch.zeros(self.num_envs, self.num_actions, device=self.device, requires_grad=False)
        self._d_gains = torch.zeros(self.num_envs, self.num_actions, device=self.device, requires_grad=False)
        self._torque_limits = torch.zeros(self.num_envs, self.num_actions, device=self.device, requires_grad=False)
        self._action_scale = self.scenario.task.action_scale * torch.ones(
            self.num_envs, self.num_actions, device=self.device, requires_grad=False
        )
        dof_names = self.env.get_joint_names(self.robot.name, sort=True)
        for i, dof_name in enumerate(dof_names):
            i_actuator_cfg = self.robot.actuators[dof_name]
            self._p_gains[:, i] = i_actuator_cfg.stiffness
            self._d_gains[:, i] = i_actuator_cfg.damping
            torque_limit = self.robot.torque_limits[dof_name]
            self._torque_limits[:, i] = self.scenario.task.torque_limit_scale * torque_limit

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

        # store globally for reset update and pass to obs and privileged_obs
        self.actions = torch.zeros(
            self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False
        )

        # history buffer for reward computation
        self._last_actions = torch.zeros(
            self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False
        )
        self.last_last_actions = torch.zeros(
            self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False
        )
        self._last_dof_vel = torch.zeros(self.num_envs, self.num_actions, device=self.device, requires_grad=False)
        self.obs_history = deque(maxlen=self.cfg.frame_stack)
        self.critic_history = deque(maxlen=self.cfg.c_frame_stack)
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
        _effort = self._p_gains * (action_scaled + self.default_joint_pd_target - self._dof_pos) - self._d_gains * self._dof_vel
        self._effort = torch.clip(_effort, -self._torque_limits, self._torque_limits)
        effort = self._effort.to(torch.float32)
        return effort

    def _get_phase(
        self,
    ):
        cycle_time = self.cfg.reward_cfg.cycle_time
        phase = self._episode_length_buf * self.dt / cycle_time
        return phase

    def _update_history(self, tensor_state: TensorState):
        """Update history buffer at the the of the frame, called after reset."""
        # we should always make a copy here
        # check whether torch.clone is necessary
        self.last_last_actions[:] = torch.clone(self._last_actions[:])
        self._last_actions[:] = self.actions[:]
        self._last_dof_vel[:] = tensor_state.robots[self.robot.name].joint_vel

    def _prepare_reward_function(self, task: BaseTableHumanoidTaskCfg):
        """Prepares a list of reward functions, which will be called to compute the total reward."""
        self.reward_scales = task.reward_weights
        for key in list(self.reward_scales.keys()):
            scale = self.reward_scales[key]
            if scale == 0:
                self.reward_scales.pop(key)
            else:
                self.reward_scales[key] *= self.dt
        # prepare list of functions
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

        # reward episode sums
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

    def _compute_observations(self, tensor_state):
        """Compute observations and priviledged observation."""
        raise NotImplementedError

    def _update_refreshed_tensors(self, tensor_state: TensorState):
        """Update tensors from are refreshed tensors after physics step."""
        self._dof_pos = tensor_state.robots[self.robot.name].joint_pos
        self._dof_vel = tensor_state.robots[self.robot.name].joint_vel

    def _post_physics_step(self):
        """After physics step, compute reward, get obs and privileged_obs, resample command."""
        self.common_step_counter += 1
        self._episode_length_buf += 1
        self.reset_buf = self._episode_length_buf >= self.cfg.max_episode_length_s / self.dt

        self._post_physics_step_callback()

        tensor_state = self.env.get_states()
        # update refreshed tensors from simulaor
        self._update_refreshed_tensors(tensor_state)
        # compute the reward
        self._compute_reward(tensor_state)
        # reset envs
        reset_env_idx = self.reset_buf.nonzero(as_tuple=False).flatten().tolist()
        self.reset(reset_env_idx)
        self._update_target_wp(reset_env_idx)
        self._update_marker_viz()

        # compute obs for actor,  privileged_obs for critic network
        self._compute_observations(tensor_state)
        self._update_history(tensor_state)

        return self.obs_buf, self.privileged_obs_buf, self.rew_buf

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
        return self.actions

    def _physics_step(self, action) -> None:
        self.env.set_dof_targets(action)

        for _ in range(self.cfg.decimation):
            # refresh dof states
            tensor_state = self.env.get_states()
            self._dof_pos = tensor_state.robots[self.robot.name].joint_pos
            self._dof_vel = tensor_state.robots[self.robot.name].joint_vel
            # compute torques
            torques = self._compute_effort(action)
            # assign torqus
            self.env.set_dof_targets(torques)
            # step physics
            self.env.simulate()

    def step(self, actions):
        action = self._pre_physics_step(actions)
        self._physics_step(action)
        self._post_physics_step()
        return self.obs_buf, self.rew_buf, self.reset_buf, self.extra_buf

    def reset(self, env_ids=None):
        # """"""
        if env_ids is None:
            env_ids = list(range(self.num_envs))
        if len(env_ids) == 0:
            return
        self.env.set_states(self.init_states, env_ids)
        self._resample_commands(env_ids)

        # reset state buffer in the wrapper
        self.actions[env_ids] = 0.0
        self._last_actions[env_ids] = 0.0
        self.last_last_actions[env_ids] = 0.0
        self._last_dof_vel[env_ids] = 0.0
        self._episode_length_buf[env_ids] = 0

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
        # set small commands to zero
        self.commands[env_ids, :2] *= (torch.norm(self.commands[env_ids, :2], dim=1) > 0.2).unsqueeze(1)

    def _post_physics_step_callback(self):
        """Callback called before computing terminations, rewards, and observations."""
        env_ids = (
            (self._episode_length_buf % int(self.cfg.commands.resampling_time / self.dt) == 0)
            .nonzero(as_tuple=False)
            .flatten()
        )
        if len(env_ids) > 0:
            self._resample_commands(env_ids)

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

    # only for reaching

    def _compute_observations(self, tensor_states: TensorState) -> None:
        q = (
            tensor_states.robots[self.robot.name].joint_pos - self.default_joint_pd_target
        ) * self.cfg.normalization.obs_scales.dof_pos
        dq = tensor_states.robots[self.robot.name].joint_vel * self.cfg.normalization.obs_scales.dof_vel

        wrist_pos = tensor_states.robots[self.robot.name].body_state[:, self.wrist_indices, :7]
        diff = wrist_pos - self.ref_wrist_pos

        ref_wrist_pos_obs = torch.flatten(self.ref_wrist_pos, start_dim=1)  # [num_envs, 14]
        wrist_pos_obs = torch.flatten(wrist_pos, start_dim=1)  # [num_envs, 14]
        diff_obs = torch.flatten(diff, start_dim=1)  # [num_envs, 14]

        self.privileged_obs_buf = torch.cat(
            (
                ref_wrist_pos_obs,  # 14
                wrist_pos_obs,  # 14
                q,  # |A|
                dq,  # |A|
                self.actions,  # |A|
                diff_obs,
            ),
            dim=-1,
        )

        obs_buf = torch.cat(
            (
                diff_obs,  # 3
                q,  # |A|
                dq,  # |A|
                self.actions,
            ),
            dim=-1,
        )

        obs_now = obs_buf.clone()
        self.obs_history.append(obs_now)
        self.critic_history.append(self.privileged_obs_buf)
        obs_buf_all = torch.stack([self.obs_history[i] for i in range(self.obs_history.maxlen)], dim=1)
        self.obs_buf = obs_buf_all.reshape(self.num_envs, -1)
        self.privileged_obs_buf = torch.cat([self.critic_history[i] for i in range(self.cfg.c_frame_stack)], dim=1)
        self.privileged_obs_buf = torch.clip(
            self.privileged_obs_buf, -self.cfg.normalization.clip_observations, self.cfg.normalization.clip_observations
        )

    def _update_target_wp(self, reset_env_ids):
        """Update target wrist positions."""
        # self.target_wp_i specifies which seq to use for each env, and self.target_wp_j specifies the timestep in the seq
        self.ref_wrist_pos = (
            self.target_wp[self.target_wp_i, self.target_wp_j] + self.ori_wrist_pos
        )  # [num_envs, 2, 7], two hands
        self.delayed_obs_target_wp = self.target_wp[
            self.target_wp_i, torch.maximum(self.target_wp_j - self.delayed_obs_target_wp_steps_int, torch.tensor(0))
        ]
        resample_i = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        if self.common_step_counter % self.target_wp_update_steps_int == 0:
            self.target_wp_j += 1
            wp_eps_end_bool = self.target_wp_j >= self.num_wp
            self.target_wp_j = torch.where(wp_eps_end_bool, torch.zeros_like(self.target_wp_j), self.target_wp_j)
            resample_i[wp_eps_end_bool.nonzero(as_tuple=False).flatten()] = True
            self.target_wp_update_steps_int = sample_int_from_float(self.target_wp_update_steps)
            self.delayed_obs_target_wp_steps_int = sample_int_from_float(self.delayed_obs_target_wp_steps)
        if self.cfg.humanoid_extra_cfg.resample_on_env_reset:
            self.target_wp_j[reset_env_ids] = 0
            resample_i[reset_env_ids] = True
        self.target_wp_i = torch.where(
            resample_i, torch.randint(0, self.num_pairs, (self.num_envs,), device=self.device), self.target_wp_i
        )

    def _update_marker_viz(self):
        # convert to world frame
        world_pos = self.ref_wrist_pos[:, :, :3] + self._env_origins[:, None, :3]
        pos = world_pos.reshape(-1, 3)
        ori = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).repeat(pos.shape[0], 1)
        idx = torch.zeros(pos.shape[0], dtype=torch.long, device=self.device)
        self.marker_viz.visualize(pos, ori, marker_indices=idx)

    def _init_target_wp(self, tensor_state: TensorState) -> None:
        self.ori_wrist_pos = (
            tensor_state.robots[self.robot.name].body_state[:, self.wrist_indices, :7].clone()
        )  # [num_envs, 2, 7], two hands
        self.target_wp, self.num_pairs, self.num_wp = sample_wp(
            self.device, num_points=2000000, num_wp=10, ranges=self.command_ranges
        )  # relative, self.target_wp.shape=[num_pairs, num_wp, 2, 7]
        self.target_wp_i = torch.randint(
            0, self.num_pairs, (self.num_envs,), device=self.device
        )  # for each env, choose one seq, [num_envs]
        self.target_wp_j = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )  # for each env, the timestep in the seq is initialized to 0, [num_envs]
        self.target_wp_dt = 1 / self.cfg.humanoid_extra_cfg.freq
        self.target_wp_update_steps = self.target_wp_dt / self.dt  # not necessary integer
        assert self.dt <= self.target_wp_dt, (
            f"self.dt {self.dt} must be less than self.target_wp_dt {self.target_wp_dt}"
        )
        self.target_wp_update_steps_int = sample_int_from_float(self.target_wp_update_steps)

        self.ref_wrist_pos = None
        self.ref_action = self.default_joint_pd_target
        self.delayed_obs_target_wp = None
        self.delayed_obs_target_wp_steps = self.cfg.humanoid_extra_cfg.delay / self.target_wp_dt
        self.delayed_obs_target_wp_steps_int = sample_int_from_float(self.delayed_obs_target_wp_steps)
        self._update_target_wp(torch.tensor([], dtype=torch.long, device=self.device))

    # ==== reward functions ====
    def _reward_wrist_pos(self, tensor_state: TensorState, robot_name: str, cfg: BaseTableHumanoidTaskCfg):
        """Reward for reaching the target position."""
        wrist_pos = tensor_state.robots[robot_name].body_state[:, self.wrist_indices, :7]  # [num_envs, 2, 7], two hands
        wrist_pos_diff = (
            wrist_pos[:, :, :3] - self.ref_wrist_pos[:, :, :3]
        )  # [num_envs, 2, 3], two hands, position only
        wrist_pos_diff = torch.flatten(wrist_pos_diff, start_dim=1)  # [num_envs, 6]
        wrist_pos_error = torch.mean(torch.abs(wrist_pos_diff), dim=1)
        return torch.exp(-4 * wrist_pos_error), wrist_pos_error

    def _reward_upper_body_pos(
        self, states: TensorState, robot_name: str, cfg: BaseTableHumanoidTaskCfg
    ) -> torch.Tensor:
        """Keep upper body joints close to default positions."""
        upper_body_diff = states.robots[robot_name].joint_pos - self.default_joint_pd_target
        upper_body_error = torch.mean(torch.abs(upper_body_diff), dim=1)
        return torch.exp(-4 * upper_body_error), upper_body_error

    def _reward_default_joint_pos(
        self, states: TensorState, robot_name: str, cfg: BaseTableHumanoidTaskCfg
    ) -> torch.Tensor:
        """Keep joint positions close to defaults (penalize yaw/roll)."""
        joint_diff = states.robots[robot_name].joint_pos - self.default_joint_pd_target
        return -0.01 * torch.norm(joint_diff, dim=1)

    def _reward_torques(self, states: TensorState, robot_name: str, cfg: BaseTableHumanoidTaskCfg) -> torch.Tensor:
        """Penalize high torques."""
        return torch.sum(torch.square(states.robots[robot_name].joint_effort_target), dim=1)

    def _reward_dof_vel(self, states: TensorState, robot_name: str, cfg: BaseTableHumanoidTaskCfg) -> torch.Tensor:
        """Penalize high dof velocities."""
        return torch.sum(torch.square(states.robots[robot_name].joint_vel), dim=1)

    def _reward_dof_acc(self, states: TensorState, robot_name: str, cfg: BaseTableHumanoidTaskCfg) -> torch.Tensor:
        """Penalize high DOF accelerations."""
        return torch.sum(
            torch.square((self._last_dof_vel - self._dof_vel) / self.dt),
            dim=1,
        )
