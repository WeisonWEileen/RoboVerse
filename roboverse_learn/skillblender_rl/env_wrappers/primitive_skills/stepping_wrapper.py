"""SkillBlench wrapper for training primitive skill: walking."""

# ruff: noqa: F405
from __future__ import annotations

import torch

from metasim.cfg.scenario import ScenarioCfg
from metasim.types import EnvState
from metasim.utils.humanoid_robot_util import *
from roboverse_learn.skillblender_rl.env_wrappers.base.humanoid_base_wrapper import HumanoidBaseWrapper


class WalkingWrapper(HumanoidBaseWrapper):
    """
    Wrapper for Skillbench:walking
    """

    def __init__(self, scenario: ScenarioCfg):
        # TODO check compatibility for other simulators
        super().__init__(scenario)
        env_states = self.env.reset()

    def _compute_ref_state(self):
        """compute reference target position for walking task."""
        phase = self._get_phase()
        sin_pos = torch.sin(2 * torch.pi * phase)
        sin_pos_l = sin_pos.clone()
        sin_pos_r = sin_pos.clone()
        self.ref_dof_pos = torch.zeros(
            self.num_envs, self.env.handler.robot_num_dof, device=self.device, requires_grad=False
        )
        scale_1 = self.cfg.reward_cfg.target_joint_pos_scale
        scale_2 = 2 * scale_1
        sin_pos_l[sin_pos_l > 0] = 0
        self.ref_dof_pos[:, 2] = sin_pos_l * scale_1  # left_hip_pitch_joint
        self.ref_dof_pos[:, 3] = sin_pos_l * scale_2  # left_knee_joint
        self.ref_dof_pos[:, 4] = sin_pos_l * scale_1  # left_ankle_joint
        sin_pos_r[sin_pos_r < 0] = 0
        self.ref_dof_pos[:, 7] = sin_pos_r * scale_1  # right_hip_pitch_joint
        self.ref_dof_pos[:, 8] = sin_pos_r * scale_2  # right_knee_joint
        self.ref_dof_pos[:, 9] = sin_pos_r * scale_1  # right_ankle_joint
        # Double support phase
        self.ref_dof_pos[torch.abs(sin_pos) < 0.1] = 0
        self.ref_dof_pos = 2 * self.ref_dof_pos

    def _parse_ref_pos(self, envstate):
        envstate.robots[self.robot.name].extra["ref_dof_pos"] = self.ref_dof_pos

    def _init_target_wp(self, env_states: EnvState) -> None:
        self.ori_feet_pos = self.rigid_state[
            :, self.feet_indices, :2
        ].clone()  # [num_envs, 2, 2], two feet's original xy positions
        self.target_wp, self.num_pairs, self.num_wp = self.sample_fp(
            self.device, num_points=1000000, num_wp=10, ranges=self.cfg.commands.ranges
        )  # relative, self.target_wp.shape=[num_pairs, num_wp, 2, 2]
        self.target_wp_i = torch.randint(
            0, self.num_pairs, (self.num_envs,), device=self.device
        )  # for each env, choose one seq, [num_envs]
        self.target_wp_j = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )  # for each env, the timestep in the seq is initialized to 0, [num_envs]
        self.target_wp_dt = 1 / self.cfg.human.freq
        self.target_wp_update_steps = self.target_wp_dt / self.dt  # not necessary integer
        assert self.dt <= self.target_wp_dt, (
            f"self.dt {self.dt} must be less than self.target_wp_dt {self.target_wp_dt}"
        )
        self.target_wp_update_steps_int = self.sample_int_from_float(self.target_wp_update_steps)

        self.ref_dof_pos = None
        self.ref_feet_pos = None
        self.ref_action = self.default_dof_pos
        self.delayed_obs_target_wp = None
        self.delayed_obs_target_wp_steps = self.cfg.human.delay / self.target_wp_dt
        self.delayed_obs_target_wp_steps_int = self.sample_int_from_float(self.delayed_obs_target_wp_steps)
        self.update_target_wp(torch.tensor([], dtype=torch.long, device=self.device))

    def _parse_state_for_reward(self, envstate):
        """
        Parse all the states to prepare for reward computation, legged_robot level reward computation.
        The

        Eg., offset the observation by default obs, compute input rewards.
        """
        # TODO read from config
        # parse those state which cannot directly get from Envstates
        super()._parse_state_for_reward(envstate)
        self._compute_ref_state()
        self._parse_ref_pos(envstate)

    def _compute_observations(self, envstates):
        """Add observation into states

        Input: envstates

        Output:

        """

        phase = self._get_phase()

        sin_pos = torch.sin(2 * torch.pi * phase).unsqueeze(1)
        cos_pos = torch.cos(2 * torch.pi * phase).unsqueeze(1)

        stance_mask = self._get_gait_phase()
        contact_mask = contact_forces_tensor(envstates, self.robot.name)[:, self.feet_indices, 2] > 5

        self.command_input = torch.cat((sin_pos, cos_pos, self.commands[:, :3] * self.commands_scale), dim=1)
        self.command_input_wo_clock = self.commands[:, :3] * self.commands_scale

        q = (
            dof_pos_tensor(envstates, self.robot.name) - self.cfg.default_joint_pd_target
        ) * self.cfg.normalization.obs_scales.dof_pos
        dq = dof_vel_tensor(envstates, self.robot.name) * self.cfg.normalization.obs_scales.dof_vel
        diff = dof_pos_tensor(envstates, self.robot.name) - ref_dof_pos_tenosr(envstates, self.robot.name)

        self.privileged_obs_buf = torch.cat(
            (
                self.command_input,  # 2 + 3
                q,  # |A|
                dq,  # |A|
                self.actions,  # |A|
                diff,  # |A|
                self.base_lin_vel * self.cfg.normalization.obs_scales.lin_vel,  # 3
                self.base_ang_vel * self.cfg.normalization.obs_scales.ang_vel,  # 3
                self.base_euler_xyz * self.cfg.normalization.obs_scales.quat,  # 3
                self.rand_push_force[:, :2],  # 3
                self.rand_push_torque,  # 3
                self.env_frictions,  # 1
                self.body_mass / 30.0,  # 1
                stance_mask,  # 2
                contact_mask,  # 2
            ),
            dim=-1,
        )

        obs_buf = torch.cat(
            (
                self.command_input_wo_clock,  # 3
                q,  # |A|
                dq,  # |A|
                self.actions,
                self.base_ang_vel * self.cfg.normalization.obs_scales.ang_vel,  # 3
                self.base_euler_xyz * self.cfg.normalization.obs_scales.quat,  # 3
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

    # TODO move to humanoid utils
    @staticmethod
    def sample_int_from_float(x):
        if int(x) == x:
            return int(x)
        return int(x) if np.random.rand() < (x - int(x)) else int(x) + 1

    # TODO move to humanoid utils
    def sample_fp(device, num_points, num_wp, ranges):
        """sample feet waypoints"""
        # left foot still, right foot move, [num_points//2, 2]
        l_positions_s = torch.zeros(num_points // 2, 2)  # left foot positions (xy)
        r_positions_m = torch.randn(num_points // 2, 2)
        r_positions_m = (
            r_positions_m / r_positions_m.norm(dim=-1, keepdim=True) * ranges.feet_max_radius
        )  # within a sphere, [-radius, +radius]
        # right foot still, left foot move, [num_points//2, 2]
        r_positions_s = torch.zeros(num_points // 2, 2)  # right foot positions (xy)
        l_positions_m = torch.randn(num_points // 2, 2)
        l_positions_m = (
            l_positions_m / l_positions_m.norm(dim=-1, keepdim=True) * ranges.feet_max_radius
        )  # within a sphere, [-radius, +radius]
        # concat
        l_positions = torch.cat([l_positions_s, l_positions_m], dim=0)  # (num_points, 2)
        r_positions = torch.cat([r_positions_m, r_positions_s], dim=0)  # (num_points, 2)
        wp = torch.stack([l_positions, r_positions], dim=1)  # (num_points, 2, 2)
        wp = wp.unsqueeze(1).repeat(1, num_wp, 1, 1)  # (num_points, num_wp, 2, 2)
        print("===> [sample_fp] return shape:", wp.shape)
        return wp.to(device), num_points, num_wp
