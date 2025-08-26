import torch

from humanoid_visualrl.cfg.humanoidVisualRLCfg import BaseTableHumanoidTaskCfg
from humanoid_visualrl.wrapper.base_humanoid_wrapper import HumanoidBaseWrapper
from humanoid_visualrl.utils.opencv_renderer import OpenCVRenderer
from metasim.scenario.scenario import ScenarioCfg
from metasim.types import TensorState


class WalkingWrapperCNN(HumanoidBaseWrapper):
    """Wrapper for walking tasks."""

    def __init__(self, scenario: ScenarioCfg, enable_opencv_display: bool = False, opencv_fps: int = 30):
        super().__init__(scenario)
        self._prepare_ref_indices()

        # Initialize OpenCV renderer for real-time visualization
        self.enable_opencv_display = enable_opencv_display
        self.opencv_renderer = None
        if self.enable_opencv_display:
            self.opencv_renderer = OpenCVRenderer(
                window_name="Humanoid First Person View",
                window_size=(640, 480),  # Upscale from 64x48 to 640x480
                fps_limit=opencv_fps,
                enable_recording=True,  # Allow video recording
                recording_path="humanoid_vision_recording.mp4",
            )

    def _prepare_ref_indices(self):
        joint_names = self.env.get_joint_names(self.robot.name)
        self.left_hip_pitch_joint_idx = joint_names.index("left_hip_pitch_joint")
        self.left_knee_joint_idx = joint_names.index("left_knee_joint")
        self.right_hip_pitch_joint_idx = joint_names.index("right_hip_pitch_joint")
        self.right_knee_joint_idx = joint_names.index("right_knee_joint")

        self.left_ankle_joint_idx = joint_names.index("left_ankle_pitch_joint")
        self.right_ankle_joint_idx = joint_names.index("right_ankle_pitch_joint")

    def _pre_compute_reward(self):
        self._compute_ref_state()

    def _init_buffers(self):
        super()._init_buffers()
        self.obs_buf_state = torch.zeros(self.num_envs, self.cfg.num_obs_state, device=self.device)
        # self.vision_buf = torch.zeros(self.num_envs, 3, 48, 64, device=self.device)
        self.obs_buf = (self.obs_buf_state, self.vision_buf)

    def _refreshed_tensors(self, tensor_state: TensorState):
        super()._refreshed_tensors(tensor_state)
        # Convert from HWC (H, W, C) to CHW (C, H, W) format for PyTorch CNN
        # Convert from uint8 to float and normalize to [0, 1]
        vision_rgb = tensor_state.cameras["camera_first_person"].rgb
        self.vision_buf = vision_rgb.permute(0, 3, 1, 2).float() / 255.0

        # Display image in OpenCV window if enabled
        if self.enable_opencv_display and self.opencv_renderer is not None:
            # Use the original uint8 RGB image for display (before normalization)
            # vision_rgb is in format (batch_size, height, width, channels)
            display_image = vision_rgb[0]  # Take first environment

            # Display the image and check if window is still open
            window_open = self.opencv_renderer.display(display_image)
            if not window_open:
                # User closed the window, disable further display
                self.enable_opencv_display = False
                print("OpenCV display window closed by user")

    def _compute_ref_state(self):
        phase = self._get_phase()
        sin_pos = torch.sin(2 * torch.pi * phase)
        sin_pos_l = sin_pos.clone()
        sin_pos_r = sin_pos.clone()
        self.ref_dof_pos = torch.zeros(self.num_envs, self.num_actions, device=self.device, requires_grad=False)
        scale_1 = self.cfg.reward_cfg.target_joint_pos_scale
        scale_2 = 2 * scale_1
        sin_pos_l[sin_pos_l > 0] = 0
        self.ref_dof_pos[:, self.left_hip_pitch_joint_idx] = sin_pos_l * scale_1  # left_hip_pitch_joint
        self.ref_dof_pos[:, self.left_knee_joint_idx] = sin_pos_l * scale_2  # left_knee_joint
        self.ref_dof_pos[:, self.left_ankle_joint_idx] = sin_pos_l * scale_1  # left_ankle_joint
        sin_pos_r[sin_pos_r < 0] = 0
        self.ref_dof_pos[:, self.right_hip_pitch_joint_idx] = sin_pos_r * scale_1  # right_hip_pitch_joint
        self.ref_dof_pos[:, self.right_knee_joint_idx] = sin_pos_r * scale_2  # right_knee_joint
        self.ref_dof_pos[:, self.right_ankle_joint_idx] = sin_pos_r * scale_1  # right_ankle_joint
        # Double support phase
        self.ref_dof_pos[torch.abs(sin_pos) < 0.1] = 0
        self.ref_dof_pos = 2 * self.ref_dof_pos

    def step(self, actions):
        action = self._pre_physics_step(actions)
        self._physics_step(action)
        self._post_physics_step()
        return self.obs_buf, self.rew_buf, self.reset_buf, self.extra_buf

    def _compute_observations(self, tensor_state: TensorState):
        phase = self._get_phase()

        sin_pos = torch.sin(2 * torch.pi * phase).unsqueeze(1)
        cos_pos = torch.cos(2 * torch.pi * phase).unsqueeze(1)

        stance_mask = self._get_gait_phase()
        contact_mask = self.contact_forces[:, self.feet_indices, 2] > 5

        self.command_input = torch.cat((sin_pos, cos_pos, self.commands[:, :3] * self.commands_scale), dim=1)
        self.command_input_wo_clock = self.commands[:, :3] * self.commands_scale

        q = (self.dof_pos - self.default_joint_pd_target) * self.cfg.normalization.obs_scales.dof_pos
        dq = self.dof_vel * self.cfg.normalization.obs_scales.dof_vel
        diff = self.dof_pos - self.ref_dof_pos

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
                self.rand_push_force[:, :2],  # 2
                self.rand_push_torque,  # 3
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
        self.obs_buf_state = obs_buf_all.reshape(self.num_envs, -1)
        self.privileged_obs_buf = torch.cat([self.critic_history[i] for i in range(self.cfg.c_frame_stack)], dim=1)

        self.privileged_obs_buf = torch.clip(
            self.privileged_obs_buf, -self.cfg.normalization.clip_observations, self.cfg.normalization.clip_observations
        )

        # update extra_buf and attach vision to extra_buf
        self.obs_buf = (self.obs_buf_state, self.vision_buf)
        # self.extra_buf["observations"]["critic"] = (self.privileged_obs_buf, self.vision_buf)

    # ================================ Reward Functions ================================

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
        upper_body_diff = self.dof_pos - self.default_joint_pd_target
        upper_body_error = torch.mean(torch.abs(upper_body_diff), dim=1)
        return torch.exp(-4 * upper_body_error), upper_body_error

    def _reward_default_joint_pos(
        self, states: TensorState, robot_name: str, cfg: BaseTableHumanoidTaskCfg
    ) -> torch.Tensor:
        """Keep joint positions close to defaults (penalize yaw/roll)."""
        joint_diff = self.dof_pos - self.default_joint_pd_target
        return -0.01 * torch.norm(joint_diff, dim=1)

    def _reward_torques(self, states: TensorState, robot_name: str, cfg: BaseTableHumanoidTaskCfg) -> torch.Tensor:
        """Penalize high torques."""
        return torch.sum(torch.square(self.effort), dim=1)

    def _reward_dof_vel(self, states: TensorState, robot_name: str, cfg: BaseTableHumanoidTaskCfg) -> torch.Tensor:
        """Penalize high dof velocities."""
        return torch.sum(torch.square(states.robots[robot_name].joint_vel), dim=1)

    def _reward_dof_acc(self, states: TensorState, robot_name: str, cfg: BaseTableHumanoidTaskCfg) -> torch.Tensor:
        """Penalize high DOF accelerations."""
        return torch.sum(
            torch.square((self.last_dof_vel - self.dof_vel) / self.dt),
            dim=1,
        )

    def _reward_feet_air_time(
        self, states: TensorState, robot_name: str, cfg: BaseTableHumanoidTaskCfg
    ) -> torch.Tensor:
        """Calculates the reward for feet air time, promoting longer steps."""
        contact = self.contact_forces[:, self.feet_indices, 2] > 5.0
        stance_mask = self._get_gait_phase()
        self.contact_filt = torch.logical_or(torch.logical_or(contact, stance_mask), self.last_contacts)
        self.last_contacts = contact
        first_contact = (self.feet_air_time > 0.0) * self.contact_filt
        self.feet_air_time += self.dt
        air_time = self.feet_air_time.clamp(0, 0.5) * first_contact
        self.feet_air_time *= ~self.contact_filt
        return air_time.sum(dim=1)

    def _reward_action_rate(self, states: TensorState, robot_name: str, cfg: BaseTableHumanoidTaskCfg) -> torch.Tensor:
        """Penalize high action rate."""
        return torch.sum(
            torch.square(self.last_actions - self.actions),
            dim=1,
        )

    def _reward_action_smoothness(
        self, states: TensorState, robot_name: str, cfg: BaseTableHumanoidTaskCfg
    ) -> torch.Tensor:
        """Penalize jerk in actions."""
        term_1 = torch.sum(
            torch.square(self.last_actions - self.actions),
            dim=1,
        )
        term_2 = torch.sum(
            torch.square(self.actions + self.last_last_actions - 2 * self.last_actions),
            dim=1,
        )
        term_3 = 0.05 * torch.sum(torch.abs(self.actions), dim=1)
        return term_1 + term_2 + term_3

    def close_opencv_display(self):
        """Close OpenCV display window and cleanup resources."""
        if self.opencv_renderer is not None:
            self.opencv_renderer.destroy_window()
            self.opencv_renderer = None
            self.enable_opencv_display = False
            print("OpenCV display closed and resources cleaned up")

    def __del__(self):
        """Cleanup when wrapper is destroyed."""
        if hasattr(self, "opencv_renderer") and self.opencv_renderer is not None:
            self.close_opencv_display()

    def _reward_ang_vel_xy(self, states: TensorState, robot_name: str, cfg: BaseTableHumanoidTaskCfg) -> torch.Tensor:
        """Reward for xy angular velocity."""
        return torch.sum(torch.square(self.base_ang_vel[:, :2]), dim=1)

    def _reward_base_acc(self, states: TensorState, robot_name: str, cfg: BaseTableHumanoidTaskCfg) -> torch.Tensor:
        """Penalize base acceleration."""
        root_acc = self.last_root_vel - states.robots[robot_name].root_state[:, 7:13]
        rew = torch.exp(-torch.norm(root_acc, dim=1) * 3)
        return rew

    # def _reward_base_height(self, states: TensorState, robot_name: str, cfg: BaseTableHumanoidTaskCfg) -> torch.Tensor:
    #     """Penalize base height deviation from target."""
    #     stance_mask = self._get_gait_phase()
    #     measured_heights = torch.sum(
    #         states.robots[robot_name].body_state[:, self.feet_indices, 2] * stance_mask,
    #         dim=1,
    #     ) / torch.sum(stance_mask, dim=1)
    #     base_height = states.robots[robot_name].root_state[:, 2] - (measured_heights - 0.05)
    #     return torch.exp(-torch.abs(base_height - cfg.reward_cfg.base_height_target) * 100)

    def _reward_base_height(self, states: TensorState, robot_name: str, cfg: BaseTableHumanoidTaskCfg) -> torch.Tensor:
        # Penalize base height away from target
        base_height = states.robots[robot_name].root_state[:, 2]
        return torch.square(base_height - self.cfg.reward_cfg.base_height_target)

    def _reward_collision(self, states: TensorState, robot_name: str, cfg: BaseTableHumanoidTaskCfg) -> torch.Tensor:
        """Penalize collisions."""
        return torch.sum(
            1.0
            * (
                torch.norm(
                    self.contact_forces[:, self.penalised_contact_indices, :],
                    dim=-1,
                )
                > 0.1
            ),
            dim=1,
        )

    def _reward_dof_pos_limits(
        self, states: TensorState, robot_name: str, cfg: BaseTableHumanoidTaskCfg
    ) -> torch.Tensor:
        """Penalize DOF positions that are out of limits."""
        out_of_limits = -(states.robots[robot_name].joint_pos - cfg.dof_pos_limits[:, 0]).clip(max=0.0)
        out_of_limits += (states.robots[robot_name].joint_pos - cfg.dof_pos_limits[:, 1]).clip(min=0.0)
        return torch.sum(out_of_limits, dim=1)

    def _reward_dof_vel_limits(
        self, states: TensorState, robot_name: str, cfg: BaseTableHumanoidTaskCfg
    ) -> torch.Tensor:
        """Penalize high DOF velocities."""
        return torch.sum(
            (
                torch.abs(states.robots[robot_name].dof_vel) - self.dof_vel_limits * cfg.reward_cfg.soft_dof_vel_limit
            ).clip(min=0.0, max=1.0),
            dim=1,
        )

    def _reward_elbow_distance(
        self, states: TensorState, robot_name: str, cfg: BaseTableHumanoidTaskCfg
    ) -> torch.Tensor:
        """Calculates the reward based on the distance between the elbow of the humanoid."""
        elbow_pos = states.robots[robot_name].body_state[:, self.elbow_indices, :2]
        elbow_dist = torch.norm(elbow_pos[:, 0, :] - elbow_pos[:, 1, :], dim=1)
        fd = cfg.reward_cfg.min_dist
        max_df = cfg.reward_cfg.max_dist / 2
        d_min = torch.clamp(elbow_dist - fd, -0.5, 0.0)
        d_max = torch.clamp(elbow_dist - max_df, 0, 0.5)
        return (torch.exp(-torch.abs(d_min) * 100) + torch.exp(-torch.abs(d_max) * 100)) / 2, elbow_dist

    def _reward_feet_clearance(
        self, states: TensorState, robot_name: str, cfg: BaseTableHumanoidTaskCfg
    ) -> torch.Tensor:
        """Reward swing leg clearance."""
        contact = self.contact_forces[:, self.feet_indices, 2] > 5.0
        feet_z = states.robots[robot_name].body_state[:, self.feet_indices, 2] - 0.05
        delta_z = feet_z - self.last_feet_z
        self.feet_height += delta_z
        self.last_feet_z = feet_z

        # Compute swing mask
        swing_mask = 1 - self._get_gait_phase()

        # feet height should be closed to target feet height at the peak
        rew_pos = torch.abs(self.feet_height - self.cfg.reward_cfg.target_feet_height) < 0.01
        rew_pos = torch.sum(rew_pos * swing_mask, dim=1)
        self.feet_height *= ~contact
        return rew_pos

    def _reward_feet_contact_forces(
        self, states: TensorState, robot_name: str, cfg: BaseTableHumanoidTaskCfg
    ) -> torch.Tensor:
        """Penalize high contact forces on feet."""
        return torch.sum(
            (
                torch.norm(
                    self.contact_forces[:, self.feet_indices, :],
                    dim=-1,
                )
                - cfg.reward_cfg.max_contact_force
            ).clip(0, 400),
            dim=1,
        )

    def _reward_feet_contact_number(
        self, states: TensorState, robot_name: str, cfg: BaseTableHumanoidTaskCfg
    ) -> torch.Tensor:
        """Reward based on feet contact matching gait phase."""
        contact = self.contact_forces[:, self.feet_indices, 2] > 5.0
        stance_mask = self._get_gait_phase()
        reward = torch.where(contact == stance_mask, 1.0, -0.3)
        return torch.mean(reward, dim=1)

    def _reward_feet_distance(
        self, states: TensorState, robot_name: str, cfg: BaseTableHumanoidTaskCfg
    ) -> torch.Tensor:
        """Calculates the reward based on the distance between the feet. Penalize feet get close to each other or too far away."""
        foot_pos = states.robots[robot_name].body_state[:, self.feet_indices, :2]
        foot_dist = torch.norm(foot_pos[:, 0, :] - foot_pos[:, 1, :], dim=1)
        fd = cfg.reward_cfg.min_dist
        max_df = cfg.reward_cfg.max_dist
        d_min = torch.clamp(foot_dist - fd, -0.5, 0.0)
        d_max = torch.clamp(foot_dist - max_df, 0, 0.5)
        return (torch.exp(-torch.abs(d_min) * 100) + torch.exp(-torch.abs(d_max) * 100)) / 2, foot_dist

    def _reward_foot_slip(self, states: TensorState, robot_name: str, cfg: BaseTableHumanoidTaskCfg) -> torch.Tensor:
        """Calculates the reward for minimizing foot slip."""
        contact = self.contact_forces[:, self.feet_indices, 2] > 5.0
        foot_speed_norm = torch.norm(states.robots[robot_name].body_state[:, self.feet_indices, 10:12], dim=2)
        rew = torch.sqrt(foot_speed_norm)
        rew *= contact
        return torch.sum(rew, dim=1)

    def _reward_joint_pos(self, states: TensorState, robot_name: str, cfg: BaseTableHumanoidTaskCfg) -> torch.Tensor:
        """Calculates the reward based on the difference between the current joint positions and the target joint positions."""
        joint_pos = states.robots[robot_name].joint_pos.clone()
        pos_target = self.ref_dof_pos.clone()
        diff = joint_pos - pos_target
        r = torch.exp(-2 * torch.norm(diff, dim=1)) - 0.2 * torch.norm(diff, dim=1).clamp(0, 0.5)
        return r, torch.mean(torch.abs(diff), dim=1)

    def _reward_knee_distance(
        self, states: TensorState, robot_name: str, cfg: BaseTableHumanoidTaskCfg
    ) -> torch.Tensor:
        """Calculates the reward based on the distance between the knee of the humanoid."""
        knee_pos = states.robots[robot_name].body_state[:, self.knee_indices, :2]
        knee_dist = torch.norm(knee_pos[:, 0, :] - knee_pos[:, 1, :], dim=1)
        fd = cfg.reward_cfg.min_dist
        max_df = cfg.reward_cfg.max_dist / 2
        d_min = torch.clamp(knee_dist - fd, -0.5, 0.0)
        d_max = torch.clamp(knee_dist - max_df, 0, 0.5)
        return (torch.exp(-torch.abs(d_min) * 100) + torch.exp(-torch.abs(d_max) * 100)) / 2, knee_dist

    def _reward_lin_vel_z(self, states: TensorState, robot_name: str, cfg: BaseTableHumanoidTaskCfg) -> torch.Tensor:
        """Reward for z linear velocity."""
        return torch.square(self.base_lin_vel[:, 2])

    def _reward_low_speed(self, states: TensorState, robot_name: str, cfg: BaseTableHumanoidTaskCfg) -> torch.Tensor:
        """Penalize speed mismatch with command."""
        absolute_speed = torch.abs(self.base_lin_vel[:, 0])
        absolute_command = torch.abs(self.commands[:, 0])
        speed_too_low = absolute_speed < 0.5 * absolute_command
        speed_too_high = absolute_speed > 1.2 * absolute_command
        speed_desired = ~(speed_too_low | speed_too_high)
        sign_mismatch = torch.sign(self.base_lin_vel[:, 0]) != torch.sign(self.commands[:, 0])
        reward = torch.zeros_like(self.base_lin_vel[:, 0])
        reward[speed_too_low] = -1.0
        reward[speed_too_high] = 0.0
        reward[speed_desired] = 1.2
        reward[sign_mismatch] = -2.0
        return reward * (self.commands[:, 0].abs() > 0.1)

    def _reward_orientation(self, states: TensorState, robot_name: str, cfg: BaseTableHumanoidTaskCfg) -> torch.Tensor:
        """Penalize deviation from flat base orientation."""
        quat_mismatch = torch.exp(-torch.sum(torch.abs(self.base_euler_xyz[:, :2]), dim=1) * 10)
        orientation = torch.exp(-torch.norm(self.projected_gravity[:, :2], dim=1) * 20)
        return (quat_mismatch + orientation) / 2.0

    def _reward_stand_still(self, states: TensorState, robot_name: str, cfg: BaseTableHumanoidTaskCfg) -> torch.Tensor:
        """Reward for standing still, penalizing deviation from default joint positions."""
        return torch.sum(torch.abs(states.robots[robot_name].joint_pos - cfg.default_dof_pos), dim=1) * (
            torch.norm(self.commands[:, :2], dim=1) < 0.1
        )

    def _reward_stumble(self, states: TensorState, robot_name: str, cfg: BaseTableHumanoidTaskCfg) -> torch.Tensor:
        """Penalize stumbling based on contact forces."""
        return torch.any(
            torch.norm(self.contact_forces[:, self.feet_indices, :2], dim=2)
            > 5 * torch.abs(self.contact_forces[:, self.feet_indices, 2]),
            dim=1,
        )

    def _reward_torque_limits(
        self, states: TensorState, robot_name: str, cfg: BaseTableHumanoidTaskCfg
    ) -> torch.Tensor:
        """Penalize high torques."""
        return torch.sum(
            (
                torch.abs(states.robots[robot_name].joint_effort_target)
                - self.torque_limits * cfg.reward_cfg.soft_torque_limit
            ).clip(min=0.0),
            dim=1,
        )

    def _reward_tracking_ang_vel(
        self, states: TensorState, robot_name: str, cfg: BaseTableHumanoidTaskCfg
    ) -> torch.Tensor:
        """Track angular velocity commands (yaw)."""
        ang_vel_diff = self.commands[:, 2] - self.base_ang_vel[:, 2]
        ang_vel_error = torch.square(ang_vel_diff)
        return torch.exp(-ang_vel_error * cfg.reward_cfg.tracking_sigma), torch.abs(ang_vel_diff)

    def _reward_tracking_lin_vel(
        self, states: TensorState, robot_name: str, cfg: BaseTableHumanoidTaskCfg
    ) -> torch.Tensor:
        """Track linear velocity commands (xy axes)."""
        lin_vel_diff = self.commands[:, :2] - self.base_lin_vel[:, :2]
        lin_vel_error = torch.sum(torch.square(lin_vel_diff), dim=1)
        return torch.exp(-lin_vel_error * cfg.reward_cfg.tracking_sigma), torch.mean(torch.abs(lin_vel_diff), dim=1)

    def _reward_track_vel_hard(
        self, states: TensorState, robot_name: str, cfg: BaseTableHumanoidTaskCfg
    ) -> torch.Tensor:
        """Track linear and angular velocity commands."""
        lin_vel_error = torch.norm(
            self.commands[:, :2] - self.base_lin_vel[:, :2],
            dim=1,
        )
        lin_vel_error_exp = torch.exp(-lin_vel_error * 10)
        ang_vel_error = torch.abs(self.commands[:, 2] - self.base_ang_vel[:, 2])
        ang_vel_error_exp = torch.exp(-ang_vel_error * 10)
        linear_error = 0.2 * (lin_vel_error + ang_vel_error)
        return (lin_vel_error_exp + ang_vel_error_exp) / 2.0 - linear_error

    def _reward_vel_mismatch_exp(
        self, states: TensorState, robot_name: str, cfg: BaseTableHumanoidTaskCfg
    ) -> torch.Tensor:
        """Penalize velocity mismatch."""
        lin_mismatch = torch.exp(-torch.square(self.base_lin_vel[:, 2]) * 10)
        ang_mismatch = torch.exp(-torch.norm(self.base_ang_vel[:, :2], dim=1) * 5.0)
        return (lin_mismatch + ang_mismatch) / 2.0
