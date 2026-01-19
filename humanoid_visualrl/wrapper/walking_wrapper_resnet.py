import torch

from humanoid_visualrl.utils.opencv_renderer import OpenCVRenderer
from humanoid_visualrl.wrapper.reset_18_extractor import Reset18Extractor
from humanoid_visualrl.wrapper.walking_wrapper import WalkingWrapper
from metasim.scenario.scenario import ScenarioCfg
from metasim.types import TensorState


class WalkingWrapperResNet(WalkingWrapper):
    """Walking wrapper with ResNet-18 visual feature extraction.

    This wrapper extends the base walking wrapper to include ResNet-18 as a frozen
    feature extractor for visual observations. The extracted features are concatenated
    to both privileged and regular observations.
    """

    def __init__(self, scenario: ScenarioCfg, enable_opencv_display: bool = False, opencv_fps: int = 30):
        super().__init__(scenario)

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
        # Initialize ResNet-18 as frozen encoder
        self.feature_extractor = Reset18Extractor(device=self.device)

        # Initialize vision buffer for storing raw images
        self.camera_name = scenario.task.camera.name
        self.vision_buf = None
        self.resnet_features = None

        # ResNet-18 encoder initialized with frozen weights

    def _refreshed_tensors(self, tensor_state: TensorState):
        """Process tensor state and extract visual features."""
        super()._refreshed_tensors(tensor_state)

        camera_data = tensor_state.cameras[self.camera_name]
        self.vision_buf = camera_data.rgb
        self.resnet_features = self.feature_extractor.extract_visual_features(camera_data.rgb)

        # Display image in OpenCV window if enabled
        if self.enable_opencv_display and self.opencv_renderer is not None:
            # Use the original uint8 RGB image for display (before normalization)
            # vision_rgb is in format (batch_size, height, width, channels)
            display_image = self.vision_buf[0]  # Take first environment

            # Display the image and check if window is still open
            window_open = self.opencv_renderer.display(display_image)
            if not window_open:
                # User closed the window, disable further display
                self.enable_opencv_display = False

    def close_opencv_display(self):
        """Close OpenCV display window and cleanup resources."""
        if self.opencv_renderer is not None:
            self.opencv_renderer.destroy_window()
            self.opencv_renderer = None
            self.enable_opencv_display = False

    def __del__(self):
        """Cleanup when wrapper is destroyed."""
        if hasattr(self, "opencv_renderer") and self.opencv_renderer is not None:
            self.close_opencv_display()

    def _compute_observations(self):
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

        # Extract ResNet features from vision if available
        if self.resnet_features is not None:
            visual_features = self.resnet_features
        else:
            visual_features = torch.zeros(self.num_envs, self.resnet_feature_dim, device=self.device)

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
                visual_features,  # 512 (ResNet-18 features)
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
                visual_features,  # 512 (ResNet-18 features)
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

        # update extra_buf
        self.extra_buf["observations"]["critic"] = self.privileged_obs_buf
