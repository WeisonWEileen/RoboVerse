import torch
import torch.nn as nn
from torchvision import models

from humanoid_visualrl.wrapper.walking_wrapper import WalkingWrapper
from metasim.scenario.scenario import ScenarioCfg
from metasim.types import TensorState
from humanoid_visualrl.utils.opencv_renderer import OpenCVRenderer


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
        self._init_resnet_encoder()

        # Initialize vision buffer for storing raw images
        self.camera_name = scenario.task.camera.name
        self.vision_buf = None
        self.resnet_features = None

    def _init_resnet_encoder(self):
        """Initialize ResNet-18 as a frozen feature extractor."""
        # Load pre-trained ResNet-18
        self.resnet = models.resnet18(pretrained=True)

        # Remove the final classification layer to get feature representations
        # ResNet-18 outputs 512-dimensional features before the final FC layer
        self.resnet = nn.Sequential(*list(self.resnet.children())[:-1])

        # Freeze all parameters
        for param in self.resnet.parameters():
            param.requires_grad = False

        # Set to evaluation mode
        self.resnet.eval()

        # Move to device
        self.resnet = self.resnet.to(self.device)

        # Feature dimension: ResNet-18 outputs 512-dim features
        self.resnet_feature_dim = 512

        # ResNet-18 encoder initialized with frozen weights

    def _extract_visual_features(self, rgb_images):
        """Extract features from RGB images using ResNet-18.

        Args:
            rgb_images: tensor of shape (batch_size, height, width, channels) or (batch_size, channels, height, width)

        Returns:
            features: tensor of shape (batch_size, 512)
        """
        if rgb_images is None:
            return torch.zeros(self.num_envs, self.resnet_feature_dim, device=self.device)

        # Ensure images are in the right format (batch_size, channels, height, width)
        if rgb_images.dim() == 4 and rgb_images.shape[-1] in [1, 3, 4]:  # HWC format
            rgb_images = rgb_images.permute(0, 3, 1, 2)  # Convert to CHW

        # Ensure we have 3 channels (RGB)
        if rgb_images.shape[1] == 4:  # RGBA
            rgb_images = rgb_images[:, :3, :, :]  # Take only RGB channels
        elif rgb_images.shape[1] == 1:  # Grayscale
            rgb_images = rgb_images.repeat(1, 3, 1, 1)  # Convert to RGB

        # Normalize to [0, 1] if needed
        if rgb_images.dtype == torch.uint8:
            rgb_images = rgb_images.float() / 255.0

        # ResNet expects images normalized with ImageNet stats
        # But for simplicity, we'll use the raw normalized images
        # You might want to apply ImageNet normalization for better features:
        # mean = torch.tensor([0.485, 0.456, 0.406]).to(self.device)
        # std = torch.tensor([0.229, 0.224, 0.225]).to(self.device)
        # rgb_images = (rgb_images - mean.view(1, 3, 1, 1)) / std.view(1, 3, 1, 1)

        with torch.no_grad():
            features = self.resnet(rgb_images)
            # Remove spatial dimensions (global average pooling is already applied)
            features = features.view(features.size(0), -1)  # (batch_size, 512)

        return features

    def _refreshed_tensors(self, tensor_state: TensorState):
        """Process tensor state and extract visual features."""
        super()._refreshed_tensors(tensor_state)


        camera_data = tensor_state.cameras[self.camera_name]
        self.vision_buf = camera_data.rgb
        self.resnet_features = self._extract_visual_features(camera_data.rgb)

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
                print("OpenCV display window closed by user")

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
