"""OpenCV-based real-time image renderer for camera mounting visualization."""

import cv2
import numpy as np
import torch
from typing import Optional, Tuple


class OpenCVRenderer:
    """OpenCV-based renderer for real-time environment image visualization."""

    def __init__(
        self,
        window_name: str = "Environment Vision",
        window_size: Tuple[int, int] = (640, 480),
        fps_limit: int = 30,
        enable_recording: bool = False,
        recording_path: Optional[str] = None,
    ):
        """Initialize the OpenCV renderer.

        Args:
            window_name: Name of the OpenCV window
            window_size: Target size for the display window (width, height)
            fps_limit: Maximum FPS for display (to prevent overwhelming)
            enable_recording: Whether to enable video recording
            recording_path: Path to save recorded video (if recording enabled)
        """
        self.window_name = window_name
        self.window_size = window_size
        self.fps_limit = fps_limit
        self.enable_recording = enable_recording
        self.recording_path = recording_path

        # Internal state
        self.window_created = False
        self.last_display_time = 0.0
        self.frame_interval = 1.0 / fps_limit if fps_limit > 0 else 0.0

        # Video recording
        self.video_writer = None
        self.recording_active = False

        # Display statistics
        self.frame_count = 0
        self.display_fps = 0.0
        self.fps_update_interval = 30  # Update FPS display every N frames

    def create_window(self):
        """Create the OpenCV window."""
        if not self.window_created:
            cv2.namedWindow(self.window_name, cv2.WINDOW_AUTOSIZE)
            self.window_created = True
            print(f"OpenCV window '{self.window_name}' created")

    def destroy_window(self):
        """Destroy the OpenCV window and cleanup resources."""
        if self.window_created:
            cv2.destroyWindow(self.window_name)
            self.window_created = False

        if self.video_writer is not None:
            self.video_writer.release()
            self.video_writer = None
            self.recording_active = False
            print("Video recording stopped and saved")

    def start_recording(self, path: Optional[str] = None):
        """Start video recording."""
        if not self.enable_recording:
            print("Recording not enabled in renderer configuration")
            return

        recording_path = path or self.recording_path
        if recording_path is None:
            print("No recording path specified")
            return

        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.video_writer = cv2.VideoWriter(recording_path, fourcc, self.fps_limit, self.window_size)
        self.recording_active = True
        print(f"Started recording to: {recording_path}")

    def stop_recording(self):
        """Stop video recording."""
        if self.video_writer is not None:
            self.video_writer.release()
            self.video_writer = None
            self.recording_active = False
            print("Recording stopped")

    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Preprocess the image tensor for OpenCV display.

        Args:
            image: Input image tensor, expected to be uint8 with shape (H, W, C) or (B, H, W, C)

        Returns:
            Processed numpy array ready for OpenCV display
        """

        # # Handle batch dimension
        # if image.dim() == 4:
        #     # Take the first image in the batch
        #     image = image[0]
        # elif image.dim() == 3:
        #     pass  # Single image, no batch dimension
        # else:
        #     raise ValueError(f"Unexpected image dimensions: {image.shape}")

        # # Convert to numpy
        # if isinstance(image, torch.Tensor):
        #     # Move to CPU if on GPU
        #     if image.is_cuda:
        #         image = image.cpu()
        #     image_np = image.numpy()
        # else:
        #     image_np = np.array(image)

        # Ensure uint8 format
        image_np = image
        if image_np.dtype != np.uint8:
            # If the image is normalized [0, 1], scale to [0, 255]
            if image_np.max() <= 1.0:
                image_np = (image_np * 255).astype(np.uint8)
            else:
                image_np = image_np.astype(np.uint8)

        # Handle different channel orders and formats
        if image_np.shape[-1] == 3:
            # RGB to BGR for OpenCV
            image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        elif image_np.shape[-1] == 4:
            # RGBA to BGR
            image_np = cv2.cvtColor(image_np, cv2.COLOR_RGBA2BGR)
        elif len(image_np.shape) == 2:
            # Grayscale to BGR
            image_np = cv2.cvtColor(image_np, cv2.COLOR_GRAY2BGR)
        else:
            raise ValueError(f"Unsupported image format: {image_np.shape}")

        return image_np

    def _resize_image(self, image_np: np.ndarray) -> np.ndarray:
        """Resize image to target window size."""
        current_height, current_width = image_np.shape[:2]
        target_width, target_height = self.window_size

        # Only resize if necessary
        if current_width != target_width or current_height != target_height:
            image_np = cv2.resize(image_np, (target_width, target_height), interpolation=cv2.INTER_LINEAR)

        return image_np

    def _add_info_overlay(self, image_np: np.ndarray) -> np.ndarray:
        """Add information overlay to the image."""
        # Add FPS counter
        # fps_text = f"FPS: {self.display_fps:.1f}"
        # cv2.putText(image_np, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Add frame counter
        if self.frame_count % 100 == 0:
            frame_text = f"Frame: {self.frame_count}"
            cv2.putText(image_np, frame_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Add recording indicator
        if self.recording_active:
            cv2.putText(image_np, "REC", (image_np.shape[1] - 60, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            # Add recording dot
            cv2.circle(image_np, (image_np.shape[1] - 80, 25), 5, (0, 0, 255), -1)

        return image_np

    def display(self, image: np.ndarray, force_update: bool = False) -> bool:
        """Display an image in the OpenCV window.

        Args:
            image: Input image tensor (uint8, shape: H×W×C or B×H×W×C)
            force_update: Force display update regardless of FPS limit

        Returns:
            True if window is still open, False if user closed it
        """
        import time

        # Check FPS limiting
        current_time = time.time()
        if not force_update and self.frame_interval > 0:
            if current_time - self.last_display_time < self.frame_interval:
                return True  # Skip this frame

        # Create window if needed
        if not self.window_created:
            self.create_window()

        # Preprocess the image
        # try:
        image_np = self._preprocess_image(image)
        image_np = self._resize_image(image_np)
        image_np = self._add_info_overlay(image_np)
        # except Exception as e:
        #     print(f"Error preprocessing image: {e}")
        #     return True

        # Display the image
        cv2.imshow(self.window_name, image_np)

        # Record if enabled
        if self.recording_active and self.video_writer is not None:
            self.video_writer.write(image_np)

        # Update timing and statistics
        self.last_display_time = current_time
        self.frame_count += 1

        # Update FPS calculation
        # if self.frame_count % self.fps_update_interval == 0:
        # if hasattr(self, "_fps_start_time"):
        # elapsed = current_time - self._fps_start_time
        # self.display_fps = self.fps_update_interval / elapsed
        # self._fps_start_time = current_time

        # Handle window events and check if window is still open
        key = cv2.waitKey(1) & 0xFF

        # Check for key presses
        if key == ord("q") or key == 27:  # 'q' or ESC to quit
            self.destroy_window()
            return False
        elif key == ord("r") and self.enable_recording:  # 'r' to toggle recording
            if self.recording_active:
                self.stop_recording()
            else:
                self.start_recording()
        elif key == ord("s"):  # 's' to save screenshot
            screenshot_path = f"screenshot_{int(current_time)}.png"
            cv2.imwrite(screenshot_path, image_np)
            print(f"Screenshot saved: {screenshot_path}")

        # Check if window was closed by user clicking X
        try:
            # This is a workaround to detect if window is closed
            cv2.getWindowProperty(self.window_name, cv2.WND_PROP_VISIBLE)
        except cv2.error:
            self.window_created = False
            return False

        return True

    def __del__(self):
        """Cleanup when object is destroyed."""
        self.destroy_window()


class MultiCameraRenderer:
    """Renderer for displaying multiple camera feeds simultaneously."""

    def __init__(self, camera_names: list[str], **renderer_kwargs):
        """Initialize multi-camera renderer.

        Args:
            camera_names: List of camera names to display
            **renderer_kwargs: Arguments passed to individual OpenCVRenderer instances
        """
        self.camera_names = camera_names
        self.renderers = {}

        for camera_name in camera_names:
            window_name = f"Camera: {camera_name}"
            self.renderers[camera_name] = OpenCVRenderer(window_name=window_name, **renderer_kwargs)

    def display(self, camera_images: dict[str, torch.Tensor], force_update: bool = False) -> bool:
        """Display images from multiple cameras.

        Args:
            camera_images: Dictionary mapping camera names to image tensors
            force_update: Force display update regardless of FPS limit

        Returns:
            True if all windows are still open, False if any was closed
        """
        all_open = True

        for camera_name, renderer in self.renderers.items():
            if camera_name in camera_images:
                is_open = renderer.display(camera_images[camera_name], force_update)
                all_open = all_open and is_open

        return all_open

    def destroy_all(self):
        """Destroy all windows."""
        for renderer in self.renderers.values():
            renderer.destroy_window()
