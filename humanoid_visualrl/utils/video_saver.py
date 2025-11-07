import numpy as np
import cv2
import imageio.v2 as iio

class VideoSaver:
    def __init__(self, video_path: str):
        self.video_path = video_path
        self.images: list[np.ndarray] = []
    
    def add(self, env_wrapper):
        rgb_frame = env_wrapper.env._get_offscreen_viewport_render() 
        egocentric_frame = env_wrapper.env.scene.sensors["camera_first_person"].data.output["rgb"][0]
        egocentric_frame = egocentric_frame.cpu().numpy()
        # change dimension from \
        # egocentric_frame = egocentric_frame.transpose(1, 2, 0)
        # resize egocentric frame to 374x374
        egocentric_frame = cv2.resize(egocentric_frame, (374, 374))
        # images.append(rgb_frame)
        # horizontal concat the egocentric frame and the rgb frame
        self.images.append(np.concatenate([egocentric_frame, rgb_frame], axis=1))

    def save(self):
        iio.mimsave(self.video_path,    self.images, fps=30)