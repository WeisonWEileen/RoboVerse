from omni.kit.viewport.utility import get_active_viewport, get_active_viewport_camera_path

# Retrieve the active viewport
viewport = get_active_viewport()

# Get the Sdf.Path to the camera used by the active viewport
camera_path = get_active_viewport_camera_path()
