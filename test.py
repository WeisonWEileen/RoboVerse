from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": True})
# load isaaclab 
import omni.kit.commands
from omni.kit.viewport.utility import get_active_viewport
from pxr import Sdf, Usd, UsdGeom

# Useful variables that will be passed to the FramePrimsCommand
camera_path = None
prim_to_frame = "/World/Cube"
time = Usd.TimeCode.Default()
resolution = (1, 1)
zoom = 0.6

# Get the stage
stage = omni.usd.get_context().get_stage()

active_viewport = get_active_viewport()
if active_viewport:
    # Pull meaningful information from the Viewport to frame a specific prim
    time = active_viewport.time
    resolution = active_viewport.resolution
    camera_path = active_viewport.camera_path
else:
    # Otherwise, create a camera that will be used to frame the prim_to_frame
    camera_path = "/World/New_Camera"
    UsdGeom.Camera.Define(stage, camera_path)

# Finally run the undo-able FramePrimsCommand
omni.kit.commands.execute(
    "FramePrimsCommand",
    # The path to the camera that is begin moved
    prim_to_move=camera_path,
    # The prim that is begin framed / looked at
    prims_to_frame=[prim_to_frame],
    # The Usd.TimCode that camera_path will use to set new location and orientation
    time_code=time,
    # The aspect_ratio of the image-place that is being viewed
    aspect_ratio=resolution[0] / resolution[1],
    # Additional slop to use for the framing
    zoom=zoom,
)
