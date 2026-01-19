from humanoid_visualrl.cfg.active_vision_cube_cfg import BaseTableHumanoidTaskCfg
import torch
from metasim.scenario.objects import RigidObjCfg
from metasim.constants import PhysicStateType
from metasim.utils import configclass
import math


# hierarchy: active_vision_insertion -> active_vision_cube -> active_vision_booster -> active_vision_articulated -> active_vision_junggle
@configclass(name="active_vision_insertion")
class ActiveVisionInsertionCfg(BaseTableHumanoidTaskCfg):
    """Base class for legged-gym style humanoid tasks.

    Attributes:
    robotname: name of the robot
    feet_indices: indices of the feet joints
    penalised_contact_indices: indices of the contact joints
    """

    def __post_init__(self):
        super().__post_init__()

        self.objects.append(
            RigidObjCfg(
                name="insertion_female_box",
                scale=(1, 1, 1),
                physics=PhysicStateType.RIGIDBODY,
                usd_path="roboverse_data/objects/female_box_bigger_flattened.usd",
                fix_base_link=False,
                default_position=(0.55, 0.1, 0.51),
                # default_orientation=(0.7071, 0.0, 0.0, 0.7071),
                default_orientation=(1.0, 0.0, 0.0, 0.0),
                collision_enabled=True,
                mass_density=100,
            )
        )

        self.init_states[0]["objects"]["insertion_female_box"] = {
            "pos": torch.tensor([0.55, 0.1, 0.51]),
            "rot": torch.tensor([0.7071, 0.0, 0.0, 0.7071]),
        }

        self.randomize_box = True
        self.randomize_box_xy_range_scale = 0.04
        self.randomize_box_rot_range_scale = math.pi / 6  # pi / 2
