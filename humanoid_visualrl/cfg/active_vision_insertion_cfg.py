from humanoid_visualrl.cfg.active_vision_cube_cfg import BaseTableHumanoidTaskCfg
import torch
from metasim.scenario.objects import RigidObjCfg
from metasim.constants import PhysicStateType
from metasim.utils import configclass
import math
from metasim.utils.setup_util import get_robot
from metasim.scenario.robot import BaseActuatorCfg


# hierarchy: active_vision_insertion -> active_vision_cube -> active_vision_booster -> active_vision_articulated -> active_vision_junggle
@configclass(name="active_vision_insertion")
class ActiveVisionInsertionCfg(BaseTableHumanoidTaskCfg):
    """Base class for legged-gym style humanoid tasks.

    Attributes:
    robotname: name of the robot
    feet_indices: indices of the feet joints
    penalised_contact_indices: indices of the contact joints
    """

    scale = 1.0
    reward_weights_phase0 = {
        "pixel_norm_at_object": 0.8 * scale,
        "action_smoothness": -0.1 * scale,
        # "energy_consumption": -3e-7,
    }

    reward_weights_phase1 = {
        "pixel_norm_at_object": 0.4 * scale,
        "action_smoothness": -0.1 * scale,
        "energy_consumption": -3e-7,
        "finger_close_to_object": 1.3 * scale,
    }

    # cube ready in hand. reward for holding the cube not falling down
    reward_weights_phase2 = {
        "pixel_norm_at_object": 0.4 * scale,
        "action_smoothness": -0.1 * scale,
        "energy_consumption": -3e-7,
        "finger_close_to_object": 1.0 * scale,
        "contact_force": 0.8 * scale,
        "contact_force_upward": 0.8 * scale,
        # "lift_object": 100.0 * scale,
        "success": 100.0 * scale,
    }

    def __post_init__(self):
        super().__post_init__()
        self.objects[1].size = (0.045, 0.045, 0.045)
        self.objects.append(
            RigidObjCfg(
                name="insertion_female_box",
                scale=(1, 1, 1),
                physics=PhysicStateType.RIGIDBODY,
                usd_path="roboverse_data/objects/female_box_bigger_flattened_convex.usd",
                fix_base_link=True,
                default_position=(0.55, 0.1, 0.51),
                # default_orientation=(0.7071, 0.0, 0.0, 0.7071),
                default_orientation=(0.8660254, 0.0, 0.0, 0.5),
                collision_enabled=True,
                mass_density=10000,
            )
        )

        self.init_states[0]["objects"]["insertion_female_box"] = {
            "pos": torch.tensor([0.55, 0.1, 0.51]),
            "rot": torch.tensor([0.8660254, 0.0, 0.0, 0.5]),
        }

        # self.init_states[0]["objects"]["object"] = {
        #     "pos": torch.tensor([0.55, 0.1, 0.51]),
        #     "rot": torch.tensor([0.8660254, 0.0, 0.0, 0.51]),
        # }      
        self.init_states[0]["robots"]["vega"]["dof_pos"].update({
            "torso_j2": 0.3,
        })

        self.randomize_box = True
        self.randomize_box_xy_range_scale = 0.02
        self.randomize_box_rot_range_scale = math.pi / 20  # pi / 2

        self.robot = get_robot("vega")

        # add torso_j2 for actuators
        self.robot.actuators.update({
            "torso_j2": BaseActuatorCfg(velocity_limit=6.28, torque_limit=0.9, stiffness=1000, damping=200),
        })

        # Update num_actions to match the actual number of actuators
        self.num_actions = len(self.robot.actuators)

        self.robot.default_fixed_joints.discard("torso_j2")

        self.robot.modified_joint_limits.update({
            "torso_j2": (0.00, 0.60),
        })

        self.robot.action_scale.update({
            "torso_j2": 0.008 * self.robot.scale_factor,
        })

        self.robot.default_joint_positions.update({
            # "torso_j1": 0.15,
            "torso_j2": 0.3,
        })

        self.robot.control_type.update({
            "torso_j2": "position",
        })

        self.robot.torque_limits.update({
            "torso_j2": 1000.0,
        })

        self.robot.num_joints = len(self.robot.actuators) + len(self.robot.mimic_joints)

        self.robot.modified_joint_limits.update({"base_yaw_joint": (-math.pi / 6, math.pi / 6)})

        self.num_observations = self.robot.num_joints * 2 + self.num_actions
        self.num_privileged_obs = self.robot.num_joints * 2 + self.num_actions

        self.robot.action_scale["head_j2"] = 0.2 * self.robot.scale_factor

        if self.phase == 0:
            self.reward_weights = self.reward_weights_phase0
        elif self.phase == 1:
            self.reward_weights = self.reward_weights_phase1
        elif self.phase == 2:
            self.reward_weights = self.reward_weights_phase2
