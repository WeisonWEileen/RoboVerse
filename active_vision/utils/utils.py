import numpy as np
import torch
from loguru import logger as log

from metasim.sim.base import BaseSimHandler


def get_body_reindexed_indices_from_substring(
    sim_handler: BaseSimHandler, obj_name: str, body_names: list[str], device
):
    """Given substrings of body name, find all the bodies indices in sorted order."""
    matches = []
    sorted_names = sim_handler.get_body_names(obj_name, sort=True)

    for name in body_names:
        for i, s in enumerate(sorted_names):
            if name in s:
                matches.append(i)

    index = torch.tensor(matches, dtype=torch.int32, device=device)
    return index


def get_joint_reindexed_indices_from_substring(
    sim_handler: BaseSimHandler, obj_name: str, joint_names: list[str], device: str
):
    """Given substrings of joint name, find all the bodies indices in sorted order."""
    matches = []
    sorted_names = sim_handler.get_joint_names(obj_name, sort=True)

    for name in joint_names:
        for i, s in enumerate(sorted_names):
            if name in s:
                matches.append(i)

    index = torch.tensor(matches, dtype=torch.int32, device=device)
    return index


def torch_rand_float(lower: float, upper: float, shape: tuple[int, int], device: str) -> torch.Tensor:
    """Generate a tensor of random floats in the range [lower, upper]."""
    return (upper - lower) * torch.rand(*shape, device=device) + lower


@torch.jit.script
def copysign(mag: float, other: torch.Tensor) -> torch.Tensor:
    """Create a new floating-point tensor with the magnitude of input and the sign of other, element-wise.

    Note:
        The implementation follows from `torch.copysign`. The function allows a scalar magnitude.

    Args:
        mag: The magnitude scalar.
        other: The tensor containing values whose signbits are applied to magnitude.

    Returns:
        The output tensor.
    """
    mag_torch = torch.tensor(mag, device=other.device, dtype=torch.float).repeat(other.shape[0])
    return torch.abs(mag_torch) * torch.sign(other)


@torch.jit.script
def euler_xyz_from_quat(quat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convert rotations given as quaternions to Euler angles in radians.

    Note:
        The euler angles are assumed in XYZ convention.

    Args:
        quat: The quaternion orientation in (w, x, y, z). Shape is (N, 4).

    Returns:
        A tuple containing roll-pitch-yaw. Each element is a tensor of shape (N,).

    Reference:
        https://en.wikipedia.org/wiki/Conversion_between_quaternions_and_Euler_angles
    """
    q_w, q_x, q_y, q_z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    # roll (x-axis rotation)
    sin_roll = 2.0 * (q_w * q_x + q_y * q_z)
    cos_roll = 1 - 2 * (q_x * q_x + q_y * q_y)
    roll = torch.atan2(sin_roll, cos_roll)

    # pitch (y-axis rotation)
    sin_pitch = 2.0 * (q_w * q_y - q_z * q_x)
    pitch = torch.where(torch.abs(sin_pitch) >= 1, copysign(torch.pi / 2.0, sin_pitch), torch.asin(sin_pitch))

    # yaw (z-axis rotation)
    sin_yaw = 2.0 * (q_w * q_z + q_x * q_y)
    cos_yaw = 1 - 2 * (q_y * q_y + q_z * q_z)
    yaw = torch.atan2(sin_yaw, cos_yaw)

    return roll % (2 * torch.pi), pitch % (2 * torch.pi), yaw % (2 * torch.pi)  # TODO: why not wrap_to_pi here ?


def get_euler_xyz_tensor(quat):
    """Convert quaternion to Euler angles (roll, pitch, yaw) in radians for a batch of quaternions.

    Args:
        quat (torch.Tensor): Quaternion tensor of shape (N, 4) where N is the batch size.

    Returns:
        torch.Tensor: Euler angles tensor of shape (N, 3) where each row contains (roll, pitch, yaw).
    """
    r, p, w = euler_xyz_from_quat(quat)
    # stack r, p, w in dim1
    euler_xyz = torch.stack((r, p, w), dim=1)
    euler_xyz[euler_xyz > torch.pi] -= 2 * torch.pi
    return euler_xyz


def sample_int_from_float(x):
    """Samples an int from a float."""
    if int(x) == x:
        return int(x)
    return int(x) if np.random.rand() < (x - int(x)) else int(x) + 1


def sample_wp(device, num_points, num_wp, ranges):
    """Sample waypoints, relative to the starting point."""
    # position
    l_positions = torch.randn(num_points, 3)  # left wrist positions
    l_positions = (
        l_positions / l_positions.norm(dim=-1, keepdim=True) * ranges.wrist_max_radius
    )  # within a sphere, [-radius, +radius]
    l_positions = l_positions[
        l_positions[:, 0] > ranges.l_wrist_pos_x[0]
    ]  # keep the ones that x > ranges.l_wrist_pos_x[0]
    l_positions = l_positions[
        l_positions[:, 0] < ranges.l_wrist_pos_x[1]
    ]  # keep the ones that x < ranges.l_wrist_pos_x[1]
    l_positions = l_positions[
        l_positions[:, 1] > ranges.l_wrist_pos_y[0]
    ]  # keep the ones that y > ranges.l_wrist_pos_y[0]
    l_positions = l_positions[
        l_positions[:, 1] < ranges.l_wrist_pos_y[1]
    ]  # keep the ones that y < ranges.l_wrist_pos_y[1]
    l_positions = l_positions[
        l_positions[:, 2] > ranges.l_wrist_pos_z[0]
    ]  # keep the ones that z > ranges.l_wrist_pos_z[0]
    l_positions = l_positions[
        l_positions[:, 2] < ranges.l_wrist_pos_z[1]
    ]  # keep the ones that z < ranges.l_wrist_pos_z[1]

    r_positions = torch.randn(num_points, 3)  # right wrist positions
    r_positions = (
        r_positions / r_positions.norm(dim=-1, keepdim=True) * ranges.wrist_max_radius
    )  # within a sphere, [-radius, +radius]
    r_positions = r_positions[
        r_positions[:, 0] > ranges.r_wrist_pos_x[0]
    ]  # keep the ones that x > ranges.r_wrist_pos_x[0]
    r_positions = r_positions[
        r_positions[:, 0] < ranges.r_wrist_pos_x[1]
    ]  # keep the ones that x < ranges.r_wrist_pos_x[1]
    r_positions = r_positions[
        r_positions[:, 1] > ranges.r_wrist_pos_y[0]
    ]  # keep the ones that y > ranges.r_wrist_pos_y[0]
    r_positions = r_positions[
        r_positions[:, 1] < ranges.r_wrist_pos_y[1]
    ]  # keep the ones that y < ranges.r_wrist_pos_y[1]
    r_positions = r_positions[
        r_positions[:, 2] > ranges.r_wrist_pos_z[0]
    ]  # keep the ones that z > ranges.r_wrist_pos_z[0]
    r_positions = r_positions[
        r_positions[:, 2] < ranges.r_wrist_pos_z[1]
    ]  # keep the ones that z < ranges.r_wrist_pos_z[1]

    num_pairs = min(l_positions.size(0), r_positions.size(0))
    positions = torch.stack([l_positions[:num_pairs], r_positions[:num_pairs]], dim=1)  # (num_pairs, 2, 3)

    # rotation (quaternion)
    quaternions = torch.randn(num_pairs, 2, 4)
    quaternions = quaternions / quaternions.norm(dim=-1, keepdim=True)

    # concat
    wp = torch.cat([positions, quaternions], dim=-1)  # (num_pairs, 2, 7)
    # repeat for num_wp
    wp = wp.unsqueeze(1).repeat(1, num_wp, 1, 1)  # (num_pairs, num_wp, 2, 7)
    log.info("===> [sample_wp] return shape:", wp.shape)
    return wp.to(device), num_pairs, num_wp
