"""This script is for real-time visualization of critic values.

Load a checkpoint and visualize the critic value in real-time using a single environment.
"""

import rootutils

rootutils.setup_root(__file__, pythonpath=True)

import random
from collections import deque

import matplotlib
import matplotlib.pyplot as plt
import torch
from loguru import logger as log

# Try to use interactive backend for real-time visualization
try:
    matplotlib.use("TkAgg")
except Exception:
    try:
        matplotlib.use("Qt5Agg")
    except Exception:
        matplotlib.use("Agg")  # Fallback to non-interactive if needed
        log.warning("Using non-interactive backend. Real-time visualization may not work.")

from humanoid_visualrl.actor_critic.on_policy_runner import OnPolicyRunner
from humanoid_visualrl.utils.utils import (
    get_args,
    get_load_path,
    load_task_cfg,
    load_wrapper,
)
from metasim.scenario.lights import DomeLightCfg
from metasim.scenario.scenario import ScenarioCfg


class CriticVisualizer:
    """Real-time visualizer for critic values."""

    def __init__(self, window_size=200):
        """Initialize the visualizer.

        Args:
            window_size: Initial window size for x-axis (not used for data storage)
        """
        self.window_size = window_size
        # Store all historical data without maxlen limit
        self.critic_values = deque()
        self.steps = deque()
        self.step_count = 0

        # Setup matplotlib figure
        self.fig, self.ax = plt.subplots(figsize=(12, 6))
        (self.line,) = self.ax.plot([], [], "b-", linewidth=2, label="Critic Value")
        self.ax.set_xlabel("Step", fontsize=12)
        self.ax.set_ylabel("Critic Value", fontsize=12)
        self.ax.set_title("Real-time Critic Value Visualization (All History)", fontsize=14, fontweight="bold")
        self.ax.grid(True, alpha=0.3)
        self.ax.legend()
        self.ax.set_xlim(0, window_size)

        plt.ion()  # Turn on interactive mode
        plt.show(block=False)

    def update(self, critic_value: float):
        """Update the plot with a new critic value.

        Args:
            critic_value: The critic value to add
        """
        self.critic_values.append(critic_value)
        self.steps.append(self.step_count)
        self.step_count += 1

        if len(self.critic_values) > 1:
            # Update the plot with all historical data
            self.line.set_data(list(self.steps), list(self.critic_values))

            # Auto-scale y-axis based on all data
            if len(self.critic_values) > 0:
                min_val = min(self.critic_values)
                max_val = max(self.critic_values)
                margin = (max_val - min_val) * 0.1 if max_val != min_val else 0.1
                self.ax.set_ylim(min_val - margin, max_val + margin)

            # Update x-axis to show all historical data
            # Add some padding on the right side
            x_max = max(self.step_count, self.window_size)
            self.ax.set_xlim(0, x_max + 10)

            # Add current value text with statistics
            if hasattr(self, "text"):
                self.text.remove()

            # Calculate statistics
            avg_val = sum(self.critic_values) / len(self.critic_values)
            min_val = min(self.critic_values)
            max_val = max(self.critic_values)

            stats_text = (
                f"Current: {critic_value:.4f}\n"
                f"Step: {self.step_count}\n"
                f"Avg: {avg_val:.4f}\n"
                f"Min: {min_val:.4f}\n"
                f"Max: {max_val:.4f}"
            )
            self.text = self.ax.text(
                0.02,
                0.98,
                stats_text,
                transform=self.ax.transAxes,
                fontsize=10,
                verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
            )

            self.fig.canvas.draw()
            self.fig.canvas.flush_events()

    def close(self):
        """Close the visualizer."""
        plt.close(self.fig)


def play(args):
    """Run the trained policy and visualize critic values in real-time.

    Args:
        args: Command line arguments containing configuration
    """
    device = args.device
    load_path = get_load_path(args)

    # Get task cfg from cfg.py in load_path
    task_cfg_cls = load_task_cfg(args)
    task_cfg = task_cfg_cls(
        actor_critic_class=args.actor_critic_class,
        finetune=args.resume,
        occlude_cube=args.eval_occlu,
        enable_grasp=args.enable_grasp,
    )

    # Setup scenario with single environment
    scenario = ScenarioCfg(
        robots=[task_cfg.robot],
        num_envs=1,  # Single environment for visualization
        simulator=args.sim,
        headless=args.headless,
        cameras=[],
    )
    scenario.device = args.device
    scenario.lights = [
        DomeLightCfg(
            intensity=1000.0,
            color=(0.85, 0.9, 1.0),
        ),
    ]

    task_cfg.commands.curriculum = False
    task_cfg.ppo_cfg.resume = True
    task_cfg.ppo_cfg.finetune = True
    task_cfg.ppo_cfg.policy.finetune = True

    # Handle occlusion cube if needed
    if args.eval_occlu:
        from metasim.constants import PhysicStateType
        from metasim.scenario.objects import PrimitiveCubeCfg

        task_cfg.objects.append(
            PrimitiveCubeCfg(
                name="occlusion_cube",
                size=(0.10, 0.10, 0.2),
                color=[0.5, 0.5, 0.5],
                physics=PhysicStateType.RIGIDBODY,
                collision_enabled=True,
                fix_base_link=False,
                default_position=(0.3, 0.1, 0.93),
                mass=0.2,
            ),
        )
        task_cfg.init_states[0]["objects"]["occlusion_cube"] = {
            "pos": torch.tensor([0.3, 0.1, 0.92]),
            "rot": torch.tensor([1.0, 0.0, 0.0, 0.0]),
        }
        task_cfg.filter_pairs.append((task_cfg.robot, "occlusion_cube"))
        task_cfg.filter_pairs.append(("object", "occlusion_cube"))

    # Add objects
    scenario.objects = task_cfg.objects
    scenario.cameras = task_cfg.cameras

    # Task assign and override
    scenario.sim_params = task_cfg.sim_params
    scenario.decimation = task_cfg.decimation
    scenario.render_interval = scenario.decimation
    scenario.task = task_cfg
    scenario.env_spacing = task_cfg.env_spacing

    task_cfg.randomization = False
    scenario.filter_pairs = task_cfg.filter_pairs if hasattr(task_cfg, "filter_pairs") else []

    from humanoid_visualrl.wrapper.active_vision_cube_wrapper import ActiveVisionWrapper

    assert not (args.eval_randomize_material_test and args.eval_randomize_material_train), (
        "Must evaluate material in test or train mode, not both"
    )

    if args.eval_randomize_material_test:
        task_cfg.mode = "test"
        task_cfg.randomize_material = True
        log.info("Evaluating material in test mode")
    elif args.eval_randomize_material_train:
        task_cfg.mode = "train"
        task_cfg.randomize_material = True
        log.info("Evaluating material in train mode")

    env_wrapper: ActiveVisionWrapper = load_wrapper(args, scenario)

    # Load policy
    ppo_runner = OnPolicyRunner(
        env=env_wrapper,
        train_cfg=env_wrapper.train_cfg,
        device=device,
        use_vision=task_cfg.use_vision,
    )
    ppo_runner.load(load_path, load_optimizer=False, device=device)
    policy = ppo_runner.get_inference_policy(device=env_wrapper.device)

    # Get actor_critic for evaluating critic
    actor_critic = ppo_runner.alg.policy
    actor_critic.eval()

    # Handle occlusion cube collision filtering
    if args.eval_occlu:
        env_wrapper.env.filter_collisions(env_wrapper.robot.name, "occlusion_cube")
        env_wrapper.env.filter_collisions("object", "occlusion_cube")

    # Initialize environment
    env_wrapper.init_states.objects["object"].root_state[0, 1] = 0.0
    env_wrapper.env.set_states(env_wrapper.init_states)
    env_wrapper.env._render_viewport = True
    env_wrapper.env.init_marker_viz()
    env_wrapper._update_camera_pose = True
    obs, _ = env_wrapper.get_observations()

    # Initialize visualizer
    visualizer = CriticVisualizer(window_size=200)
    log.info("Critic visualizer initialized. Close the plot window to stop.")

    # Reset interval
    reset_interval = 275
    obj_rand_range = task_cfg.randomize_object_yaw_range
    radius = task_cfg.randomize_object_radius

    try:
        for step in range(10000):
            # Reset environment periodically
            if step % reset_interval == 0:
                # Sample random yaw
                # yaw = (random.random() - 0.5) * 2 * obj_rand_range
                yaw = 2.2
                yaw = torch.tensor(yaw, device=env_wrapper.device)

                radius_bias = 0.07 * torch.ones(1, device=env_wrapper.device)
                object_x = torch.cos(yaw) * (radius + radius_bias)
                object_y = torch.sin(yaw) * (radius + radius_bias)

                object_state = env_wrapper.init_states.objects["object"].root_state
                object_state[0, 0] = object_x
                object_state[0, 1] = object_y

                env_wrapper.env._set_object_pose(
                    env_wrapper.cfg.objects[1], object_state[:, :3], object_state[:, 3:7], env_ids=[0]
                )
                env_wrapper._compute_observations()

                # Reset texture and material
                if task_cfg.randomize_obj_material:
                    env_wrapper.env.randomize_obj_material([0], env_wrapper.obj)

                # Handle occlusion cube randomization
                if args.eval_occlu:
                    env_wrapper._randomize_occlusion_cube(object_state, yaw)

                # Randomize material if needed
                if args.eval_randomize_material_train or args.eval_randomize_material_test:
                    env_wrapper.domain_randomization_helper.randomization(env_ids=[0], force_randomize=True)

                # Reset policy hidden states
                ppo_runner.alg.policy.reset([0])

                log.info(
                    f"Step {step}: Reset - Object at ({object_x.item():.3f}, {object_y.item():.3f}), yaw={yaw.item():.3f}"
                )

            # Get action from policy
            if task_cfg.use_vision:
                actions = policy(obs)
            else:
                actions = policy(obs.detach())

            # Evaluate critic value
            with torch.no_grad():
                # Get critic observations from extra_buf or use obs directly
                # For vision models, obs is (state, vision) tuple
                # For non-vision models, obs might be just state tensor
                critic_obs = env_wrapper.extra_buf.get("observations", {}).get("critic", obs)

                # Evaluate critic - hidden states are managed internally by the model
                critic_value = actor_critic.evaluate(critic_obs, masks=None, hidden_states=None)

                # Extract scalar value (assuming single environment)
                if isinstance(critic_value, torch.Tensor):
                    critic_val = critic_value[0].item() if critic_value.numel() > 1 else critic_value.item()
                else:
                    critic_val = float(critic_value)

            # Update visualizer
            visualizer.update(critic_val)

            # Step environment
            obs, _, _, infos = env_wrapper.step_evaluate(actions.detach())
            state = env_wrapper.env.get_states()
            env_wrapper._refreshed_tensors(state)

            # Update camera visualization
            camera_pos = env_wrapper.camera_pos_w[:, :3]
            camera_quat = env_wrapper.camera_quat_w[:, :4]
            camera_direction = env_wrapper.object_pose_buf[:, :3] - camera_pos

            env_wrapper._update_marker_viz(
                camera_pos,
                camera_quat,
                camera_direction,
            )

            # Print critic value periodically
            if step % 10 == 0:
                log.info(f"Step {step}: Critic Value = {critic_val:.4f}, See Flag = {env_wrapper.see_flag[0].item()}")

            # Check if plot window is closed
            if not plt.get_fignums():
                log.info("Plot window closed. Stopping visualization.")
                break

    except KeyboardInterrupt:
        log.info("Interrupted by user.")
    finally:
        visualizer.close()
        env_wrapper.env.close()
        log.info("Visualization closed.")


if __name__ == "__main__":
    args = get_args()
    play(args)
