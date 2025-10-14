"""Evaluator for tracking and visualizing metrics during policy evaluation."""

from __future__ import annotations

import os
from collections import deque

import matplotlib

matplotlib.use("Agg")  # Use non-interactive backend for saving plots without display
import matplotlib.pyplot as plt
import numpy as np
import torch
from loguru import logger as log


class SeeFlagEvaluator:
    """Evaluator for tracking see_flag metric and generating real-time visualizations."""

    def __init__(
        self,
        num_envs: int = 1,
        window_size: int = 100,
        save_dir: str | None = None,
        plot_interval: int = 10,
        enable_plot: bool = True,
    ):
        """Initialize the evaluator.

        Args:
            num_envs: Number of environments to track
            window_size: Size of the sliding window for computing rolling average
            save_dir: Directory to save plots and data. If None, will not save.
            plot_interval: Update plot every N steps
            enable_plot: Whether to enable real-time plotting
        """
        self.num_envs = num_envs
        self.window_size = window_size
        self.save_dir = save_dir
        self.plot_interval = plot_interval
        self.enable_plot = enable_plot

        # Create save directory if needed
        if self.save_dir is not None:
            os.makedirs(self.save_dir, exist_ok=True)

        # Data storage
        self.see_flag_history: list[float] = []  # Store all see_flag averages
        self.see_flag_window = deque(maxlen=window_size)  # Rolling window
        self.step_count = 0

        # Per-environment tracking
        self.env_see_flag_counts = np.zeros(num_envs)
        self.env_step_counts = np.zeros(num_envs)

        # Statistics
        self.total_see_flag_count = 0
        self.total_steps = 0

        # Episode reward tracking
        self.episode_reward_sums = {}  # Dict of reward_name -> list of episode rewards
        self.episode_reward_counts = {}  # Dict of reward_name -> count
        self.episode_count = 0

        # Plotting
        if self.enable_plot:
            self.fig, self.axes = plt.subplots(3, 1, figsize=(12, 12))
            self.fig.suptitle("See Flag Evaluation Metrics", fontsize=14, fontweight="bold")

    def update(self, see_flag: torch.Tensor):
        """Update the evaluator with new see_flag data.

        Args:
            see_flag: Boolean tensor of shape (num_envs,) indicating if object is visible
        """
        # Convert to numpy
        if isinstance(see_flag, torch.Tensor):
            see_flag_np = see_flag.cpu().numpy()
        else:
            see_flag_np = np.array(see_flag)

        # Compute average see_flag for this step
        see_flag_avg = see_flag_np.mean()

        # Update history
        self.see_flag_history.append(see_flag_avg)
        self.see_flag_window.append(see_flag_avg)

        # Update per-environment statistics
        self.env_see_flag_counts += see_flag_np.astype(float)
        self.env_step_counts += 1

        # Update global statistics
        self.total_see_flag_count += see_flag_np.sum()
        self.total_steps += len(see_flag_np)
        self.step_count += 1

        # Update plot
        if self.enable_plot and self.step_count % self.plot_interval == 0:
            self.plot()

    def update_episode_reward(self, episode_sums: dict[str, torch.Tensor]):
        """Update episode reward statistics.

        Args:
            episode_sums: Dictionary of reward_name -> tensor of episode sum rewards
        """
        self.episode_count += 1

        for reward_name, reward_value in episode_sums.items():
            # Convert tensor to float
            if isinstance(reward_value, torch.Tensor):
                reward_val = reward_value.mean().item()  # Take mean if multiple envs
            else:
                reward_val = float(reward_value)

            # Initialize if first time seeing this reward
            if reward_name not in self.episode_reward_sums:
                self.episode_reward_sums[reward_name] = []
                self.episode_reward_counts[reward_name] = 0

            # Store the reward
            self.episode_reward_sums[reward_name].append(reward_val)
            self.episode_reward_counts[reward_name] += 1

    def get_episode_reward_averages(self) -> dict[str, float]:
        """Get average episode rewards for each reward component.

        Returns:
            Dictionary of reward_name -> average reward
        """
        averages = {}
        for reward_name, rewards in self.episode_reward_sums.items():
            if len(rewards) > 0:
                averages[reward_name] = np.mean(rewards)
            else:
                averages[reward_name] = 0.0
        return averages

    def get_rolling_average(self) -> float:
        """Get the rolling average of see_flag over the window."""
        if len(self.see_flag_window) == 0:
            return 0.0
        return np.mean(self.see_flag_window)

    def get_overall_average(self) -> float:
        """Get the overall average see_flag since start."""
        if self.total_steps == 0:
            return 0.0
        return self.total_see_flag_count / self.total_steps

    def get_env_statistics(self) -> dict[str, np.ndarray]:
        """Get per-environment statistics."""
        env_averages = np.zeros(self.num_envs)
        valid_envs = self.env_step_counts > 0
        env_averages[valid_envs] = self.env_see_flag_counts[valid_envs] / self.env_step_counts[valid_envs]

        return {
            "env_averages": env_averages,
            "env_counts": self.env_see_flag_counts,
            "env_steps": self.env_step_counts,
        }

    def plot(self):
        """Generate real-time plots of see_flag metrics."""
        if not self.enable_plot:
            return

        # Clear previous plots
        for ax in self.axes:
            ax.clear()

        # Plot 1: See Flag over time with rolling average
        ax1 = self.axes[0]
        steps = np.arange(len(self.see_flag_history))

        # Plot raw data
        ax1.plot(steps, self.see_flag_history, alpha=0.3, color="blue", label="See Flag (instantaneous)", linewidth=0.5)

        # Plot rolling average
        if len(self.see_flag_history) >= self.window_size:
            rolling_avg = np.convolve(self.see_flag_history, np.ones(self.window_size) / self.window_size, mode="valid")
            rolling_steps = steps[self.window_size - 1 :]
            ax1.plot(
                rolling_steps,
                rolling_avg,
                color="red",
                linewidth=2,
                label=f"Rolling Average (window={self.window_size})",
            )

        # Plot overall average as horizontal line
        overall_avg = self.get_overall_average()
        ax1.axhline(
            y=overall_avg, color="green", linestyle="--", linewidth=2, label=f"Overall Average: {overall_avg:.3f}"
        )

        ax1.set_xlabel("Step", fontsize=11)
        ax1.set_ylabel("See Flag Rate", fontsize=11)
        ax1.set_title("See Flag Over Time", fontsize=12, fontweight="bold")
        ax1.legend(loc="upper right", fontsize=9)
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim([-0.05, 1.05])

        # Plot 2: Per-environment statistics (if multiple envs)
        ax2 = self.axes[1]
        if self.num_envs > 1:
            env_stats = self.get_env_statistics()
            env_indices = np.arange(self.num_envs)

            bars = ax2.bar(env_indices, env_stats["env_averages"], color="steelblue", alpha=0.7, edgecolor="black")

            # Add value labels on bars
            for i, (bar, val) in enumerate(zip(bars, env_stats["env_averages"])):
                height = bar.get_height()
                ax2.text(
                    bar.get_x() + bar.get_width() / 2.0, height, f"{val:.2f}", ha="center", va="bottom", fontsize=8
                )

            ax2.set_xlabel("Environment ID", fontsize=11)
            ax2.set_ylabel("Average See Flag Rate", fontsize=11)
            ax2.set_title("Per-Environment See Flag Statistics", fontsize=12, fontweight="bold")
            ax2.grid(True, alpha=0.3, axis="y")
            ax2.set_ylim([0, 1.05])
        else:
            # For single environment, show statistics text
            stats_text = (
                f"Total Steps: {self.step_count}\n"
                f"Overall Average: {self.get_overall_average():.4f}\n"
                f"Rolling Average: {self.get_rolling_average():.4f}\n"
                f"Total See Count: {int(self.total_see_flag_count)}/{self.total_steps}\n"
                f"Episodes: {self.episode_count}"
            )
            ax2.text(
                0.5,
                0.5,
                stats_text,
                transform=ax2.transAxes,
                fontsize=14,
                verticalalignment="center",
                horizontalalignment="center",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
            )
            ax2.set_xlim([0, 1])
            ax2.set_ylim([0, 1])
            ax2.axis("off")

        # Plot 3: Average episode rewards per reward component
        ax3 = self.axes[2]
        if self.episode_reward_sums:
            reward_averages = self.get_episode_reward_averages()

            # Sort by reward value for better visualization
            sorted_rewards = sorted(reward_averages.items(), key=lambda x: x[1], reverse=True)
            reward_names = [name for name, _ in sorted_rewards]
            reward_values = [val for _, val in sorted_rewards]

            # Create color map - positive rewards in green, negative in red
            colors = ["green" if val >= 0 else "red" for val in reward_values]

            # Create bar plot
            x_pos = np.arange(len(reward_names))
            bars = ax3.bar(x_pos, reward_values, color=colors, alpha=0.7, edgecolor="black")

            # Add value labels on bars
            for i, (bar, val) in enumerate(zip(bars, reward_values)):
                height = bar.get_height()
                y_pos = height if height > 0 else 0
                ax3.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    y_pos,
                    f"{val:.3f}",
                    ha="center",
                    va="bottom" if height > 0 else "top",
                    fontsize=9,
                    fontweight="bold",
                )

            # Format x-axis labels
            ax3.set_xticks(x_pos)
            ax3.set_xticklabels(reward_names, rotation=45, ha="right", fontsize=9)
            ax3.set_xlabel("Reward Component", fontsize=11)
            ax3.set_ylabel("Average Episode Reward", fontsize=11)
            ax3.set_title(
                f"Average Episode Unscaled Rewards ({self.episode_count} episodes)", fontsize=12, fontweight="bold"
            )
            ax3.grid(True, alpha=0.3, axis="y")
            ax3.axhline(y=0, color="black", linestyle="-", linewidth=0.5)
        else:
            # No episode data yet
            ax3.text(
                0.5,
                0.5,
                "No episode data collected yet",
                transform=ax3.transAxes,
                fontsize=12,
                verticalalignment="center",
                horizontalalignment="center",
            )
            ax3.axis("off")

        plt.tight_layout()

        # Save the plot if save_dir is specified
        if self.save_dir is not None:
            plot_path = os.path.join(self.save_dir, "see_flag_evaluation.png")
            self.fig.savefig(plot_path, dpi=150, bbox_inches="tight")
            plt.close(self.fig)  # Close to free memory
            # Recreate figure for next update
            self.fig, self.axes = plt.subplots(3, 1, figsize=(12, 12))
            self.fig.suptitle("See Flag Evaluation Metrics", fontsize=14, fontweight="bold")

    def save_data(self, filename: str = "see_flag_data.npz"):
        """Save collected data to file.

        Args:
            filename: Name of the file to save data
        """
        if self.save_dir is None:
            log.warning("save_dir is None, cannot save data")
            return

        save_path = os.path.join(self.save_dir, filename)
        env_stats = self.get_env_statistics()
        reward_averages = self.get_episode_reward_averages()

        # Prepare episode reward data for saving
        episode_reward_data = {}
        for reward_name, rewards in self.episode_reward_sums.items():
            episode_reward_data[f"episode_rewards_{reward_name}"] = np.array(rewards)

        np.savez(
            save_path,
            see_flag_history=np.array(self.see_flag_history),
            env_averages=env_stats["env_averages"],
            env_counts=env_stats["env_counts"],
            env_steps=env_stats["env_steps"],
            overall_average=self.get_overall_average(),
            total_steps=self.total_steps,
            episode_count=self.episode_count,
            **episode_reward_data,  # Unpack episode reward data
        )
        log.info(f"Data saved to {save_path}")

        # Also save reward averages as a separate readable file
        if reward_averages:
            reward_avg_path = os.path.join(self.save_dir, "episode_reward_averages.txt")
            with open(reward_avg_path, "w") as f:
                f.write(f"Episode Reward Averages ({self.episode_count} episodes):\n")
                f.write("=" * 60 + "\n")
                for reward_name, avg_value in sorted(reward_averages.items(), key=lambda x: x[1], reverse=True):
                    f.write(f"{reward_name:40s}: {avg_value:10.4f}\n")
            log.info(f"Episode reward averages saved to {reward_avg_path}")

    def print_summary(self):
        """Print a summary of the evaluation results."""
        separator = "=" * 60
        log.info(f"\n{separator}")
        log.info("See Flag Evaluation Summary")
        log.info(separator)
        log.info(f"Total Steps: {self.step_count}")
        log.info(f"Total Episodes: {self.episode_count}")
        log.info(f"Overall Average See Flag: {self.get_overall_average():.4f}")
        log.info(f"Rolling Average See Flag (last {len(self.see_flag_window)} steps): {self.get_rolling_average():.4f}")
        log.info(f"Total Object Visible Count: {int(self.total_see_flag_count)}/{self.total_steps}")

        if self.num_envs > 1:
            env_stats = self.get_env_statistics()
            log.info("\nPer-Environment Statistics:")
            for i in range(self.num_envs):
                log.info(
                    f"  Env {i}: {env_stats['env_averages'][i]:.4f} "
                    f"({int(env_stats['env_counts'][i])}/{int(env_stats['env_steps'][i])} steps)"
                )

        # Print episode reward averages
        if self.episode_reward_sums:
            log.info(f"\nAverage Episode Rewards ({self.episode_count} episodes):")
            reward_averages = self.get_episode_reward_averages()
            for reward_name, avg_value in sorted(reward_averages.items(), key=lambda x: x[1], reverse=True):
                log.info(f"  {reward_name:40s}: {avg_value:10.4f}")

        log.info(separator)

    def close(self):
        """Close the evaluator and cleanup resources."""
        # Print final summary
        self.print_summary()

        # Save data and final plot if save_dir is specified
        if self.save_dir is not None:
            self.save_data()
            # Generate and save final plot
            if self.enable_plot:
                self.plot()  # Generate final plot
                final_plot_path = os.path.join(self.save_dir, "see_flag_evaluation_final.png")
                self.fig.savefig(final_plot_path, dpi=300, bbox_inches="tight")
                log.info(f"Final plot saved to {final_plot_path}")
                plt.close(self.fig)  # Close figure
