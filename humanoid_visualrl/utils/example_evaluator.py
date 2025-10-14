"""Example script demonstrating how to use the SeeFlagEvaluator independently."""

from __future__ import annotations

import numpy as np
import torch

from humanoid_visualrl.utils.evaluator import SeeFlagEvaluator


def simulate_see_flag_data(num_steps: int = 1000, num_envs: int = 1) -> list[torch.Tensor]:
    """Simulate see_flag data for demonstration.

    This simulates a scenario where the robot gradually learns to see the object more frequently.

    Args:
        num_steps: Number of simulation steps
        num_envs: Number of parallel environments

    Returns:
        List of see_flag tensors
    """
    see_flag_data = []

    for step in range(num_steps):
        # Simulate improving performance over time
        # Start with 30% success rate, improve to 80%
        base_prob = 0.3 + (0.5 * step / num_steps)

        # Add some noise to make it realistic
        noise = np.random.normal(0, 0.1)
        prob = np.clip(base_prob + noise, 0, 1)

        # Generate see_flag for each environment
        see_flag = torch.rand(num_envs) < prob
        see_flag_data.append(see_flag)

    return see_flag_data


def main():
    """Main demonstration function."""
    print("SeeFlagEvaluator Demonstration")
    print("=" * 60)

    # Configuration
    num_steps = 1000
    num_envs = 4

    # Initialize evaluator
    evaluator = SeeFlagEvaluator(
        num_envs=num_envs,
        window_size=100,
        save_dir="./example_evaluation_results",
        plot_interval=10,
        enable_plot=True,
    )

    print(f"Running simulation with {num_steps} steps and {num_envs} environments...")

    # Generate simulated data
    see_flag_data = simulate_see_flag_data(num_steps, num_envs)

    # Update evaluator with simulated data
    for step, see_flag in enumerate(see_flag_data):
        evaluator.update(see_flag)

        # Print progress every 100 steps
        if (step + 1) % 100 == 0:
            rolling_avg = evaluator.get_rolling_average()
            overall_avg = evaluator.get_overall_average()
            print(f"Step {step + 1}/{num_steps}: Rolling Avg={rolling_avg:.3f}, Overall Avg={overall_avg:.3f}")

    # Close and save results
    evaluator.close()

    print("\nDemonstration complete!")
    print("Check './example_evaluation_results/' for saved plots and data.")


if __name__ == "__main__":
    main()
