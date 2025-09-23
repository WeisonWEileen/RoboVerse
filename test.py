#!/usr/bin/env python3
"""
Script to plot average curves for knee, hip, and ankle data from CSV files.
Each file contains multiple rows of comma-separated values.
This script calculates the average across all rows for each dataset and plots the curves.
"""

import numpy as np
import matplotlib

matplotlib.use("Agg")  # Use non-interactive backend
import matplotlib.pyplot as plt
import os


def read_csv_data(filename):
    """
    Read CSV data from file and return as numpy array.
    Each row contains comma-separated values.
    """
    data = []
    with open(filename, "r") as file:
        for line in file:
            line = line.strip()
            if line:  # Skip empty lines
                # Split by comma and convert to float
                row_data = [float(x) for x in line.split(",")]
                data.append(row_data)
    return np.array(data)


def calculate_average_curve(data):
    """
    Calculate the average curve across all rows.
    Returns the mean values and standard deviation for error bars.
    """
    mean_curve = np.mean(data, axis=0)
    std_curve = np.std(data, axis=0)
    return mean_curve, std_curve


def plot_average_curves():
    """
    Load data from all three files and create plots showing average curves.
    """
    # Define file paths
    data_dir = "outputs/test"
    files = {
        "Knee": os.path.join(data_dir, "Comma_Separated_Knee_Data.txt"),
        "Hip": os.path.join(data_dir, "Comma_Separated_Hip_Data.txt"),
        "Ankle": os.path.join(data_dir, "Comma_Separated_Ankle_Data.txt"),
    }

    # Colors for each curve
    colors = {"Knee": "blue", "Hip": "red", "Ankle": "green"}

    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle("Average Curves for Knee, Hip, and Ankle Data", fontsize=16, fontweight="bold")

    # Individual subplot positions
    subplot_positions = {"Knee": (0, 0), "Hip": (0, 1), "Ankle": (1, 0)}

    # Store data for combined plot
    all_data = {}

    # Process each dataset
    for joint_name, filename in files.items():
        print(f"Processing {joint_name} data from {filename}...")

        # Read data
        try:
            data = read_csv_data(filename)
            print(f"  Loaded {data.shape[0]} rows with {data.shape[1]} data points each")

            # Calculate average curve
            mean_curve, std_curve = calculate_average_curve(data)
            all_data[joint_name] = (mean_curve, std_curve)

            # Create individual subplot
            row, col = subplot_positions[joint_name]
            ax = axes[row, col]

            # Create x-axis (time points or sample indices)
            x_values = np.arange(len(mean_curve))

            # Plot mean curve with error bars (standard deviation)
            ax.plot(x_values, mean_curve, color=colors[joint_name], linewidth=2, label=f"{joint_name} Average")
            ax.fill_between(
                x_values,
                mean_curve - std_curve,
                mean_curve + std_curve,
                color=colors[joint_name],
                alpha=0.3,
                label=f"{joint_name} ±1 STD",
            )

            ax.set_title(f"{joint_name} Data - Average Curve", fontweight="bold")
            ax.set_xlabel("Sample Index")
            ax.set_ylabel("Value")
            ax.grid(True, alpha=0.3)
            ax.legend()

            # Add statistics text
            stats_text = f"Mean: {np.mean(mean_curve):.2f}\nSTD: {np.mean(std_curve):.2f}\nSamples: {data.shape[0]}"
            ax.text(
                0.02,
                0.98,
                stats_text,
                transform=ax.transAxes,
                verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
            )

        except Exception as e:
            print(f"  Error processing {joint_name}: {e}")
            continue

    # Combined plot (bottom right)
    ax_combined = axes[1, 1]
    for joint_name, (mean_curve, std_curve) in all_data.items():
        x_values = np.arange(len(mean_curve))
        ax_combined.plot(x_values, mean_curve, color=colors[joint_name], linewidth=2, label=f"{joint_name} Average")

    ax_combined.set_title("All Joints - Average Curves Comparison", fontweight="bold")
    ax_combined.set_xlabel("Sample Index")
    ax_combined.set_ylabel("Value")
    ax_combined.grid(True, alpha=0.3)
    ax_combined.legend()

    # Adjust layout and save
    plt.tight_layout()

    # Save the plot
    output_filename = "average_curves_plot.png"
    plt.savefig(output_filename, dpi=300, bbox_inches="tight")
    print(f"\nPlot saved as: {output_filename}")

    # Note: plt.show() is commented out since we're using non-interactive backend
    # plt.show()

    # Print summary statistics
    print("\n" + "=" * 50)
    print("SUMMARY STATISTICS")
    print("=" * 50)
    for joint_name, (mean_curve, std_curve) in all_data.items():
        print(f"\n{joint_name.upper()} DATA:")
        print(f"  Data points per sample: {len(mean_curve)}")
        print(f"  Average value across all data: {np.mean(mean_curve):.3f}")
        print(f"  Overall standard deviation: {np.mean(std_curve):.3f}")
        print(f"  Min value: {np.min(mean_curve):.3f}")
        print(f"  Max value: {np.max(mean_curve):.3f}")
        print(f"  Range: {np.max(mean_curve) - np.min(mean_curve):.3f}")


if __name__ == "__main__":
    print("Starting analysis of knee, hip, and ankle data...")
    print("=" * 50)

    # Check if data directory exists
    if not os.path.exists("outputs/test"):
        print("Error: outputs/test directory not found!")
        print("Please make sure you're running this script from the project root directory.")
        exit(1)

    # Generate plots
    plot_average_curves()

    print("\nAnalysis complete!")
