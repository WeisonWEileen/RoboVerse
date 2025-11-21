import matplotlib.pyplot as plt
import numpy as np

# --- Data ---
intervals = [
    "[-2.30,-1.84]",
    "[-1.84,-1.38]",
    "[-1.38,-0.92]",
    "[-0.92,-0.46]",
    "[-0.46,0.00]",
    "[0.00,0.46]",
    "[0.46,0.92]",
    "[0.92,1.38]",
    "[1.38,1.84]",
    "[1.84,2.30]",
]
success_no_occlusion= [1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 0.98, 0.62]
success_occlusion=[1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 0.82, 0.90]
# --- Plot ---
x = np.arange(len(intervals))
width = 0.38  # bar width

plt.figure(figsize=(16, 6))
plt.bar(x - width / 2, success_no_occlusion, width, label="Evalute", color="skyblue")
plt.bar(x + width / 2, success_occlusion, width, label="Training", color="salmon")

# --- Add labels above bars ---
for i, v in enumerate(success_no_occlusion):
    plt.text(i - width / 2, v + 0.02, f"{v * 100:.0f}%", ha="center", va="bottom", fontsize=10)

for i, v in enumerate(success_occlusion):
    plt.text(i + width / 2, v + 0.02, f"{v * 100:.0f}%", ha="center", va="bottom", fontsize=10)

plt.xticks(x, intervals, rotation=45, ha="right")
plt.ylim(0, 1.15)
plt.ylabel("Success Rate")
plt.title("Success Rate Comparison Eval on Train/Test material set (No Occlusion vs Occlusion)")
plt.legend()
plt.tight_layout()

# save the plot to a file
plt.savefig("success_rate_comparison_randomize_material_2025_1113_010110.png")
