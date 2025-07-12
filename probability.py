import re
import numpy as np
import matplotlib.pyplot as plt

def extract_high_low_confidences(filepath):
    high_confidences = []
    low_confidences = []

    with open(filepath, 'r') as f:
        lines = f.readlines()

    confidences = []
    for line in lines:
        line = line.strip()
        match = re.search(r'Class \d+:\s+([0-9.]+)', line)
        if match:
            prob = float(match.group(1))
            confidences.append(prob)

            # Mỗi image có 10 class → tính và reset
            if len(confidences) == 10:
                max_conf = max(confidences)
                rest = confidences.copy()
                rest.remove(max_conf)
                avg_rest = np.mean(rest) if rest else 0

                high_confidences.append(max_conf)
                low_confidences.append(avg_rest)

                confidences = []  # reset cho ảnh kế tiếp

    return high_confidences, low_confidences

# === Load dữ liệu ===
distributed_file = 'input_data/master_slave/distributed_probability.txt'
single_file = 'inptu_data/single_container/single_probability.txt'

# === Extract từng loại ===
single_highs, single_lows = extract_high_low_confidences(single_file)
dist_highs, dist_lows = extract_high_low_confidences(distributed_file)

# === Tính trung bình tất cả sample ===
single_high_avg = np.mean(single_highs) if single_highs else 0
single_low_avg = np.mean(single_lows) if single_lows else 0
dist_high_avg = np.mean(dist_highs) if dist_highs else 0
dist_low_avg = np.mean(dist_lows) if dist_lows else 0

# === In thống kê ===
print(f"Single - Avg High Confidence: {single_high_avg:.4f}")
print(f"Single - Avg Low Confidence: {single_low_avg:.4f}")
print(f"Distributed - Avg High Confidence: {dist_high_avg:.4f}")
print(f"Distributed - Avg Low Confidence: {dist_low_avg:.4f}")

# === Vẽ biểu đồ ===
labels = ['Single\nHigh Confidence', 'Single\nLow Confidence', 'Distributed\nHigh Confidence', 'Distributed\nLow Confidence']
values = [single_high_avg, single_low_avg, dist_high_avg, dist_low_avg]
colors = ['lightgreen', 'lightyellow', 'darkgreen', 'gold']

fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.bar(labels, values, color=colors, edgecolor='black')

# Ghi nhãn
for bar, value in zip(bars, values):
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.005,
            f'{value:.4f}', ha='center', va='bottom', fontweight='bold')

ax.set_title("Avg High vs Low Confidence per Sample")
ax.set_ylabel("Confidence")
ax.set_ylim(0, max(values) * 1.15)
ax.grid(True, axis='y', linestyle='--', alpha=0.4)
ax.text(0.5, 0.95, f'Samples - Single: {len(single_highs)} | Distributed: {len(dist_highs)}',
        transform=ax.transAxes, ha='center', fontsize=9,
        bbox=dict(boxstyle='round,pad=0.3', facecolor='lightblue', alpha=0.6))

plt.tight_layout()
plt.savefig('output_data/master_slave/confidence_summary.png', dpi=300)
print("✅ Biểu đồ đã lưu: output_data/master_slave/confidence_summary.png")
plt.show()
