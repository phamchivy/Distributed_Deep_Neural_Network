import re
import matplotlib.pyplot as plt
import numpy as np

def extract_distributed_accuracy_grouped(filepath):
    """
    Gom accuracy master và slaver lại thành từng cặp theo thời điểm cập nhật.
    Mỗi cặp (acc_m, acc_s) được gộp trung bình thành một điểm.
    Trả về: list accuracy trung bình.
    """
    master_accuracies = {}
    slaver_accuracies = {}

    with open(filepath, 'r') as f:
        for line in f:
            # Pattern cho accuracy: *** After XXXX images: XX.XX% accuracy ***
            match = re.search(r'\*\*\* After (\d+) images: ([0-9.]+)% accuracy \*\*\*', line)
            if match:
                img = int(match.group(1))
                accuracy = float(match.group(2))
                if img < 30000:
                    master_accuracies[img] = accuracy
                else:
                    slaver_accuracies[img] = accuracy

    # Ghép theo chỉ số cập nhật
    img_step = sorted(set(master_accuracies.keys()) | set(slaver_accuracies.keys()))
    y_vals = []

    for img in sorted(img_step):
        if img in master_accuracies and (img + 30000) in slaver_accuracies:
            avg_accuracy = (master_accuracies[img] + slaver_accuracies[img + 30000]) / 2
            y_vals.append(avg_accuracy)

    return y_vals

def extract_single_accuracy_grouped(filepath):
    """
    Trích accuracy từ single model.
    """
    y_vals = []
    seen = set()

    with open(filepath, 'r') as f:
        for line in f:
            # Pattern cho single accuracy
            match = re.search(r'After (\d+) images: ([0-9.]+)% accuracy', line)
            if match:
                img = int(match.group(1))
                if img in seen:
                    continue
                seen.add(img)
                y_vals.append(float(match.group(2)))
    return y_vals

# === Cấu hình file ===
distributed_file = 'input_data/master_slave/distributed.txt'
single_file = 'input_data/single_container/single.txt'

# Lấy dữ liệu
y_dist = extract_distributed_accuracy_grouped(distributed_file)
y_single = extract_single_accuracy_grouped(single_file)

print(f"Distributed points: {len(y_dist)}")
print(f"Single points: {len(y_single)}")

# Tạo trục X có độ dài bằng nhau để so sánh
x_dist = np.linspace(0, 1, len(y_dist))  # Distributed points từ 0 đến 1
x_single = np.linspace(0, 1, len(y_single))  # Single points từ 0 đến 1

# Tính final accuracy
final_dist_accuracy = y_dist[-1] if y_dist else 0
final_single_accuracy = y_single[-1] if y_single else 0

# Vẽ biểu đồ
plt.figure(figsize=(12, 8))
plt.plot(x_dist, y_dist, label=f'Distributed (avg master+slaver) - {len(y_dist)} points', 
         marker='o', linewidth=2, markersize=6)
plt.plot(x_single, y_single, label=f'Single Training - {len(y_single)} points', 
         marker='x', linewidth=2, markersize=4)

plt.xlabel('Normalized Training Progress (0 = start, 1 = end)')
plt.ylabel('Accuracy (%)')
plt.title('Accuracy Comparison: Distributed vs Single Training')
plt.legend()
plt.grid(True, alpha=0.3)

# Thêm text hiển thị final accuracy bên trong biểu đồ, dưới legend
info_text = f'Final Accuracy\nDistributed: {final_dist_accuracy:.2f}% | Single: {final_single_accuracy:.2f}%'
plt.text(0.5, 0.85, info_text, transform=plt.gca().transAxes, ha='center', va='top', 
         fontsize=10, bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', alpha=0.8))

plt.tight_layout()

# Lưu thành file PNG
plt.savefig('output_data/master_slave/accuracy_comparison.png', dpi=300, bbox_inches='tight')
print("Đã lưu biểu đồ vào file: accuracy_comparison.png")

# Hiển thị thống kê
print(f"\nDistributed - Final accuracy: {final_dist_accuracy:.2f}%, Max accuracy: {max(y_dist):.2f}%")
print(f"Single - Final accuracy: {final_single_accuracy:.2f}%, Max accuracy: {max(y_single):.2f}%")

plt.show()