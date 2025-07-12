import re
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

def extract_distributed_loss_grouped(filepath):
    """
    Gom loss master và slaver lại thành từng cặp theo thời điểm cập nhật.
    Mỗi cặp (loss_m, loss_s) được gộp trung bình thành một điểm.
    Trả về: list loss trung bình.
    """
    master_losses = {}
    slaver_losses = {}

    with open(filepath, 'r') as f:
        for line in f:
            match = re.search(r'\*\*\* (Average )?loss after (\d+) images: ([0-9.]+)', line)
            if match:
                img = int(match.group(2))
                loss = float(match.group(3))
                if img < 30000:
                    master_losses[img] = loss
                else:
                    slaver_losses[img] = loss

    # Ghép theo chỉ số cập nhật
    img_step = sorted(set(master_losses.keys()) | set(slaver_losses.keys()))
    y_vals = []

    for img in sorted(img_step):
        if img in master_losses and (img + 30000) in slaver_losses:
            avg_loss = (master_losses[img] + slaver_losses[img + 30000]) / 2
            y_vals.append(avg_loss)

    return y_vals

def extract_single_loss_grouped(filepath):
    """
    Trích loss từ single model.
    """
    y_vals = []
    seen = set()

    with open(filepath, 'r') as f:
        for line in f:
            match = re.search(r'loss after (\d+) images: ([0-9.]+)', line)
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
y_dist = extract_distributed_loss_grouped(distributed_file)
y_single = extract_single_loss_grouped(single_file)

print(f"Distributed points: {len(y_dist)}")
print(f"Single points: {len(y_single)}")

# Tạo trục X có độ dài bằng nhau để so sánh
# Distributed: 30 điểm spread over [0, 1]  
# Single: 60 điểm spread over [0, 1]
x_dist = np.linspace(0, 1, len(y_dist))  # 30 điểm từ 0 đến 1
x_single = np.linspace(0, 1, len(y_single))  # 60 điểm từ 0 đến 1

# Vẽ biểu đồ
plt.figure(figsize=(12, 8))
plt.plot(x_dist, y_dist, label=f'Distributed (avg master+slaver) - {len(y_dist)} points', 
         marker='o', linewidth=2, markersize=6)
plt.plot(x_single, y_single, label=f'Single Training - {len(y_single)} points', 
         marker='x', linewidth=2, markersize=4)

# Tính final loss
final_dist_loss = y_dist[-1]
final_single_loss = y_single[-1]

plt.xlabel('Normalized Training Progress (0 = start, 1 = end)')
plt.ylabel('Loss')
plt.title('Loss Comparison: Distributed vs Single Training')
plt.legend()
plt.grid(True, alpha=0.3)

# Thêm text hiển thị final loss bên trong biểu đồ, dưới legend
info_text = f'Final Loss\nDistributed: {final_dist_loss:.4f} | Single: {final_single_loss:.4f}'
plt.text(0.5, 0.85, info_text, transform=plt.gca().transAxes, ha='center', va='top', 
         fontsize=10, bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.8))

plt.tight_layout()

# Lưu thành file PNG
plt.savefig('output_data/master_slave/loss_comparison.png', dpi=300, bbox_inches='tight')
print("Đã lưu biểu đồ vào file: loss_comparison.png")

# Hiển thị thống kê
print(f"\nDistributed - Final loss: {final_dist_loss:.4f}, Min loss: {min(y_dist):.4f}")
print(f"Single - Final loss: {final_single_loss:.4f}, Min loss: {min(y_single):.4f}")

plt.show()