import pandas as pd
import matplotlib.pyplot as plt

# === Đọc file CSV ===
csv_file = 'input_data/master_slave/sync_report.csv'  # Đổi tên file cho đúng
df = pd.read_csv(csv_file)

# === Vẽ biểu đồ chỉ hiển thị điểm ===
plt.figure(figsize=(10, 6))

plt.scatter(df['Update'], df['Master Time'], label='Master Time', color='blue', marker='o')
plt.scatter(df['Update'], df['Slaver Time'], label='Slaver Time', color='green', marker='s')

plt.title('Communication Time vs. Update (Scatter Only)')
plt.xlabel('Update')
plt.ylabel('Time (seconds)')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)

# === Lưu hình ảnh ===
plt.tight_layout()
plt.savefig('output_data/master_slave/comm_time_scatter.png', dpi=300)
print("Đã lưu biểu đồ vào: output_data/master_slave/comm_time_scatter.png")

plt.show()
