import pandas as pd
import matplotlib.pyplot as plt

# Đọc dữ liệu từ file CSV
df = pd.read_csv('input_data/master_slave/comm_times.csv')

# Vẽ biểu đồ đường
plt.figure(figsize=(10, 6))
plt.plot(df['batch_size'], df['communication_time'],
         marker='o', linestyle='-', color='blue', linewidth=2, label='Comm. Time')

# Cài đặt nhãn trục, tiêu đề, lưới
plt.xlabel('Batch Size', fontsize=13)
plt.ylabel('Communication Time (s)', fontsize=13)
plt.title('Communication Time vs Batch Size', fontsize=15, fontweight='bold')
plt.grid(True, linestyle='--', alpha=0.5)

# Gắn nhãn từng điểm với vị trí thông minh
for x, y in zip(df['batch_size'], df['communication_time']):
    if x >= 10000:
        # Với batch lớn như 10000 trở lên → đặt nhãn phía trên
        plt.text(x, y + 0.05, f'{y:.2f}',
                 ha='center', va='bottom', fontsize=11, fontweight='bold', color='black')
    else:
        # Với batch nhỏ hơn → đặt nhãn lệch sang phải
        plt.text(x + (x * 0.015), y +0.04, f'{y:.2f}',
                 ha='left', va='bottom', fontsize=11, fontweight='bold', color='black')

# Hiển thị chú thích và lưu ảnh
plt.legend()
plt.tight_layout()
plt.savefig('output_data/master_slave/communication_vs_batch_size.png', dpi=300)
print("✅ Đã lưu biểu đồ vào file: output_data/master_slave/communication_vs_batch_size.png")
plt.show()