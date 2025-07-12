import pandas as pd
import matplotlib.pyplot as plt
import os

# Tạo thư mục output nếu chưa tồn tại
output_dir = 'output_data/master_slave'
os.makedirs(output_dir, exist_ok=True)

# Đọc dữ liệu CSV
df = pd.read_csv('input_data/master_slave/master_slaver.csv')

# Vẽ bảng dữ liệu
fig, ax = plt.subplots(figsize=(8, len(df)*0.6 + 1))  # Chiều cao động theo số dòng
ax.axis('off')  # Tắt trục

# Thêm tiêu đề bảng
ax.set_title('Bảng dữ liệu master slave với số cpu', fontsize=16, weight='bold', pad=20)

# Tạo bảng
table_data = df.values
column_labels = df.columns
table = ax.table(cellText=table_data, colLabels=column_labels, loc='center', cellLoc='center')

# Tùy chỉnh bảng
table.auto_set_font_size(False)
table.set_fontsize(12)

# Giãn bảng cho dễ đọc
table.scale(1.2, 1.5)  # (x_scale, y_scale): tăng chiều ngang và dọc của ô

# Lưu bảng
table_path = os.path.join(output_dir, 'master_slave_table.png')
plt.savefig(table_path, dpi=300, bbox_inches='tight')
plt.close()

print(f"✅ Đã lưu bảng dữ liệu tại: {table_path}")