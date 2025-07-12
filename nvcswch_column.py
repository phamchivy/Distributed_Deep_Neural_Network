import matplotlib.pyplot as plt

def read_avg_nvcswch(filepath):
    nvcswch_values = []
    with open(filepath, 'r') as f:
        lines = f.readlines()
        for line in lines:
            parts = line.strip().split()
            if len(parts) >= 6 and parts[0][0].isdigit():
                try:
                    nvcswch = float(parts[4])
                    nvcswch_values.append(nvcswch)
                except ValueError:
                    pass
    return sum(nvcswch_values) / len(nvcswch_values) if nvcswch_values else 0.0

# Đường dẫn tới hai file log
file1 = 'input_data/single_container/1_cpu_1_thread.txt'  # Ít switching
file2 = 'input_data/single_container/1_cpu_4_thread.txt'  # Nhiều switching

# Tính trung bình
avg1 = read_avg_nvcswch(file1)
avg2 = read_avg_nvcswch(file2)

# Dữ liệu biểu đồ
labels = ['1 core 1 thread', '1 core 4 thread']
values = [avg1, avg2]
colors = ['skyblue', 'salmon']

# Vẽ biểu đồ
plt.figure(figsize=(8, 6))
bars = plt.bar(labels, values, color=colors, width=0.4)
plt.yscale('log')

# Hiển thị giá trị trên mỗi cột
for bar in bars:
    yval = bar.get_height()
    offset = yval * 0.05 if yval > 10 else 5
    plt.text(bar.get_x() + bar.get_width() / 2, yval + offset,
             f'{yval:.2f}', ha='center', va='bottom', fontsize=12,
             bbox=dict(facecolor='white', edgecolor='none', pad=0.2))

# Tiêu đề và nhãn
plt.title('Comparison of Average nvcswch/s Between Two Context Switching States', fontsize=14)
plt.ylabel('Average nvcswch/s Value (log scale)', fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.6)

# Lưu biểu đồ thành file PNG
plt.tight_layout()
plt.savefig('output_data/single_container/nvcswch_comparison.png', dpi=300)
plt.show()
