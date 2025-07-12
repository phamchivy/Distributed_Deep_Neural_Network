import re
import matplotlib.pyplot as plt

# Đọc nội dung file log
with open('input_data/easgd/async-ea-5.txt', 'r') as f:
    lines = f.readlines()

# Regex để lấy thông tin Sync #
pattern = re.compile(r'worker(\d)-1\s+\|\s+\[Worker \d\] Sync #(\d+) completed in ([\d.]+)ms')

# Lưu thông tin từng worker
sync_data = {1: [], 2: []}

# Duyệt từng dòng log
for line in lines:
    match = pattern.search(line)
    if match:
        worker_id = int(match.group(1))
        sync_num = int(match.group(2))
        time_s = float(match.group(3)) / 1000  # Đổi từ ms sang s
        sync_data[worker_id].append((sync_num, time_s))

# Sắp xếp theo sync #
for worker in sync_data:
    sync_data[worker].sort(key=lambda x: x[0])

# Vẽ scatter plot
plt.figure(figsize=(10, 6))

for worker_id, data in sync_data.items():
    x = [d[0] for d in data]
    y = [d[1] for d in data]
    plt.scatter(x, y, label=f'Worker {worker_id}-1')

plt.xlabel('Sync #')
plt.ylabel('Sync Time (s)')
plt.title('Sync Time per Sync # (in seconds)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("output_data/easgd/async_time_scatter.png", dpi=300)
plt.show()