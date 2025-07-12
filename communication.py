import re
import csv

def extract_syncs(lines, role_label, img_limit):
    result = []
    for line in lines:
        match = re.search(rf'{role_label}.*?total sync took ([\d.]+) seconds at img (\d+)', line, re.IGNORECASE)
        if match:
            time = float(match.group(1))
            img = int(match.group(2))
            if (role_label == 'master' and img < img_limit) or (role_label == 'slaver' and img >= img_limit):
                result.append((img, time))
    # Loại bỏ log trùng theo img
    seen = {}
    for img, time in result:
        if img not in seen:
            seen[img] = time
    return sorted(seen.items())

def sync_max_analysis(filename, save_to_file=None):
    with open(filename, 'r') as f:
        lines = f.readlines()

    masters = extract_syncs(lines, 'master', 30000)
    slavers = extract_syncs(lines, 'slaver', 30000)

    min_len = min(len(masters), len(slavers))
    total = 0.0

    headers = ['Update', 'Master(img)', 'Master Time', 'Slaver(img)', 'Slaver Time', 'Max Time']
    rows = []

    print(f"{'Update':>6} | {'Master(img)':>12} | {'Time':>8} | {'Slaver(img)':>12} | {'Time':>8} | {'Max':>8}")
    print("-" * 65)
    for i in range(min_len):
        m_img, m_time = masters[i]
        s_img, s_time = slavers[i]
        max_time = max(m_time, s_time)
        print(f"{i:6} | {m_img:12} | {m_time:8.5f} | {s_img:12} | {s_time:8.5f} | {max_time:8.5f}")
        total += max_time
        rows.append([i, m_img, m_time, s_img, s_time, max_time])

    print(f"\nTổng thời gian đại diện cho {min_len} lần cập nhật: {total:.6f} giây")

    # ✅ Ghi vào file nếu có yêu cầu
    if save_to_file:
        with open(save_to_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(headers)
            writer.writerows(rows)
        print(f"\n✅ Đã lưu bảng kết quả vào: {save_to_file}")

# Chạy thử
sync_max_analysis('input_data/master_slave/batch-1000-1-cpu.txt', save_to_file='input_data/master_slave/sync_report.csv')
