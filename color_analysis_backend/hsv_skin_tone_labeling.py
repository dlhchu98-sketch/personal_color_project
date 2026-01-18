import os
import cv2
import csv
import numpy as np
from collections import defaultdict

raw_root = r"D:\NMAI\data\raw"
processed_dir = r"D:\NMAI\data\processed"
csv_file = os.path.join(processed_dir, "all_data.csv")

os.makedirs(processed_dir, exist_ok=True)

SEASONS = ["spring", "summer", "autumn", "winter"]

# kiểm tra thư mục raw
print("RAW ROOT:", raw_root)
print("Tồn tại raw_root?", os.path.exists(raw_root))
print("Thư mục con trong raw_root:", os.listdir(raw_root))

def collect_images(root):
    """
    Duyệt tất cả ảnh trong root và các thư mục con.
    Chỉ lấy các file .jpg, .jpeg, .png
    """
    image_paths = []
    for root_dir, _, files in os.walk(root):
        for f in files:
            ext = os.path.splitext(f)[1].lower()
            if ext in [".jpg", ".jpeg", ".png"]:
                image_paths.append(os.path.join(root_dir, f))
    return image_paths

def estimate_skin_tone_hsv(img):
    """
    Gán nhãn mùa dựa trên vùng trung tâm của ảnh (HSV)
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, w, _ = hsv.shape
    center = hsv[h//3:2*h//3, w//3:2*w//3]

    H = np.mean(center[:, :, 0])
    S = np.mean(center[:, :, 1])
    V = np.mean(center[:, :, 2])

    if V > 160 and S > 80:
        return "spring"
    elif V > 160 and S <= 80:
        return "summer"
    elif V <= 160 and S > 80:
        return "autumn"
    else:
        return "winter"

image_paths = collect_images(raw_root)
print(f"Tổng số ảnh tìm thấy: {len(image_paths)}")

if len(image_paths) == 0:
    print("Không tìm thấy ảnh, kiểm tra lại thư mục raw")
    exit()

print("Ví dụ 5 ảnh đầu:")
for p in image_paths[:5]:
    print(" ", p)

counters = defaultdict(int)
rows = []

for img_path in image_paths:
    img = cv2.imread(img_path)
    if img is None:
        continue

    label = estimate_skin_tone_hsv(img)
    counters[label] += 1

    new_name = f"{label}_{counters[label]:04d}.jpg"
    label_dir = os.path.join(processed_dir, label)
    os.makedirs(label_dir, exist_ok=True)

    save_path = os.path.join(label_dir, new_name)
    cv2.imwrite(save_path, img)

    rows.append([save_path, label])

with open(csv_file, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["image_path", "season"])
    writer.writerows(rows)

print("Xong rồi")
print(f"Tổng ảnh xử lý: {len(rows)}")
print("Phân bố theo mùa:")
for s in SEASONS:
    print(f"  {s}: {counters[s]}")

print("CSV lưu ở:", csv_file)
