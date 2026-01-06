import pandas as pd
import numpy as np
import cv2
import os

# đường dẫn CSV đầu vào và thư mục lưu CSV HSV
CSV_INPUT = "data/processed/all_data.csv"  
FEATURE_DIR = "features_csv"
os.makedirs(FEATURE_DIR, exist_ok=True)

# đọc CSV gốc
df = pd.read_csv(CSV_INPUT)

# danh sách lưu toàn bộ feature HSV
all_rows_hsv = []

# lưu một số ảnh mẫu để hiển thị trực quan
sample_images = []
MAX_DISPLAY = 3  # hiển thị tối đa 3 ảnh để kiểm tra

# duyệt từng ảnh trong CSV
for _, row in df.iterrows():
    try:
        img_path = row["image_path"]
        season_value = row["season"] if "season" in df.columns else "unknown"

        # đọc ảnh
        img = cv2.imread(img_path)
        if img is None:
            continue

        # resize nhỏ để trích HSV ổn định
        img_small = cv2.resize(img, (32, 32))

        # chuyển ảnh sang không gian màu HSV
        img_hsv = cv2.cvtColor(img_small, cv2.COLOR_BGR2HSV)
        h, w, _ = img_hsv.shape

        # trích màu trung bình từng vùng: skin, eyes, brows, hair
        skin_hsv = img_hsv[int(h*0.3):int(h*0.7), int(w*0.3):int(w*0.7)].mean(axis=(0,1)).astype(int)
        eyes_hsv = img_hsv[int(h*0.25):int(h*0.45), int(w*0.25):int(w*0.75)].mean(axis=(0,1)).astype(int)
        brows_hsv = img_hsv[int(h*0.20):int(h*0.30), int(w*0.25):int(w*0.75)].mean(axis=(0,1)).astype(int)
        hair_hsv = img_hsv[0:int(h*0.20), :].mean(axis=(0,1)).astype(int)

        # lưu feature vào danh sách tổng hợp
        all_rows_hsv.append([
            img_path, season_value,
            *skin_hsv, *eyes_hsv, *brows_hsv, *hair_hsv
        ])

        # lưu một số ảnh mẫu để hiển thị trực quan
        if len(sample_images) < MAX_DISPLAY:
            sample_images.append((img.copy(), skin_hsv, eyes_hsv, brows_hsv, hair_hsv))

    except Exception:
        # bỏ qua ảnh lỗi để không làm gián đoạn toàn bộ quá trình
        continue

# tên cột cho CSV HSV
columns_hsv = [
    "image_path", "season",
    "skin_H", "skin_S", "skin_V",
    "eyes_H", "eyes_S", "eyes_V",
    "brows_H", "brows_S", "brows_V",
    "hair_H", "hair_S", "hair_V"
]

# lưu toàn bộ feature HSV ra CSV
pd.DataFrame(all_rows_hsv, columns=columns_hsv).to_csv(
    os.path.join(FEATURE_DIR, "all_hsv_features.csv"),
    index=False
)
print(f"HSV features saved in: {os.path.join(FEATURE_DIR, 'all_hsv_features.csv')}")

# hiển thị một số ảnh mẫu với khối màu HSV 4 vùng
for img, skin_hsv, eyes_hsv, brows_hsv, hair_hsv in sample_images:
    # tạo khối màu 4 ô HSV
    color_block = np.zeros((32, 128, 3), dtype=np.uint8)
    color_block[:, 0:32, :] = skin_hsv
    color_block[:, 32:64, :] = eyes_hsv
    color_block[:, 64:96, :] = brows_hsv
    color_block[:, 96:128, :] = hair_hsv

    # chuyển HSV -> BGR để hiển thị bằng OpenCV
    color_block_bgr = cv2.cvtColor(color_block, cv2.COLOR_HSV2BGR)

    # hiển thị ảnh và khối màu
    cv2.imshow("HSV Color Features", color_block_bgr)
    cv2.imshow("Image", cv2.resize(img, (128, 128)))

    if cv2.waitKey(500) == 27:  # 500ms mỗi ảnh, nhấn ESC để thoát
        break

cv2.destroyAllWindows()
