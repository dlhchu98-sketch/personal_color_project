import pandas as pd
import numpy as np
import cv2
import os

# cấu hình đường dẫn CSV đầu vào và thư mục lưu feature / preview
CSV_INPUT = "data/processed/all_data.csv"  # CSV gốc chứa đường dẫn ảnh và nhãn season
FEATURE_DIR = "features_csv"               # thư mục lưu CSV đặc trưng màu
PREVIEW_DIR = "previews"                   # thư mục lưu hình preview bảng màu
os.makedirs(FEATURE_DIR, exist_ok=True)
os.makedirs(PREVIEW_DIR, exist_ok=True)

# đọc CSV gốc
df = pd.read_csv(CSV_INPUT)

# danh sách lưu từng feature cho các vùng khác nhau
all_rows = []     # lưu tất cả feature
skin_rows = []    # feature vùng da
eyes_rows = []    # feature vùng mắt
brows_rows = []   # feature vùng lông mày
hair_rows = []    # feature vùng tóc

# hàm tạo hình preview 4 ô màu: skin, eyes, brows, hair
def create_color_block(skin, eyes, brows, hair, width=256, height=64):
    block = np.zeros((height, width, 3), dtype=np.uint8)
    block[:, 0:64, :] = skin
    block[:, 64:128, :] = eyes
    block[:, 128:192, :] = brows
    block[:, 192:256, :] = hair
    return block

# giới hạn số ảnh hiển thị để kiểm tra trực quan
display_count = 0
MAX_DISPLAY = 5

# duyệt từng ảnh trong CSV
for _, row in df.iterrows():
    img = cv2.imread(row["image_path"])
    if img is None:
        continue

    # chuyển từ BGR sang RGB để dễ thao tác màu
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w, _ = img_rgb.shape

# Các vùng màu (skin, eyes, brows, hair) được xác định bằng cách cắt ảnh
# theo tỷ lệ hình học phổ biến của khuôn mặt trong ảnh chân dung chính diện.
# Phương pháp heuristic này nhằm trích xuất màu trung bình đại diện,
# không yêu cầu phát hiện landmark để đơn giản hóa tiền xử lý.

    skin = img_rgb[int(h*0.3):int(h*0.7), int(w*0.3):int(w*0.7)].mean(axis=(0,1)).astype(int)
    eyes = img_rgb[int(h*0.25):int(h*0.45), int(w*0.25):int(w*0.75)].mean(axis=(0,1)).astype(int)
    brows = img_rgb[int(h*0.20):int(h*0.30), int(w*0.25):int(w*0.75)].mean(axis=(0,1)).astype(int)
    hair = img_rgb[0:int(h*0.20), :].mean(axis=(0,1)).astype(int)

    # lấy giá trị season, nếu không có cột season thì mặc định là "unknown"
    season_value = row["season"] if "season" in df.columns else "unknown"

    # lưu feature vào các danh sách tương ứng
    all_rows.append([row["image_path"], season_value, *skin, *eyes, *brows, *hair])
    skin_rows.append([row["image_path"], season_value, *skin])
    eyes_rows.append([row["image_path"], season_value, *eyes])
    brows_rows.append([row["image_path"], season_value, *brows])
    hair_rows.append([row["image_path"], season_value, *hair])

    # tạo hình preview 4 ô màu và lưu ra thư mục
    color_block = create_color_block(skin, eyes, brows, hair)
    preview_name = os.path.splitext(os.path.basename(row["image_path"]))[0] + "_preview.png"
    cv2.imwrite(os.path.join(PREVIEW_DIR, preview_name), cv2.cvtColor(color_block, cv2.COLOR_RGB2BGR))

    # hiển thị trực quan tối đa MAX_DISPLAY ảnh
    if display_count < MAX_DISPLAY:
        cv2.imshow("Image", cv2.resize(img_rgb, (256,256)))
        cv2.imshow("Color Features", color_block)
        key = cv2.waitKey(500)  # mỗi ảnh hiển thị 500ms
        if key == 27:
            break
        display_count += 1

cv2.destroyAllWindows()

# lưu CSV từng vùng màu riêng lẻ
columns_rgb = ["image_path", "season", "R", "G", "B"]
pd.DataFrame(skin_rows, columns=columns_rgb).to_csv(os.path.join(FEATURE_DIR, "skin.csv"), index=False)
pd.DataFrame(eyes_rows, columns=columns_rgb).to_csv(os.path.join(FEATURE_DIR, "eyes.csv"), index=False)
pd.DataFrame(brows_rows, columns=columns_rgb).to_csv(os.path.join(FEATURE_DIR, "brows.csv"), index=False)
pd.DataFrame(hair_rows, columns=columns_rgb).to_csv(os.path.join(FEATURE_DIR, "hair.csv"), index=False)

# lưu CSV tổng hợp tất cả feature
columns_all = ["image_path", "season",
               "skin_R", "skin_G", "skin_B",
               "eyes_R", "eyes_G", "eyes_B",
               "brows_R", "brows_G", "brows_B",
               "hair_R", "hair_G", "hair_B"]
pd.DataFrame(all_rows, columns=columns_all).to_csv(os.path.join(FEATURE_DIR, "all_features.csv"), index=False)

print("Feature extraction and CSV export DONE")
print(f"Previews saved in: {PREVIEW_DIR}")
