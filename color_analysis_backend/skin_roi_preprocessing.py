import os
import cv2
import pandas as pd
import numpy as np
import mediapipe as mp
from tqdm import tqdm

# cấu hình thư mục và các thông số
PROCESSED_DIR = r"D:\NMAI\data\processed"
PRE_DIR = os.path.join(PROCESSED_DIR, "preprocessed")

IMG_SIZE = 224   # kích thước chuẩn sau khi resize

# các file CSV train/test unbalanced/balanced
CSV_FILES = {
    "train_unbalanced": "train_unbalanced.csv",
    "test_unbalanced": "test_unbalanced.csv",
    "train_balanced": "train_balanced.csv",
    "test_balanced": "test_balanced.csv"
}

os.makedirs(PRE_DIR, exist_ok=True)

# khởi tạo MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=True
)

# các index landmark để cắt ROI
FOREHEAD = [10, 338, 297, 332, 284, 251, 389, 356]
LEFT_CHEEK = [234, 93, 132, 58, 172, 136, 150]
RIGHT_CHEEK = [454, 323, 361, 288, 397, 365, 379]

# hàm cắt ROI từ landmark
def crop_roi(img, landmarks, indices):
    h, w, _ = img.shape
    points = []
    for idx in indices:
        lm = landmarks[idx]
        x = int(lm.x * w)
        y = int(lm.y * h)
        points.append((x, y))

    x, y, bw, bh = cv2.boundingRect(np.array(points))
    roi = img[y:y+bh, x:x+bw]

    if roi.size == 0:
        return None
    return roi

# chuẩn hóa ánh sáng ảnh
def normalize_lighting(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    img = cv2.merge((l, a, b))
    return cv2.cvtColor(img, cv2.COLOR_LAB2BGR)

# tiền xử lý ảnh: blur, ánh sáng, resize, normalize pixel
def preprocess_image(img):
    img = cv2.GaussianBlur(img, (5, 5), 0)   # giảm nhiễu
    img = normalize_lighting(img)            # cân bằng ánh sáng
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))  # resize chuẩn
    img = img.astype("float32") / 255.0         # normalize pixel
    return img

# xử lý từng split: train/test unbalanced/balanced
for split_name, csv_name in CSV_FILES.items():
    print(f"Preprocessing: {split_name}")

    csv_path = os.path.join(PROCESSED_DIR, csv_name)
    df = pd.read_csv(csv_path)

    out_img_dir = os.path.join(PRE_DIR, split_name)
    os.makedirs(out_img_dir, exist_ok=True)

    out_csv = os.path.join(PROCESSED_DIR, f"{split_name}_pre.csv")
    rows = []

    # duyệt từng ảnh
    for _, row in tqdm(df.iterrows(), total=len(df)):
        img_path = row["image_path"]
        label = row["label"]

        img = cv2.imread(img_path)
        if img is None:
            continue

        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)

        if not results.multi_face_landmarks:
            continue

        landmarks = results.multi_face_landmarks[0].landmark

        # cắt ROI trán và hai má
        forehead = crop_roi(img, landmarks, FOREHEAD)
        left_cheek = crop_roi(img, landmarks, LEFT_CHEEK)
        right_cheek = crop_roi(img, landmarks, RIGHT_CHEEK)

        if forehead is None or left_cheek is None or right_cheek is None:
            continue

        # gộp ROI theo chiều dọc
        face_roi = cv2.vconcat([
            cv2.resize(forehead, (IMG_SIZE, IMG_SIZE)),
            cv2.resize(left_cheek, (IMG_SIZE, IMG_SIZE)),
            cv2.resize(right_cheek, (IMG_SIZE, IMG_SIZE))
        ])

        # tiền xử lý ROI
        face_roi = preprocess_image(face_roi)

        # lưu ảnh đã tiền xử lý
        filename = os.path.basename(img_path)
        save_path = os.path.join(out_img_dir, filename)
        cv2.imwrite(save_path, (face_roi * 255).astype("uint8"))

        rows.append([save_path, label])

    # lưu CSV mới sau khi preprocess
    out_df = pd.DataFrame(rows, columns=["image_path", "label"])
    out_df.to_csv(out_csv, index=False)

    print(f"Lưu xong: {out_csv} ({len(out_df)} ảnh)")

print("Hoàn tất preprocessing tất cả split")
