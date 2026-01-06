import pandas as pd
import os
from sklearn.model_selection import train_test_split

# đường dẫn thư mục đã xử lý
PROCESSED_DIR = r"D:\NMAI\data\processed"

# file CSV gốc chưa cân bằng và đã cân bằng
CSV_UNBALANCED = os.path.join(PROCESSED_DIR, "all_data.csv")
CSV_BALANCED = os.path.join(PROCESSED_DIR, "balanced_data.csv")

# file CSV đầu ra cho train / test
OUT_FILES = {
    "train_unbalanced": os.path.join(PROCESSED_DIR, "train_unbalanced.csv"),
    "test_unbalanced":  os.path.join(PROCESSED_DIR, "test_unbalanced.csv"),
    "train_balanced":   os.path.join(PROCESSED_DIR, "train_balanced.csv"),
    "test_balanced":    os.path.join(PROCESSED_DIR, "test_balanced.csv"),
}

TEST_SIZE = 0.2      # 20% dữ liệu dùng làm test
RANDOM_STATE = 42    # để kết quả split ổn định

# hàm tách dữ liệu train / test và lưu ra CSV
def split_and_save(csv_path, train_out, test_out):
    df = pd.read_csv(csv_path)

    X = df["image_path"]
    y = df["label"]

    # tách dữ liệu theo tỷ lệ, giữ phân bố nhãn (stratify)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )

    # tạo DataFrame cho train và test
    train_df = pd.DataFrame({
        "image_path": X_train,
        "label": y_train
    })

    test_df = pd.DataFrame({
        "image_path": X_test,
        "label": y_test
    })

    # lưu ra CSV
    train_df.to_csv(train_out, index=False)
    test_df.to_csv(test_out, index=False)

    # in thông báo
    print(f"\nSplit xong: {os.path.basename(csv_path)}")
    print(f"  Train: {len(train_df)} ảnh")
    print(f"  Test : {len(test_df)} ảnh")

# tách dữ liệu chưa cân bằng
print("Tách dữ liệu unbalanced")
split_and_save(
    CSV_UNBALANCED,
    OUT_FILES["train_unbalanced"],
    OUT_FILES["test_unbalanced"]
)

# tách dữ liệu đã cân bằng
print("\nTách dữ liệu balanced")
split_and_save(
    CSV_BALANCED,
    OUT_FILES["train_balanced"],
    OUT_FILES["test_balanced"]
)

print("\nHoàn tất tách train / test")
