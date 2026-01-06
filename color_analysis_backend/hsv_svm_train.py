import os
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# thư mục chứa feature HSV đã trích xuất
FEATURE_DIR = r"D:\NMAI\data\features\hsv"

# file dữ liệu train/test cho bộ dữ liệu balanced/unbalanced
SETS = {
    "unbalanced": {
        "X_train": "train_unbalanced_X.npy",
        "y_train": "train_unbalanced_y.npy",
        "X_test": "test_unbalanced_X.npy",
        "y_test": "test_unbalanced_y.npy",
    },
    "balanced": {
        "X_train": "train_balanced_X.npy",
        "y_train": "train_balanced_y.npy",
        "X_test": "test_balanced_X.npy",
        "y_test": "test_balanced_y.npy",
    }
}

# hàm train SVM trên feature HSV và đánh giá kết quả
def train_and_evaluate(name, paths):
    print(f"\nTRAIN HSV - {name.upper()}")

    # load dữ liệu train/test từ file .npy
    X_train = np.load(os.path.join(FEATURE_DIR, paths["X_train"]))
    y_train = np.load(os.path.join(FEATURE_DIR, paths["y_train"]))
    X_test  = np.load(os.path.join(FEATURE_DIR, paths["X_test"]))
    y_test  = np.load(os.path.join(FEATURE_DIR, paths["y_test"]))

    # pipeline gồm chuẩn hóa dữ liệu và SVM kernel RBF
    model = Pipeline([
        ("scaler", StandardScaler()),  # chuẩn hóa dữ liệu (mean=0, std=1)
        ("svm", SVC(kernel="rbf", C=10, gamma="scale"))  # SVM kernel RBF
    ])

    # huấn luyện model
    model.fit(X_train, y_train)

    # dự đoán trên tập test
    y_pred = model.predict(X_test)

    # đánh giá accuracy và hiển thị báo cáo phân loại
    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {acc:.4f}")
    print("Classification report:")
    print(classification_report(y_test, y_pred, digits=4))

# train và đánh giá cho cả unbalanced và balanced
for name, paths in SETS.items():
    train_and_evaluate(name, paths)

print("\nHSV training DONE")
