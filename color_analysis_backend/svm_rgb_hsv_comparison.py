import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)

# thư mục chứa các feature đã trích xuất (RGB / HSV)
BASE_DIR = r"D:\NMAI\data\features"

# cấu hình các bộ thử nghiệm: RGB/HSV và balanced/unbalanced
CONFIGS = [
    ("RGB", "rgb", "unbalanced"),
    ("RGB", "rgb", "balanced"),
    ("HSV", "hsv", "unbalanced"),
    ("HSV", "hsv", "balanced"),
]

# nhãn các mùa để hiển thị confusion matrix
LABEL_NAMES = ["Spring", "Summer", "Autumn", "Winter"]

# lưu kết quả đánh giá
results = []

# hàm train SVM và đánh giá accuracy, macro F1, confusion matrix
def train_eval(feature_type, dataset_type):
    feature_dir = os.path.join(BASE_DIR, feature_type)

    # load dữ liệu train/test
    X_train = np.load(os.path.join(feature_dir, f"train_{dataset_type}_X.npy"))
    y_train = np.load(os.path.join(feature_dir, f"train_{dataset_type}_y.npy"))
    X_test  = np.load(os.path.join(feature_dir, f"test_{dataset_type}_X.npy"))
    y_test  = np.load(os.path.join(feature_dir, f"test_{dataset_type}_y.npy"))

    # pipeline gồm chuẩn hóa dữ liệu + SVM kernel RBF
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("svm", SVC(kernel="rbf", C=10, gamma="scale"))
    ])

    # huấn luyện model
    model.fit(X_train, y_train)

    # dự đoán trên tập test
    y_pred = model.predict(X_test)

    # tính accuracy, macro F1 và confusion matrix
    acc = accuracy_score(y_test, y_pred)
    f1  = f1_score(y_test, y_pred, average="macro")
    cm  = confusion_matrix(y_test, y_pred)

    return acc, f1, cm

# chạy thử nghiệm cho tất cả cấu hình RGB/HSV và balanced/unbalanced
for color_space, feature_type, dataset_type in CONFIGS:
    acc, f1, cm = train_eval(feature_type, dataset_type)

    # lưu kết quả vào bảng so sánh
    results.append({
        "Color Space": color_space,
        "Dataset": dataset_type,
        "Accuracy": round(acc, 4),
        "Macro F1": round(f1, 4)
    })

    # hiển thị confusion matrix
    disp = ConfusionMatrixDisplay(cm, display_labels=LABEL_NAMES)
    disp.plot(cmap="Blues")
    plt.title(f"{color_space} - {dataset_type}")
    plt.show()

# hiển thị bảng tổng hợp kết quả
df = pd.DataFrame(results)
print("\nFINAL COMPARISON")
print(df)
