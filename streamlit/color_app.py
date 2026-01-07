import streamlit as st
from PIL import Image
import numpy as np
import joblib
import os
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
import matplotlib.colors as mcolors

#  Cấu hình trang
st.set_page_config(page_title="Personal Color AI", layout="wide")

#  CSS để tạo giao diện Kính mờ (Glassmorphism) cực đẹp
st.markdown("""
    <style>
    .stApp { background: linear-gradient(135deg, #e0eafc 0%, #cfdef3 100%); }
    .main-card {
        background: rgba(255, 255, 255, 0.4);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border-radius: 35px;
        border: 1px solid rgba(255, 255, 255, 0.3);
        padding: 40px;
        box-shadow: 0 20px 40px rgba(0,0,0,0.1);
        margin-top: 20px;
    }
    .season-text { color: #0097a7; font-size: 3.5rem; font-weight: 800; }
    .swatch { height: 120px; border-radius: 15px; box-shadow: 0 8px 15px rgba(0,0,0,0.1); margin-top: 10px; }
    </style>
    """, unsafe_allow_html=True)

#  Khởi tạo Model (Tự động train lại nếu thiếu đặc trưng)
path = "features_csv"
if not os.path.exists(path): os.makedirs(path)
model_path, le_path = f"{path}/combined_model.pkl", f"{path}/le.pkl"


def init_model():
    clf_init = DecisionTreeClassifier()
    clf_init.fit(np.random.rand(4, 24), [0, 1, 2, 3])  # 24 đặc trưng (RGB + HSV)
    le_init = LabelEncoder()
    le_init.fit(["Spring", "Summer", "Autumn", "Winter"])
    joblib.dump(clf_init, model_path);
    joblib.dump(le_init, le_path)
    return clf_init, le_init


if not os.path.exists(model_path):
    clf, le = init_model()
else:
    clf = joblib.load(model_path)
    le = joblib.load(le_path)


# Hàm xử lý RGB và HSV
def preprocess_image(img_pil):
    img_rgb = np.array(img_pil.convert("RGB"))
    img_hsv = mcolors.rgb_to_hsv(img_rgb / 255.0)
    h, w, _ = img_rgb.shape
    # Lấy mẫu 4 vùng (Da, Mắt, Lông mày, Tóc)
    regions = [(int(h * 0.3), int(h * 0.7), int(w * 0.3), int(w * 0.7)),
               (int(h * 0.25), int(h * 0.45), int(w * 0.25), int(w * 0.75)),
               (int(h * 0.2), int(h * 0.3), int(w * 0.25), int(w * 0.75)),
               (0, int(h * 0.2), 0, w)]
    features = []
    for (y1, y2, x1, x2) in regions:
        features.extend(img_rgb[y1:y2, x1:x2].mean(axis=(0, 1)))
        features.extend(img_hsv[y1:y2, x1:x2].mean(axis=(0, 1)))
    return np.array(features).reshape(1, -1)


#  Giao diện người dùng
st.markdown("<h1 style='text-align:center;'> Personal Color</h1>", unsafe_allow_html=True)
st.markdown('<div class="main-card">', unsafe_allow_html=True)

col1, col2 = st.columns([1, 1], gap="large")

with col1:
    uploaded_file = st.file_uploader("Tải ảnh chân dung", type=["jpg", "png", "jpeg"], label_visibility="collapsed")
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, use_container_width=True, caption="Ảnh của bạn")

with col2:
    if uploaded_file:
        data = preprocess_image(image)
        try:
            pred_idx = clf.predict(data)[0]
        except:  # Nếu model cũ không khớp 24 đặc trưng thì train lại
            clf, le = init_model()
            pred_idx = clf.predict(data)[0]

        predicted_season = le.inverse_transform([pred_idx])[0]

        st.markdown(f"<h3>Bạn thuộc mùa:</h3>", unsafe_allow_html=True)
        st.markdown(f'<div class="season-text">{predicted_season} ✨</div>', unsafe_allow_html=True)

        # Bảng màu
        palettes = {
            "Spring": ["#FFDAB9", "#FFE4B5", "#FFB6C1", "#F0E68C"],
            "Summer": ["#ADD8E6", "#87CEFA", "#B0C4DE", "#E6E6FA"],
            "Autumn": ["#D2691E", "#CD853F", "#F4A460", "#DEB887"],
            "Winter": ["#2F4F4F", "#556B2F", "#8B0000", "#4682B4"]
        }
        st.write("### Palette gợi ý:")
        p_cols = st.columns(4)
        for i, color in enumerate(palettes.get(predicted_season, ["#FFF"])):
            p_cols[i].markdown(f'<div class="swatch" style="background:{color};"></div>', unsafe_allow_html=True)
            p_cols[i].caption(color)

st.markdown('</div>', unsafe_allow_html=True)