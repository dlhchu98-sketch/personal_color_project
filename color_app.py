import streamlit as st
from PIL import Image
import numpy as np
import matplotlib.colors as mcolors

# CẤU HÌNH TRANG

st.set_page_config("Personal Color", layout="wide")
st.title("Personal Color ")


# BẢNG MÀU

SEASON_PALETTE = {
    "Spring": [
        "#E63F31", "#E75E75", "#EC6B7B", "#F47C90", "#FCCCBE",
        "#834C72", "#A856A0", "#B75697", "#D67CAA", "#D59CC3",
        "#FFE656", "#FFF2A1", "#FEFACF", "#96BE5C", "#C0D25F"
    ],
    "Summer": [
        "#3A3067", "#7A51A1", "#8A65AC", "#9891C5",
        "#BFBEBC", "#D8D5D3",
        "#BC3160", "#D6305D", "#E92778", "#F166A7",
        "#529DD4", "#ACD1EB",
        "#0AB592", "#80C2A4", "#A5D8B8"
    ],
    "Autumn": [
        "#245A76", "#0A8281", "#16A48D",
        "#114421", "#176031", "#537F3A",
        "#9A8F35", "#F3C01A", "#F7E578",
        "#5A3518", "#924D26",
        "#CCBC84", "#E9D3A0",
        "#B92027", "#EF3E28"
    ],
    "Winter": [
        "#060905", "#384450", "#D7D3D2", "#E6E2DF",
        "#161B42", "#262973", "#414AA1", "#258DC9",
        "#047C49", "#1FAA6D",
        "#F5EA16", "#F9EF77", "#FCF9C4",
        "#AB1F67", "#EE1C24"
    ]
}


# RGB → HSV

def hex_to_hsv(hex_color):
    rgb = np.array(mcolors.to_rgb(hex_color))
    return mcolors.rgb_to_hsv(rgb)


# LẤY HSV VÙNG DA (GIỮA MẶT)

def extract_skin_hsv(image):
    img = np.array(image.convert("RGB")) / 255.0
    h, w, _ = img.shape

    # vùng trung tâm khuôn mặt
    y1, y2 = int(h * 0.3), int(h * 0.6)
    x1, x2 = int(w * 0.35), int(w * 0.65)

    region = img[y1:y2, x1:x2]
    hsv = mcolors.rgb_to_hsv(region)

    # Hue (0–360), S, V
    return hsv[..., 0].mean() * 360, hsv[..., 1].mean(), hsv[..., 2].mean()


# XÁC ĐỊNH MÙA (RULE-BASED)

def detect_season(hsv):
    h, s, v = hsv

    if 15 <= h <= 50:
        return "Spring" if v > 0.6 else "Autumn"
    else:
        return "Summer" if s < 0.5 else "Winter"


# CHỌN 4 MÀU THEO VAI TRÒ

def pick_4_colors(colors):
    hsv_colors = [(c, hex_to_hsv(c)) for c in colors]

    # Primary: bão hòa cao
    primary = max(hsv_colors, key=lambda x: x[1][1])[0]

    # Neutral: bão hòa thấp
    neutral = min(hsv_colors, key=lambda x: x[1][1])[0]

    remain = [c for c in colors if c not in [primary, neutral]]
    accent1 = remain[0]
    accent2 = remain[1]

    return {
        "Primary": primary,
        "Accent 1": accent1,
        "Accent 2": accent2,
        "Neutral": neutral
    }


# UI

uploaded = st.file_uploader("📸 Tải ảnh khuôn mặt", type=["jpg", "png", "jpeg"])

if uploaded:
    image = Image.open(uploaded)
    st.image(image, width=280)

    skin_hsv = extract_skin_hsv(image)
    season = detect_season(skin_hsv)

    st.subheader(f"🌿 Personal Color của bạn: **{season}**")

    palette = pick_4_colors(SEASON_PALETTE[season])
    cols = st.columns(4)

    for col, (role, color) in zip(cols, palette.items()):
        with col:
            st.markdown(
                f"""
                <div style="
                    background:{color};
                    height:150px;
                    border-radius:18px;
                    box-shadow:0 8px 18px rgba(0,0,0,0.15)">
                </div>
                """,
                unsafe_allow_html=True
            )
            st.markdown(f"**{role}**")
            st.caption(color)
else:
    st.info("⬆️ Upload ảnh để hệ thống phân tích tone da")
