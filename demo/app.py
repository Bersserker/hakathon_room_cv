import streamlit as st
from PIL import Image
import numpy as np

from models import predict_one
from explain import generate_gradcam

# Настройка страницы
st.set_page_config(page_title="Model Demo", layout="wide")
st.title("Image Classification Demo")

# Sidebar
st.sidebar.header("Model Info")
st.sidebar.write("Version:", "v1.2.3")

#Загрузка изображения
uploaded_file = st.file_uploader("Upload image", type=["jpg", "png", "jpeg"])

# Функция отображения результатов
def show_results(image):
    # Инференс
    result = predict_one(image)
    
    st.subheader("Top-3 Predictions")
    # Top-3 predictions
    for label, conf in result["topk"]:
        st.write(f"**{label}**: {conf:.3f}")

    st.subheader("Grad-CAM")
    # Grad-CAM
    cam = generate_gradcam(image)
    st.image(cam, use_container_width=True)
    # Дополнительные сигналы
    if "logits" in result:
        st.subheader("Extra signals")
        st.write("Logits:", result["logits"])

# Обработка пользовательского изображения
if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    # UI отображение input
    st.image(image, caption="Input", use_container_width=True)
    # Safe inference
    try:
        show_results(image)
    except Exception as e:
        st.error(f"Inference failed: {str(e)}")

st.divider()

# Examples
st.subheader("Examples")
