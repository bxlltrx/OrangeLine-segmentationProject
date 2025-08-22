import streamlit as st
from PIL import Image
import cv2
import io
import numpy as np
import torch
import albumentations as albu

# -----------------------------
# Настройки
# -----------------------------
CLASSES = ["фон", "линия"]
INFER_WIDTH = 256
INFER_HEIGHT = 256
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
MODEL_PATH = "models/best_model_new.pt"   # <-- путь к одной модели

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def compute_metrics(pred_mask: np.ndarray, true_mask: np.ndarray, class_index: int = 1):
    """
    Вычисление метрик для бинарной сегментации (по выбранному классу).
    pred_mask, true_mask: 2D numpy arrays (H, W), значения {0,1,...}
    """
    pred = (pred_mask == class_index).astype(np.uint8)
    true = (true_mask == class_index).astype(np.uint8)

    intersection = np.logical_and(pred, true).sum()
    union = np.logical_or(pred, true).sum()

    dice = (2 * intersection) / (pred.sum() + true.sum() + 1e-7)
    iou = intersection / (union + 1e-7)
    acc = (pred == true).mean()

    return {
        "IoU": round(float(iou), 4),
        "Dice": round(float(dice), 4),
        "Accuracy": round(float(acc), 4),
    }

def to_bytes(img: np.ndarray) -> bytes:
    """Конвертирует numpy-изображение в PNG-байты для st.download_button."""
    pil_img = Image.fromarray(img)
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    return buf.getvalue()

# -----------------------------
# Кэш: модель и аугментации
# -----------------------------
@st.cache_resource
def load_model(path: str):
    model = torch.jit.load(path, map_location=DEVICE)
    model.eval()
    return model

@st.cache_data
def get_validation_augmentation():
    return albu.Compose([
        albu.LongestMaxSize(max_size=INFER_HEIGHT, always_apply=True),
        albu.PadIfNeeded(min_height=INFER_HEIGHT, min_width=INFER_WIDTH, always_apply=True),
        albu.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

# -----------------------------
# Инференс
# -----------------------------
def infer_image_with_model(image: np.ndarray, model: torch.nn.Module):
    original_height, original_width, _ = image.shape
    augmentation = get_validation_augmentation()
    augmented = augmentation(image=image)
    image_transformed = augmented["image"]

    x_tensor = (
        torch.from_numpy(image_transformed)
        .to(DEVICE)
        .unsqueeze(0)
        .permute(0, 3, 1, 2)
        .float()
    )

    with torch.no_grad():
        pr = model(x_tensor)

    pr = pr.squeeze().cpu().numpy()          # (C, H, W)
    label_mask = np.argmax(pr, axis=0)       # (H, W)

    # Убираем паддинги от квадратного пайплайна
    if original_height > original_width:
        delta = int(((original_height - original_width) / 2) / original_height * INFER_HEIGHT)
        mask_cropped = label_mask[:, delta: INFER_WIDTH - delta]
    elif original_height < original_width:
        delta = int(((original_width - original_height) / 2) / original_width * INFER_WIDTH)
        mask_cropped = label_mask[delta: INFER_HEIGHT - delta, :]
    else:
        mask_cropped = label_mask

    mask_real = cv2.resize(mask_cropped, (original_width, original_height), interpolation=cv2.INTER_NEAREST)
    return mask_real

# -----------------------------
# Вспомогательные функции UI/обработки
# -----------------------------
def overlay_mask(img_rgb: np.ndarray, mask: np.ndarray, alpha: float = 0.45):
    """Полупрозрачный оверлей предсказанной маски."""
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    # Подсветим класс "линия" (index=1) цветом
    color = (0, 255, 255)  # желтый в BGR
    overlay = img_bgr.copy()
    overlay[mask == 1] = (overlay[mask == 1] * (1 - alpha) + np.array(color) * alpha).astype(np.uint8)
    out = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
    return out

def adjust_hsv(image: np.ndarray, mask: np.ndarray, h_adjust: int, s_adjust: int, v_adjust: int, index: int):
    """Корректировка HSV в области mask == index."""
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV).astype(np.float32)
    h, s, v = cv2.split(hsv)

    m = (mask == index)
    h[m] = np.clip(h[m] + h_adjust, 0, 179)
    s[m] = np.clip(s[m] + s_adjust, 0, 255)
    v[m] = np.clip(v[m] + v_adjust, 0, 255)

    out = cv2.merge([h, s, v]).astype(np.uint8)
    return cv2.cvtColor(out, cv2.COLOR_HSV2RGB)

def to_pil(img: np.ndarray) -> Image.Image:
    return Image.fromarray(img)

# -----------------------------
# UI
# -----------------------------
def main():
    st.set_page_config(page_title="Line Segmentation", page_icon="🧠", layout="wide")

    # Ограничиваем высоту всех изображений высотой окна, тянем на всю ширину
    st.markdown("""
        <style>
        .stImage img {
            width: 100% !important;      /* растягиваем по ширине контейнера */
            height: auto !important;      /* сохраняем пропорции */
            max-height: 85vh !important;  /* не выше 85% высоты экрана */
            object-fit: contain !important;
        }
        </style>
    """, unsafe_allow_html=True)

    st.title("🧠 Line Segmentation")
    st.caption("DeepLabV3+")

    with st.sidebar:
        st.subheader("Параметры")
        st.markdown("**Модель:**")
        st.code(MODEL_PATH, language="text")
        alpha = st.slider("Прозрачность маски", 0.0, 1.0, 0.45, 0.05)

        st.markdown("---")
        st.subheader("HSV-коррекция")
        h_adjust = st.slider("Оттенок (H)", -179, 179, 0)
        s_adjust = st.slider("Насыщенность (S)", -255, 255, 0)
        v_adjust = st.slider("Яркость (V)", -255, 255, 0)
        region = st.selectbox("Область", CLASSES)
        index = CLASSES.index(region)

        st.markdown("---")
        st.info("Загрузите изображение в формате JPG/PNG. Маска подсвечивает класс «линия».", icon="ℹ️")

    st.markdown("---")
    st.subheader("Масштабирование")
    scale_percent = st.slider("Масштаб изображения (%)", 10, 100, 50, 5)

    def resize_img(img: np.ndarray, scale: int) -> np.ndarray:
        """Масштабирование изображения по процентам."""
        h, w = img.shape[:2]
        new_w = int(w * scale / 100)
        new_h = int(h * scale / 100)
        return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Загрузка модели (кэшир.)
    model = load_model(MODEL_PATH)

    # Загрузка изображения
    uploaded = st.file_uploader("Загрузите изображение", type=["jpg", "jpeg", "png"])
    if uploaded is None:
        st.stop()

    image = np.array(Image.open(uploaded).convert("RGB"))

    # Инференс
    mask = infer_image_with_model(image, model)

    # --- ВКЛАДКИ И ОТРИСОВКА ---
    tab1, tab2, tab3 = st.tabs(["🖼️ Оригинал", "🎯 Маска", "🎛️ Результат (HSV)"])

    # --- TAB 1: Оригинал ---
    with tab1:
        img_small = resize_img(image, scale_percent)
        st.image(img_small, caption="Оригинал", use_container_width=True)
        st.download_button(
            "Скачать исходное изображение",
            data=to_bytes(img_small),
            file_name="original_small.png",
            mime="image/png"
        )

    # --- TAB 2: Маска ---
    with tab2:
        col1, col2 = st.columns([2, 1])  # слева пошире
        overlay = overlay_mask(image, mask, alpha=alpha)
        mask_vis = (mask == 1).astype(np.uint8) * 255

        with col1:
            st.image(overlay, caption="Оверлей маски (класс «линия»)", use_container_width=True)

        with col2:
            st.image(mask_vis, caption="Маска (линия)", use_container_width=True)

        st.download_button(
            "Скачать маску (линия)",
            data=to_bytes(np.stack([mask_vis] * 3, axis=-1)),
            file_name="mask_line.png",
            mime="image/png"
        )
    # --- TAB 3: Результат ---
    with tab3:
        adjusted = adjust_hsv(image, mask, h_adjust, s_adjust, v_adjust, index)
        adjusted_small = resize_img(adjusted, scale_percent)
        st.image(adjusted_small, caption=f"HSV-коррекция — {CLASSES[index]}", use_container_width=True)

        st.download_button(
            "💾 Скачать результат",
            data=to_bytes(adjusted_small),
            file_name="adjusted_small.png",
            mime="image/png"
        )

if __name__ == "__main__":
    main()
>>>>>>> master
