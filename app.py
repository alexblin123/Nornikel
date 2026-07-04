import os
import cv2
import json
import base64
import datetime
from io import BytesIO
import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
from streamlit_drawable_canvas import st_canvas

# 安排 Импортируем анализатор
from analyzer import ShlifAnalyzer, resize_if_large

analyzer = ShlifAnalyzer()

try:
    from report import export_csv, export_pdf, export_geojson
except ImportError:
    export_csv, export_pdf, export_geojson = None, None, None


# --- 🚀 КЕШИРОВАНИЕ ДАННЫХ (Убирает зависания и повторные расчеты) ---
@st.cache_data
def cached_analyze(temp_path):
    return analyzer.analyze(temp_path)


# --- НАСТРОЙКА СТРАНИЦЫ ---
st.set_page_config(
    page_title="НОРНИКЕЛЬ | Автоматизация анализа шлифов",
    page_icon=None,
    layout="wide"
)

# --- ИНЪЕКЦИЯ СТИЛЕЙ СВЕРХКОМПАКТНОСТИ ---
st.markdown("""
    <style>
        [data-testid="stAppViewContainer"] { background-color: #FFFFFF !important; }
        [data-testid="stHeader"] { display: none !important; }
        .stMarkdown, p, label, h3, h5, span, th, td, div { color: #2A2A2A !important; }

        .brand-header { color: #0080C8 !important; font-family: 'Segoe UI', Arial, sans-serif; font-weight: 700; font-size: 26px; margin-bottom: 2px; margin-top: -30px; }
        .brand-subtitle { color: #7F8C8D !important; font-size: 12px; margin-bottom: 15px; text-transform: uppercase; letter-spacing: 1px; }

        .stAlert { border-left: 5px solid #0080C8 !important; background-color: #F4F9FC !important; padding: 0.5rem !important; }
        .stAlert p { color: #2A2A2A !important; margin: 0 !important; }
        table { background-color: #FFFFFF !important; color: #2A2A2A !important; font-size: 13px; }

        div.stButton > button:first-child { background-color: #0080C8 !important; color: white !important; border-radius: 4px !important; border: none !important; font-weight: 600 !important; transition: background-color 0.3s ease; padding: 0.2rem 0.5rem !important; }
        div.stButton > button:first-child:hover { background-color: #004B78 !important; color: white !important; }
        .brand-footer { text-align: center; color: #95A5A6; font-size: 11px; margin-top: 30px; border-top: 1px solid #E2E8F0; padding-top: 10px; }

        button[title="View fullscreen"] { display: none !important; }
        [data-testid="stImageHoverButtons"] { display: none !important; visibility: hidden !important; }

        .stRadio { margin-bottom: -10px !important; }
        .stSlider { margin-bottom: -10px !important; }
        hr { margin: 0.4em 0 !important; }
    </style>
""", unsafe_allow_html=True)


def mask_to_base64(mask_array):
    img_pil = Image.fromarray(mask_array.astype(np.uint8))
    buffer = BytesIO()
    img_pil.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


# Сохраняем состояние режима редактирования в сессии
if "edit_mode" not in st.session_state:
    st.session_state.edit_mode = False

# --- БРЕНДИНГ ---
st.markdown('<div class="brand-header">НОРНИКЕЛЬ</div>', unsafe_allow_html=True)
st.markdown('<div class="brand-subtitle">Цифровая лаборатория обогащения | Автоклассификация руд</div>',
            unsafe_allow_html=True)
st.divider()

# --- СЕТКА ГЛАВНОГО ЭКРАНА: ВСЁ УПРАВЛЕНИЕ СЛЕВА, КАРТИНКА И ХОЛСТ СПРАВА ---
col_left, col_right = st.columns([1.2, 1.8])

with col_left:
    uploaded_file = st.file_uploader(
        "Выберите микрофотографию рудного шлифа:",
        type=["tiff", "tif", "png", "jpg", "jpeg"]
    )

if uploaded_file is not None:
    temp_path = "temp_core_analysis.jpg"
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Вызов анализатора (кэширован, не перезапускается при кликах)
    result = cached_analyze(temp_path)

    if "metrics" in result:
        thin_pct = result["metrics"].get("thin_percent_of_sulfides", 0)
        result["metrics"]["ordinary_percent_of_sulfides"] = round(100.0 - thin_pct, 2)

    verdict = result["verdict"]

    # Отрисовка левой панели управления
    with col_left:
        if verdict == "Оталькованная":
            status_badge = '<span style="color: #0080C8; font-weight: bold;">🔵 Оталькованная руда</span>'
        elif verdict in ["Рядовой", "Рядовая"]:
            status_badge = '<span style="color: #28A745; font-weight: bold;">🟢 Рядовая руда</span>'
        else:
            status_badge = '<span style="color: #DC3545; font-weight: bold;">🔴 Труднообогатимая руда</span>'

        st.markdown(f"**Статус:** {status_badge}", unsafe_allow_html=True)
        st.info(f"**Заключение:** {result['conclusion']}")
        st.markdown("---")

        # --- КЕЙС 1: ВКЛЮЧЕН РЕЖИМ РУЧНОЙ РАЗМЕТКИ ШЛИФА ---
        if st.session_state.edit_mode:
            st.markdown("<div style='font-weight: 700; color: #0080C8;'>⚙️ ИНСТРУМЕНТЫ РЕДАКТОРА:</div>",
                        unsafe_allow_html=True)
            drawing_mode = st.radio("Инструмент:", ("Кисть", "Полигон"), horizontal=True)
            brush_size = st.slider("Размер кисти:", min_value=1, max_value=10, value=3)

            phase_choice = st.radio(
                "Класс минерала для обводки:",
                ["Зеленый (Обычные)", "Красный (Тонкие)", "Синий (Тальк)"]
            )
            stroke_color = "rgba(40, 167, 69, 1)" if "Зеленый" in phase_choice else "rgba(220, 53, 69, 1)" if "Красный" in phase_choice else "rgba(0, 128, 200, 1)"

            corrected_verdict = st.selectbox(
                "Итоговый экспертный вердикт:",
                ["Оталькованная", "Рядовой", "Труднообогатимая"],
                index=["Оталькованная", "Рядовой", "Труднообогатимая"].index(verdict)
            )

            st.markdown("<br>", unsafe_allow_html=True)
            btn_save, btn_cancel = st.columns(2)
            with btn_save:
                submit_clicked = st.button("💾 Сохранить JSON", use_container_width=True)
            with btn_cancel:
                if st.button("❌ Выйти без сохранения", use_container_width=True):
                    st.session_state.edit_mode = False
                    st.rerun()

        # --- КЕЙС 2: РЕЖИМ ПРОСМОТРА КАРТЫ ФАЗ МЛ ---
        else:
            st.markdown("<div style='font-weight: 600; margin-bottom:5px;'>Управление маской:</div>",
                        unsafe_allow_html=True)
            show_mask = st.checkbox("Отображать слои ML", value=True)
            mask_alpha = st.slider("Прозрачность маски", min_value=0.0, max_value=1.0, value=0.45, step=0.05,
                                   disabled=not show_mask)

            st.markdown("""
            <div style="background-color: #F8F9FA; padding: 10px; border-radius: 4px; border: 1px solid #E2E8F0; margin-top: 10px; margin-bottom: 10px;">
                <span style="font-size: 12px; color: #2A2A2A;">🟢 Зеленый: Рядовые сульфиды | 🔴 Красный: Тонкие срастания | 🔵 Синий: Тальк</span>
            </div>
            """, unsafe_allow_html=True)

            if st.button("🖍️ Открыть ручной редактор зон", use_container_width=True):
                st.session_state.edit_mode = True
                st.rerun()

    # --- ОТРИСОВКА ПРАВОЙ ПАНЕЛИ (ИЗОБРАЖЕНИЕ ИЛИ ХОЛСТ СТРИМЛИТА) ---
    with col_right:
        orig_bgr = cv2.imread(temp_path)
        orig_resized = resize_if_large(orig_bgr, 1200)
        orig_w, orig_h = orig_resized.shape[1], orig_resized.shape[0]

        # Оптимальная ширина для вывода без скролла
        target_width = 720
        scale_percent = target_width / float(orig_w)
        canvas_width = target_width
        canvas_height = int(float(orig_h) * scale_percent)

        # 🟢 ЕСЛИ ВКЛЮЧЕН РЕДАКТОР: Выводим интерактивный холст
        if st.session_state.edit_mode:
            st.markdown(
                "<p style='font-size: 12px; color: #7F8C8D; margin-bottom: 2px;'>Окно интерактивной обводки шлифа:</p>",
                unsafe_allow_html=True)
            bg_pil = Image.open(temp_path).convert("RGB").resize((canvas_width, canvas_height),
                                                                 Image.Resampling.LANCZOS)

            canvas_result = st_canvas(
                fill_color="rgba(0, 0, 0, 0)",
                stroke_width=brush_size,
                stroke_color=stroke_color,
                background_image=bg_pil,
                update_streamlit=True,
                height=canvas_height,
                width=canvas_width,
                drawing_mode="freedraw" if drawing_mode == "Кисть" else "polygon",
                key="nornickel_stable_canvas"
            )

            # Логика сохранения разметки в JSON
            if submit_clicked and canvas_result is not None:
                if canvas_result.image_data is not None:
                    drawn_rgba = canvas_result.image_data
                    mask_ordinary = np.zeros((canvas_height, canvas_width), dtype=np.uint8)
                    mask_thin = np.zeros((canvas_height, canvas_width), dtype=np.uint8)
                    mask_talc = np.zeros((canvas_height, canvas_width), dtype=np.uint8)

                    green_pixels = (drawn_rgba[:, :, 0] < 100) & (drawn_rgba[:, :, 1] > 100) & (
                            drawn_rgba[:, :, 2] < 100) & (drawn_rgba[:, :, 3] > 0)
                    mask_ordinary[green_pixels] = 255
                    red_pixels = (drawn_rgba[:, :, 0] > 150) & (drawn_rgba[:, :, 1] < 100) & (
                            drawn_rgba[:, :, 2] < 100) & (drawn_rgba[:, :, 3] > 0)
                    mask_thin[red_pixels] = 255
                    blue_pixels = (drawn_rgba[:, :, 0] < 100) & (drawn_rgba[:, :, 1] > 50) & (
                            drawn_rgba[:, :, 2] > 150) & (drawn_rgba[:, :, 3] > 0)
                    mask_talc[blue_pixels] = 255

                    real_w, real_h = Image.open(temp_path).size
                    m_ordinary_orig = cv2.resize(mask_ordinary, (real_w, real_h), interpolation=cv2.INTER_NEAREST)
                    m_thin_orig = cv2.resize(mask_thin, (real_w, real_h), interpolation=cv2.INTER_NEAREST)
                    m_talc_orig = cv2.resize(mask_talc, (real_w, real_h), interpolation=cv2.INTER_NEAREST)

                    export_payload = {
                        "image_id": uploaded_file.name,
                        "image_size": {"width": real_w, "height": real_h},
                        "user_id": "geolog_ivanov",
                        "timestamp": datetime.datetime.now().isoformat(),
                        "masks": {
                            "talc": mask_to_base64(m_talc_orig),
                            "thin": mask_to_base64(m_thin_orig),
                            "ordinary": mask_to_base64(m_ordinary_orig)
                        },
                        "original_verdict": verdict,
                        "corrected_verdict": corrected_verdict
                    }

                    os.makedirs("active_learning_dataset", exist_ok=True)
                    with open(os.path.join("active_learning_dataset", f"markup_{uploaded_file.name}.json"), "w",
                              encoding="utf-8") as jf:
                        json.dump(export_payload, jf, indent=4, ensure_ascii=False)

                    st.success("✅ Пакет успешно экспортирован в `active_learning_dataset/`!")
                    st.session_state.edit_mode = False
                    st.rerun()

        # 🔵 ЕСЛИ РЕЖИМ ПРОСМОТРА: Выводим обычный результат ML
        else:
            overlay_bgr = result["overlay_image"]
            if show_mask and overlay_bgr is not None:
                blended_bgr = cv2.addWeighted(orig_resized, 1.0 - mask_alpha, overlay_bgr, mask_alpha, 0)
                display_rgb = cv2.cvtColor(blended_bgr, cv2.COLOR_BGR2RGB)
            else:
                display_rgb = cv2.cvtColor(orig_resized, cv2.COLOR_BGR2RGB)

            st.image(display_rgb, use_column_width=True)

    # --- ТАБЛИЦА МЕТРИК И КНОПКИ СКАЧИВАНИЯ ПОД ФОТО ---
    st.divider()
    st.markdown("##### Сводные количественные параметры микроструктуры")
    m = result["metrics"]

    df_metrics = pd.DataFrame({
        "Технологический параметр микроструктуры": [
            "Массовая доля талька в нерудной матрице",
            "Общая интегральная доля сульфидных фаз",
            "Доля крупных рядовых сульфидных зерен",
            "Доля тонкодисперсных труднообогатимых срастаний",
            "Общее количество распознанных включений",
            "Линейное разрешение панорамы анализа"
        ],
        "Значение": [
            f"{m['talc_percent']}%", f"{m['sulfide_percent']}%",
            f"{m['ordinary_percent_of_sulfides']}%", f"{m['thin_percent_of_sulfides']}%",
            f"{m['n_inclusions']} ед.", f"{m['image_size']} px"
        ]
    })
    st.table(df_metrics)

    st.markdown("##### Формирование отчетных документов")
    btn_col1, btn_col2, btn_col3 = st.columns(3)

    with btn_col1:
        if export_csv:
            csv_buffer = BytesIO()
            csv_df = pd.DataFrame([m])
            csv_df.to_csv(csv_buffer, index=False, sep=";")
            st.download_button("💾 Экспорт в CSV", csv_buffer.getvalue(), file_name="nornickel_metrics.csv",
                               mime="text/csv", use_container_width=True)

    with btn_col2:
        if export_pdf:
            pdf_path = "nornickel_passport.pdf"
            export_pdf(result, pdf_path)
            if os.path.exists(pdf_path):
                with open(pdf_path, "rb") as f:
                    st.download_button("📄 Сгенерировать PDF-паспорт шлифа", f.read(),
                                       file_name="nornickel_passport.pdf", mime="application/pdf",
                                       use_container_width=True)

    with btn_col3:
        if export_geojson:
            geojson_path = "nornickel_gis_contours.geojson"
            export_geojson(result, geojson_path)
            if os.path.exists(geojson_path):
                with open(geojson_path, "rb") as f:
                    st.download_button("🗺️ Выгрузить векторные контуры (GeoJSON)", f.read(),
                                       file_name="nornickel_gis_contours.geojson", mime="application/geo+json",
                                       use_container_width=True)

else:
    st.info(
        "Системное уведомление: Для начала работы загрузите исходное панорамное изображение (.tiff, .png, .jpg) в модуль обработки.")

st.markdown(
    '<div class="brand-footer">ПАО ГМК «Норильский никель» © 2026. Разработано в рамках хакатона автоматизации анализа шлифов.</div>',
    unsafe_allow_html=True)
