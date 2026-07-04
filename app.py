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

# Импортируем анализатор и утилиту ресайза
from analyzer import ShlifAnalyzer, resize_if_large

analyzer = ShlifAnalyzer()

try:
    from report import export_csv, export_pdf, export_geojson
except ImportError:
    export_csv, export_pdf, export_geojson = None, None, None

# --- НАСТРОЙКА СТРАНИЦЫ ---
st.set_page_config(
    page_title="НОРНИКЕЛЬ | Автоматизация анализа шлифов",
    page_icon=None,
    layout="wide"
)

# --- ИНЪЕКЦИЯ КОРПОРАТИВНОГО СТИЛЯ ---
st.markdown("""
    <style>
        [data-testid="stAppViewContainer"] { background-color: #FFFFFF !important; }
        [data-testid="stHeader"] { display: none !important; }
        .stMarkdown, p, label, h3, h5, span, th, td, div { color: #2A2A2A !important; }

        .brand-header { color: #0080C8 !important; font-family: 'Segoe UI', Arial, sans-serif; font-weight: 700; font-size: 28px; margin-bottom: 0px; margin-top: -30px; }
        .brand-subtitle { color: #7F8C8D !important; font-size: 13px; margin-bottom: 15px; text-transform: uppercase; letter-spacing: 1px; }

        .stAlert { border-left: 5px solid #0080C8 !important; background-color: #F4F9FC !important; padding: 0.5rem !important; }
        .stAlert p { color: #2A2A2A !important; margin: 0 !important; }
        table { background-color: #FFFFFF !important; color: #2A2A2A !important; font-size: 14px; }

        div.stButton > button:first-child { background-color: #0080C8 !important; color: white !important; border-radius: 4px !important; border: none !important; font-weight: 600 !important; transition: background-color 0.3s ease; padding: 0.25rem 0.5rem !important; }
        div.stButton > button:first-child:hover { background-color: #004B78 !important; color: white !important; }
        .brand-footer { text-align: center; color: #95A5A6; font-size: 12px; margin-top: 50px; border-top: 1px solid #E2E8F0; padding-top: 15px; }

        button[title="View fullscreen"] { display: none !important; }
        [data-testid="stImageHoverButtons"] { display: none !important; visibility: hidden !important; }

        /* Расширяем модальное окно на 95% ширины экрана */
        div[role="dialog"] {
            width: 95vw !important;
            max-width: 1350px !important;
            padding: 1rem !important;
            border-radius: 8px !important;
        }
        div[data-testid="stDialog"] {
            width: 95vw !important;
            max-width: 1350px !important;
            gap: 0.5rem !important;
        }

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


# --- 🖍️ ОКНО ЭКСПЕРТНОЙ РАЗМЕТКИ (ACTIVE LEARNING) ---
@st.experimental_dialog("Режим экспертной разметки (Обучение с подкреплением)")
def show_markup_modal(saved_img_path, original_filename, original_verdict):
    modal_left, modal_right = st.columns([1.1, 2.9])

    with modal_left:
        st.markdown("<div style='font-weight: 600; font-size: 14px;'>Инструмент рисования:</div>",
                    unsafe_allow_html=True)
        drawing_mode = st.radio("Инструмент:", ("Кисть", "Полигон"), horizontal=True, label_visibility="collapsed")

        # Исправленный размер кисти от 1 до 10
        st.markdown("<div style='font-weight: 600; font-size: 14px; margin-top: 5px;'>Размер кисти:</div>",
                    unsafe_allow_html=True)
        brush_size = st.slider("Размер кисти:", min_value=1, max_value=10, value=3, label_visibility="collapsed")

        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown("<div style='font-weight: 600; font-size: 14px;'>Минеральная фаза:</div>", unsafe_allow_html=True)
        phase_choice = st.radio(
            "Phase",
            ["Зеленый (Обычные)", "Красный (Тонкие)", "Синий (Тальк)"],
            label_visibility="collapsed"
        )
        if "Зеленый" in phase_choice:
            stroke_color = "rgba(40, 167, 69, 1)"
        elif "Красный" in phase_choice:
            stroke_color = "rgba(220, 53, 69, 1)"
        else:
            stroke_color = "rgba(0, 128, 200, 1)"

        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown("<div style='font-weight: 600; font-size: 14px;'>Экспертный вердикт:</div>", unsafe_allow_html=True)
        corrected_verdict = st.selectbox(
            "Verdict",
            ["Оталькованная", "Рядовой", "Труднообогатимая"],
            index=["Оталькованная", "Рядовой", "Труднообогатимая"].index(original_verdict),
            label_visibility="collapsed"
        )

        st.markdown("<br><br>", unsafe_allow_html=True)
        submit_clicked = st.button("💾 Зафиксировать и отправить в ML", use_container_width=True)

    actual_mode = "freedraw" if drawing_mode == "Кисть" else "polygon"

    with modal_right:
        try:
            bg_img = Image.open(saved_img_path).convert("RGB")
            orig_w, orig_h = bg_img.size

            base_canvas_width = 750

            canvas_width = base_canvas_width
            w_percent = canvas_width / float(orig_w)
            canvas_height = int(orig_h * w_percent)

            bg_img_resized = bg_img.resize(
                (canvas_width, canvas_height),
                Image.Resampling.LANCZOS
            )

            canvas_result = st_canvas(
                fill_color="rgba(0, 0, 0, 0)",
                stroke_width=brush_size,
                stroke_color=stroke_color,
                background_image=bg_img_resized,
                update_streamlit=True,
                height=canvas_height,
                width=canvas_width,
                drawing_mode=actual_mode,
                key="modal_expert_canvas",
            )
        except Exception as e:
            st.error(f"Ошибка загрузки холста: {e}")
            canvas_result = None

    # --- СОХРАНЕНИЕ JSON ---
    if submit_clicked and canvas_result is not None:
        if canvas_result.image_data is not None:
            drawn_rgba = canvas_result.image_data

            mask_ordinary = np.zeros((canvas_height, canvas_width), dtype=np.uint8)
            mask_thin = np.zeros((canvas_height, canvas_width), dtype=np.uint8)
            mask_talc = np.zeros((canvas_height, canvas_width), dtype=np.uint8)

            green_pixels = (drawn_rgba[:, :, 0] < 100) & (drawn_rgba[:, :, 1] > 100) & (drawn_rgba[:, :, 2] < 100) & (
                    drawn_rgba[:, :, 3] > 0)
            mask_ordinary[green_pixels] = 255

            red_pixels = (drawn_rgba[:, :, 0] > 150) & (drawn_rgba[:, :, 1] < 100) & (drawn_rgba[:, :, 2] < 100) & (
                    drawn_rgba[:, :, 3] > 0)
            mask_thin[red_pixels] = 255

            blue_pixels = (drawn_rgba[:, :, 0] < 100) & (drawn_rgba[:, :, 1] > 50) & (drawn_rgba[:, :, 2] > 150) & (
                    drawn_rgba[:, :, 3] > 0)
            mask_talc[blue_pixels] = 255

            mask_ordinary_orig = cv2.resize(mask_ordinary, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
            mask_thin_orig = cv2.resize(mask_thin, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
            mask_talc_orig = cv2.resize(mask_talc, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)

            export_payload = {
                "image_id": original_filename,
                "image_size": {"width": orig_w, "height": orig_h},
                "user_id": "geolog_ivanov",
                "timestamp": datetime.datetime.now().isoformat(),
                "masks": {
                    "talc": mask_to_base64(mask_talc_orig),
                    "thin": mask_to_base64(mask_thin_orig),
                    "ordinary": mask_to_base64(mask_ordinary_orig)
                },
                "original_verdict": original_verdict,
                "corrected_verdict": corrected_verdict
            }

            os.makedirs("active_learning_dataset", exist_ok=True)
            json_path = os.path.join("active_learning_dataset", f"markup_{original_filename}.json")
            with open(json_path, "w", encoding="utf-8") as jf:
                json.dump(export_payload, jf, indent=4, ensure_ascii=False)

            cv2.imwrite(os.path.join("active_learning_dataset", f"mask_ordinary_{original_filename}.png"),
                        mask_ordinary_orig)
            cv2.imwrite(os.path.join("active_learning_dataset", f"mask_thin_{original_filename}.png"), mask_thin_orig)
            cv2.imwrite(os.path.join("active_learning_dataset", f"mask_talc_{original_filename}.png"), mask_talc_orig)

            st.success("✅ Экспорт завершен! Пакет сохранен в папку `active_learning_dataset/`.")
        else:
            st.warning("Нанесите разметку перед отправкой.")


# --- ГЛАВНАЯ СТРАНИЦА ---
st.markdown('<div class="brand-header">НОРНИКЕЛЬ</div>', unsafe_allow_html=True)
st.markdown('<div class="brand-subtitle">Цифровая лаборатория обогащения | Автоклассификация руд</div>',
            unsafe_allow_html=True)
st.divider()

uploaded_file = st.file_uploader(
    "Выберите микрофотографию рудного шлифа для автоматической сегментации фаз",
    type=["tiff", "tif", "png", "jpg", "jpeg"]
)

if uploaded_file is not None:
    temp_path = "temp_core_analysis.jpg"
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    with st.spinner("Интеллектуальный анализ структуры шлифа OpenCV..."):
        result = analyzer.analyze(temp_path)
        result["original_image_path"] = temp_path

        if "metrics" in result:
            thin_pct = result["metrics"].get("thin_percent_of_sulfides", 0)
            result["metrics"]["ordinary_percent_of_sulfides"] = round(100.0 - thin_pct, 2)

    verdict = result["verdict"]
    if verdict == "Оталькованная":
        status_badge = '<span style="color: #0080C8; font-weight: bold;">🔵 Оталькованная руда</span>'
    elif verdict == "Рядовой":
        status_badge = '<span style="color: #28A745; font-weight: bold;">🟢 Рядовая руда</span>'
    else:
        status_badge = '<span style="color: #DC3545; font-weight: bold;">🔴 Труднообогатимая руда</span>'

    st.markdown(f"### Экспресс-оценка образца: {status_badge}", unsafe_allow_html=True)
    st.info(f"**Официальное заключение автоматизированной системы:** {result['conclusion']}")

    st.markdown("##### Интерактивный анализ и верификация фаз")

    orig_bgr = cv2.imread(temp_path)
    orig_resized = resize_if_large(orig_bgr, 1500)
    overlay_bgr = result["overlay_image"]

    main_col1, main_col2 = st.columns([1.3, 1.7])

    with main_col2:
        st.markdown(
            "<div style='margin-bottom: 5px; font-weight: 600; color: #2A2A2A;'>Управление слоями разметки:</div>",
            unsafe_allow_html=True)
        show_mask = st.checkbox("Отображать маску ML", value=True)
        mask_alpha = st.slider("Прозрачность маски", min_value=0.0, max_value=1.0, value=0.45, step=0.05,
                               disabled=not show_mask)

    if show_mask and overlay_bgr is not None:
        blended_bgr = cv2.addWeighted(orig_resized, 1.0 - mask_alpha, overlay_bgr, mask_alpha, 0)
        display_rgb = cv2.cvtColor(blended_bgr, cv2.COLOR_BGR2RGB)
    else:
        display_rgb = cv2.cvtColor(orig_resized, cv2.COLOR_BGR2RGB)

    with main_col1:
        st.image(display_rgb, use_column_width=True)

    with main_col2:
        st.markdown("""
        <div style="background-color: #F8F9FA; padding: 12px; border-radius: 4px; border: 1px solid #E2E8F0; margin-top: 12px; margin-bottom: 15px;">
            <strong style="color: #2A2A2A !important; display: block; margin-bottom: 8px;">Легенда карты фаз:</strong>
            <div style="display: flex; align-items: center; margin-bottom: 6px;">
                <svg width="12" height="12" style="margin-right: 8px; flex-shrink: 0;"><rect width="12" height="12" rx="2" fill="#28A745" /></svg>
                <span style="color: #2A2A2A !important; font-size: 13px;"><strong>Зеленый:</strong> Обычные срастания (Рядовая)</span>
            </div>
            <div style="display: flex; align-items: center; margin-bottom: 6px;">
                <svg width="12" height="12" style="margin-right: 8px; flex-shrink: 0;"><rect width="12" height="12" rx="2" fill="#DC3545" /></svg>
                <span style="color: #2A2A2A !important; font-size: 13px;"><strong>Красный:</strong> Тонкие срастания (Труднообогатимая)</span>
            </div>
            <div style="display: flex; align-items: center;">
                <svg width="12" height="12" style="margin-right: 8px; flex-shrink: 0;"><rect width="12" height="12" rx="2" fill="#0080C8" /></svg>
                <span style="color: #2A2A2A !important; font-size: 13px;"><strong>Синий:</strong> Тальк (&gt;10% площади)</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        if st.button("🖍️ Включить режим ручной разметки зон", use_container_width=True):
            show_markup_modal(temp_path, uploaded_file.name, verdict)

    st.divider()

    st.markdown("##### Сводные количественные параметры шлифа")
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
            f"{m['talc_percent']}%",
            f"{m['sulfide_percent']}%",
            f"{m['ordinary_percent_of_sulfides']}%",
            f"{m['thin_percent_of_sulfides']}%",
            f"{m['n_inclusions']} ед.",
            f"{m['image_size']} px"
        ]
    })
    st.table(df_metrics)

    st.markdown("##### Формирование отчетных документов")
    btn_col1, btn_col2, btn_col3 = st.columns(3)

    with btn_col1:
        if export_csv:
            csv_path = "nornickel_metrics.csv"
            export_csv(m, csv_path)
            with open(csv_path, "rb") as f:
                st.download_button("💾 Экспорт в CSV", f, file_name="nornickel_metrics.csv", mime="text/csv",
                                   use_container_width=True)

    with btn_col2:
        if export_pdf:
            pdf_path = "nornickel_passport.pdf"
            export_pdf(result, pdf_path)
            with open(pdf_path, "rb") as f:
                st.download_button("📄 Сгенерировать PDF", f, file_name="nornickel_passport.pdf",
                                   mime="application/pdf", use_container_width=True)

    with btn_col3:
        if export_geojson:
            geojson_path = "nornickel_gis_contours.geojson"
            export_geojson(result, geojson_path)
            if os.path.exists(geojson_path):
                with open(geojson_path, "rb") as f:
                    st.download_button("🗺️ Выгрузить GeoJSON", f, file_name="nornickel_gis_contours.geojson",
                                       mime="application/geo+json", use_container_width=True)

else:
    st.info(
        "Системное уведомление: Для начала работы загрузите исходное панорамное изображение (.tiff, .png, .jpg) в модуль обработки.")

st.markdown(
    '<div class="brand-footer">ПАО ГМК «Норильский никель» © 2026. Разработано в рамках хакатона автоматизации анализа шлифов.</div>',
    unsafe_allow_html=True)
