import csv
import json
import os
from fpdf import FPDF
import cv2
import numpy as np


# экспорт в csv
def export_csv(metrics, path="nornickel_metrics.csv"):
    with open(path, mode="w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f, delimiter=";")
        writer.writerow(["Параметр", "Значение", "Метрологический статус"])
        writer.writerow(["Доля талька", f"{metrics.get('talc_percent', 0)} %", "Автоматический расчет"])
        writer.writerow(["Общая доля сульфидов", f"{metrics.get('sulfide_percent', 0)} %", "Автоматический расчет"])
        writer.writerow(["Обычные срастания (рядовая)", f"{metrics.get('ordinary_percent_of_sulfides', 0)} %",
                         "Относительно объема сульфидов"])
        writer.writerow(["Тонкие срастания (труднообогатимая)", f"{metrics.get('thin_percent_of_sulfides', 0)} %",
                         "Относительно объема сульфидов"])
        writer.writerow(["Количество включений", f"{metrics.get('n_inclusions', 0)} ед.", "Количественный подсчет"])
        writer.writerow(["Размер изображения", str(metrics.get("image_size", "-")), "Геометрия кадра"])


# экспорт в pdf
class PDFReport(FPDF):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.has_cyrillic = False

        font_regular = os.path.join("fonts", "DejaVuSans.ttf")
        font_bold = os.path.join("fonts", "DejaVuSans-Bold.ttf")

        self.add_font("Cyrillic", "", font_regular)
        self.add_font("Cyrillic", "B", font_bold)

        self.has_cyrillic = True

    def header(self):
        self.set_fill_color(0, 128, 200)
        self.rect(0, 0, 210, 28, "F")

        self.set_text_color(255, 255, 255)
        self.set_font("Cyrillic", "B", 13) if self.has_cyrillic else self.set_font("Arial", "B", 13)
        self.cell(0, 4, "ПАО ГМК «НОРИЛЬСКИЙ НИКЕЛЬ»", border=0, ln=True, align="L")

        self.set_font("Cyrillic", "", 10) if self.has_cyrillic else self.set_font("Arial", "", 10)
        self.cell(0, 6, "Система автоматического контроля и фазового анализа минерального сырья", border=0, ln=True,
                  align="L")

        self.set_text_color(30, 30, 30)
        self.set_y(35)

    def footer(self):
        self.set_y(-15)
        if self.has_cyrillic:
            self.set_font("Cyrillic", "", 8)
        else:
            self.set_font("Arial", "", 8)
        self.set_text_color(128, 128, 128)
        self.cell(0, 10, f"Конфиденциально. Лабораторный паспорт сгенерирован автоматически. Страница {self.page_no()}",
                  align="R")


def export_pdf(result, path="nornickel_passport.pdf"):
    pdf = PDFReport()
    pdf.add_page()

    if pdf.has_cyrillic:
        pdf.set_font("Cyrillic", "B", 14)
    else:
        pdf.set_font("Arial", "B", 14)
    pdf.cell(0, 10, f"ТЕХНИЧЕСКИЙ ПАСПОРТ ОБРАЗЦА: {result.get('verdict', 'Не определен').upper()} РУДА", ln=True)
    pdf.ln(2)

    if pdf.has_cyrillic:
        pdf.set_font("Cyrillic", "", 11)
    else:
        pdf.set_font("Arial", "", 11)
    pdf.multi_cell(0, 6, f"Технологическое заключение: {result.get('conclusion', '-')}")
    pdf.ln(5)

    if pdf.has_cyrillic:
        pdf.set_font("Cyrillic", "B", 11)
    else:
        pdf.set_font("Arial", "B", 11)
    pdf.cell(0, 8, "Результаты количественного петрографического анализа включений:", ln=True)
    pdf.ln(2)

    if pdf.has_cyrillic:
        pdf.set_font("Cyrillic", "", 10)
    else:
        pdf.set_font("Arial", "", 10)

    metrics = result.get("metrics", {})
    table_data = [
        ("Содержание рассеянного талька в матрице", f"{metrics.get('talc_percent', 0)} %"),
        ("Общая площадь сульфидной вкрапленности", f"{metrics.get('sulfide_percent', 0)} %"),
        (" - Из них крупные зерна (Рядовая фракция)", f"{metrics.get('ordinary_percent_of_sulfides', 0)} %"),
        (" - Из них тонкие прорастания (Труднообогатимая фракция)", f"{metrics.get('thin_percent_of_sulfides', 0)} %"),
        ("Общее расчетное количество включений фаз", f"{metrics.get('n_inclusions', 0)} ед."),
        ("Разрешение кадра сканирования", f"{metrics.get('image_size', '-')} px")
    ]

    pdf.set_fill_color(240, 244, 248)
    pdf.cell(130, 8, "  Контролируемый геологический параметр", border=1, fill=True)
    pdf.cell(50, 8, "  Значение фазы", border=1, ln=True, fill=True)

    for i, (label, val) in enumerate(table_data):
        fill = (i % 2 == 1)
        pdf.set_fill_color(250, 252, 254) if fill else pdf.set_fill_color(255, 255, 255)
        pdf.cell(130, 8, f"  {label}", border=1, fill=True)
        pdf.cell(50, 8, f"  {val}", border=1, ln=True, fill=True)

    pdf.add_page()

    if "original_image_path" in result and os.path.exists(result["original_image_path"]):
        if pdf.has_cyrillic:
            pdf.set_font("Cyrillic", "B", 11)
        else:
            pdf.set_font("Arial", "B", 11)
        pdf.cell(0, 8, "1. Исходное оптическое изображение образца (шлиф):", ln=True)
        pdf.ln(2)

        y_orig = pdf.get_y()
        pdf.rect(29, y_orig - 1, 152, 77, "D")
        pdf.image(result["original_image_path"], x=30, y=y_orig, w=150, h=75)

        pdf.set_y(y_orig + 79)
        pdf.ln(6)

    if "overlay_image" in result and result["overlay_image"] is not None:
        if pdf.has_cyrillic:
            pdf.set_font("Cyrillic", "B", 11)
        else:
            pdf.set_font("Arial", "B", 11)
        pdf.cell(0, 8, "2. Карта пространственного распределения фаз (Маска сегментации):", ln=True)
        pdf.ln(2)

        temp_img_path = "temp_pdf_overlay.jpg"
        try:
            img_to_save = result["overlay_image"]
            if isinstance(img_to_save, np.ndarray):
                cv2.imwrite(temp_img_path, img_to_save)
                if os.path.exists(temp_img_path):
                    y_overlay = pdf.get_y()
                    pdf.rect(29, y_overlay - 1, 152, 77, "D")
                    pdf.image(temp_img_path, x=30, y=y_overlay, w=150, h=75)

                    pdf.set_y(y_overlay + 79)
                    pdf.ln(4)

                    pdf.set_text_color(140, 140, 140)
                    if pdf.has_cyrillic:
                        pdf.set_font("Cyrillic", "B", 9)
                    else:
                        pdf.set_font("Arial", "B", 9)
                    pdf.cell(0, 5, "Легенда карты фаз:", ln=True)

                    if pdf.has_cyrillic:
                        pdf.set_font("Cyrillic", "", 9)
                    else:
                        pdf.set_font("Arial", "", 9)
                    pdf.cell(0, 4.5, "• Зеленый - Обычные срастания", ln=True)
                    pdf.cell(0, 4.5, "• Красный - Тонкие срастания", ln=True)
                    pdf.cell(0, 4.5, "• Синий - Тальк", ln=True)

                    pdf.set_text_color(30, 30, 30)
        finally:
            if os.path.exists(temp_img_path):
                os.remove(temp_img_path)

    pdf.output(path)


# экспорт В geojson
def export_geojson(result, path="nornickel_gis_contours.geojson"):
    overlay = result.get("overlay_image")
    if overlay is None or not isinstance(overlay, np.ndarray):
        return

    features = []
    phases = {
        "Talc": {"color_bgr": (255, 0, 0), "name": "Тальк"},
        "Ordinary_Sulfides": {"color_bgr": (0, 255, 0), "name": "Обычные срастания сульфидов"},
        "Thin_Sulfides": {"color_bgr": (0, 0, 255), "name": "Тонкие срастания сульфидов"}
    }

    for phase_key, phase_info in phases.items():
        lower = np.array(phase_info["color_bgr"], dtype="uint8")
        upper = np.array(phase_info["color_bgr"], dtype="uint8")
        mask = cv2.inRange(overlay, lower, upper)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            if cv2.contourArea(cnt) < 50:
                continue
            epsilon = 0.005 * cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, epsilon, True)
            if len(approx) >= 3:
                coords = [[int(pt[0][0]), int(pt[0][1])] for pt in approx]
                coords.append(coords[0])

                features.append({
                    "type": "Feature",
                    "properties": {
                        "mineral_phase": phase_key,
                        "description": phase_info["name"],
                        "area_pixels": float(cv2.contourArea(cnt))
                    },
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [coords]
                    }
                })

    with open(path, "w", encoding="utf-8") as f:
        json.dump({"type": "FeatureCollection", "features": features}, f, ensure_ascii=False, indent=2)
