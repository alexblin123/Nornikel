"""
Пакетная обработка серии шлифов + логирование (воспроизводимость).
Использование:
    python batch_process.py --input images_folder --output results_folder
"""
import os
import csv
import glob
import json
import argparse
import datetime
import logging

import cv2
from analyzer import ShlifAnalyzer


def setup_logger(output_dir):
    os.makedirs(output_dir, exist_ok=True)
    log_path = os.path.join(output_dir, f"batch_log_{datetime.datetime.now():%Y%m%d_%H%M%S}.txt")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler(log_path, encoding="utf-8"),
            logging.StreamHandler()
        ]
    )
    return log_path


def main():
    parser = argparse.ArgumentParser(description="Пакетный анализ шлифов НОРНИКЕЛЬ")
    parser.add_argument("--input", required=True, help="Папка с изображениями")
    parser.add_argument("--output", default="batch_results", help="Папка для результатов")
    parser.add_argument("--save-overlay", action="store_true", help="Сохранять визуализации")
    args = parser.parse_args()

    log_path = setup_logger(args.output)
    logging.info("=== ПАКЕТНАЯ ОБРАБОТКА ШЛИФОВ ===")
    logging.info(f"Входная папка: {args.input}")
    logging.info(f"Выходная папка: {args.output}")

    # Логируем параметры для воспроизводимости
    analyzer = ShlifAnalyzer()
    logging.info(f"U-Net талька загружен: {getattr(analyzer, 'talc_model', None) is not None}")

    exts = ("*.png", "*.jpg", "*.jpeg", "*.tif", "*.tiff")
    files = []
    for e in exts:
        files.extend(glob.glob(os.path.join(args.input, e)))
    files = sorted(files)
    logging.info(f"Найдено изображений: {len(files)}")

    if not files:
        logging.warning("Нет изображений для обработки.")
        return

    csv_path = os.path.join(args.output, "batch_results.csv")
    fieldnames = ["filename", "verdict", "talc_percent", "sulfide_percent",
                  "thin_percent_of_sulfides", "ordinary_percent_of_sulfides",
                  "n_inclusions", "talc_source", "image_size", "status"]

    results_summary = []
    with open(csv_path, "w", newline="", encoding="utf-8") as cf:
        writer = csv.DictWriter(cf, fieldnames=fieldnames)
        writer.writeheader()

        for i, fpath in enumerate(files, 1):
            fname = os.path.basename(fpath)
            try:
                logging.info(f"[{i}/{len(files)}] Обработка: {fname}")
                result = analyzer.analyze(fpath)
                m = result["metrics"]
                thin_pct = m.get("thin_percent_of_sulfides", 0)
                m["ordinary_percent_of_sulfides"] = round(100.0 - thin_pct, 2)

                row = {
                    "filename": fname,
                    "verdict": result["verdict"],
                    "talc_percent": m.get("talc_percent"),
                    "sulfide_percent": m.get("sulfide_percent"),
                    "thin_percent_of_sulfides": m.get("thin_percent_of_sulfides"),
                    "ordinary_percent_of_sulfides": m.get("ordinary_percent_of_sulfides"),
                    "n_inclusions": m.get("n_inclusions"),
                    "talc_source": m.get("talc_source", "не обнаружен"),
                    "image_size": m.get("image_size"),
                    "status": "OK"
                }
                writer.writerow(row)
                results_summary.append(row)

                if args.save_overlay and result.get("overlay_image") is not None:
                    ov_path = os.path.join(args.output, f"overlay_{fname}.png")
                    cv2.imwrite(ov_path, result["overlay_image"])

                logging.info(f"    → Вердикт: {result['verdict']}, тальк: {m.get('talc_percent')}%")
            except Exception as ex:
                logging.error(f"    ОШИБКА обработки {fname}: {ex}")
                writer.writerow({"filename": fname, "status": f"ERROR: {ex}"})

    # Итоговая сводка в JSON (воспроизводимость)
    summary = {
        "timestamp": datetime.datetime.now().isoformat(),"input_dir": args.input,
        "total_images": len(files),
        "processed_ok": sum(1 for r in results_summary if r["status"] == "OK"),
        "verdicts": {}
    }
    for r in results_summary:
        v = r["verdict"]
        summary["verdicts"][v] = summary["verdicts"].get(v, 0) + 1

    with open(os.path.join(args.output, "batch_summary.json"), "w", encoding="utf-8") as jf:
        json.dump(summary, jf, indent=4, ensure_ascii=False)

    logging.info("=== ЗАВЕРШЕНО ===")
    logging.info(f"Результаты: {csv_path}")
    logging.info(f"Сводка: {summary['verdicts']}")
    logging.info(f"Лог: {log_path}")


if name == "__main__":
    main()
