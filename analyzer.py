"""
analyzer.py — Анализатор геологических шлифов.
Классификация сорта: CNN (ResNet18, F1 92.4%).
Сегментация маски (тальк/тонкие/обычные): OpenCV.
Классы: Оталькованная / Рядовая / Труднообогатимая.
"""
import cv2
import numpy as np
import os
import torch
import torch.nn as nn
from torchvision import models, transforms

# ---------- КОНФИГ ----------
BLUE_LOWER = np.array([110, 80, 60])
BLUE_UPPER = np.array([130, 255, 255])
SULFIDE_MIN_AREA = 30
SOLIDITY_THRESHOLD = 0.75
TALC_THRESHOLD = 10.0

_HERE = os.path.dirname(os.path.abspath(__file__))
CNN_PATH = os.path.join(_HERE, "cnn_model.pth")
CLASS_NAMES = ["Рядовая", "Труднообогатимая"]  # 0, 1 — как при обучении!
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

COLOR_ORDINARY = (0, 255, 0)  # зелёный — обычные срастания
COLOR_THIN = (0, 0, 255)  # красный — тонкие срастания
COLOR_TALC = (255, 0, 0)  # синий — тальк
OVERLAY_ALPHA = 0.45

_CNN_TF = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


# ---------- УТИЛИТЫ ----------
def imread_unicode(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Файл не найден: {path}")
    data = np.fromfile(path, dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Не удалось прочитать: {path}")
    return img


def imwrite_unicode(path, img):
    ext = os.path.splitext(path)[1] or ".png"
    ok, buf = cv2.imencode(ext, img)
    if ok:
        buf.tofile(path)
    return ok


def resize_if_large(img, max_side=1500):
    h, w = img.shape[:2]
    s = max_side / max(h, w)
    if s < 1.0:
        img = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    return img


# ---------- ТАЛЬК (по синей разметке) ----------
def detect_talc_blue(image):
    h, w = image.shape[:2]
    total = h * w
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    blue = cv2.inRange(hsv, BLUE_LOWER, BLUE_UPPER)
    if (blue > 0).sum() < 0.0005 * total:
        return np.zeros((h, w), np.uint8), 0.0, False
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    closed = cv2.morphologyEx(blue, cv2.MORPH_CLOSE, k, iterations=2)
    cnts, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros((h, w), np.uint8)
    for c in cnts:
        if cv2.contourArea(c) > 100:
            cv2.drawContours(mask, [c], -1, 255, cv2.FILLED)
    return mask, float((mask > 0).sum()) / total * 100, True


# ---------- СУЛЬФИДЫ + МАСКИ ----------
def extract_features_and_masks(image, exclude_mask=None):
    h, w = image.shape[:2]
    total = h * w
    gray = cv2.GaussianBlur(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), (5, 5), 0)
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if exclude_mask is not None:
        th = cv2.bitwise_and(th, cv2.bitwise_not(exclude_mask))
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, k)
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, k)

    n, lbl, stats, _ = cv2.connectedComponentsWithStats(th, 8)
    ordinary = np.zeros((h, w), np.uint8)
    thin = np.zeros((h, w), np.uint8)
    sol, areas, thin_a, ord_a = [], [], 0, 0
    for i in range(1, n):
        a = stats[i, cv2.CC_STAT_AREA]
        if a < SULFIDE_MIN_AREA:
            continue
        comp = (lbl == i).astype(np.uint8) * 255
        cnts, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            continue
        c = max(cnts, key=cv2.contourArea)
        ha = cv2.contourArea(cv2.convexHull(c))
        s = cv2.contourArea(c) / ha if ha > 0 else 0
        sol.append(s);
        areas.append(a)
        if s < SOLIDITY_THRESHOLD:
            thin[lbl == i] = 255;
            thin_a += a
        else:
            ordinary[lbl == i] = 255;
            ord_a += a

    tot = thin_a + ord_a
    features = {
        "thin%": thin_a / tot * 100 if tot > 0 else 0,
        "n_incl": len(sol),
        "mean_area": float(np.mean(areas)) if areas else 0,
        "median_area": float(np.median(areas)) if areas else 0,
        "sulfide%": tot / total * 100,
    }
    masks = {"ordinary": ordinary, "thin": thin,
             "sulfide%": tot / total * 100,
             "ordinary%": ord_a / total * 100, "thin%": thin_a / total * 100,
             "n_incl": len(sol)}
    return features, masks


# ---------- АНАЛИЗАТОР ----------
class ShlifAnalyzer:
    def __init__(self, model_path=CNN_PATH):
        self.model = None
        if os.path.exists(model_path):
            self.model = models.resnet18(weights=None)
            self.model.fc = nn.Linear(self.model.fc.in_features, len(CLASS_NAMES))
            self.model.load_state_dict(torch.load(model_path, map_location=DEVICE))
            self.model.to(DEVICE).eval()

    def _classify_cnn(self, image_bgr):
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        x = _CNN_TF(rgb).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            probs = torch.softmax(self.model(x), 1)[0].cpu().numpy()
        idx = int(probs.argmax())
        prob_dict = {CLASS_NAMES[i]: round(float(probs[i]), 3)
                     for i in range(len(CLASS_NAMES))}
        return CLASS_NAMES[idx], prob_dict

    def analyze(self, image_path):
        try:
            image = imread_unicode(image_path)
        except (FileNotFoundError, ValueError) as e:
            return {"verdict": "ОШИБКА", "conclusion": str(e),
                    "metrics": {}, "overlay_image": None}
        image = resize_if_large(image, 1500)
        h, w = image.shape[:2]

        # 1) тальк по синей разметке
        talc_mask, talc_pct, has_marking = detect_talc_blue(image)

        # 2) маска срастаний (исключая тальк)
        feats, masks = extract_features_and_masks(image, exclude_mask=talc_mask)

        # 3) классификация
        probs = {}
        if has_marking and talc_pct > TALC_THRESHOLD:
            verdict = "Оталькованная"
            conclusion = (f"Руда ОТАЛЬКОВАННАЯ: обнаружена размеченная область "
                          f"талька — {talc_pct:.1f}% площади (порог {TALC_THRESHOLD:.0f}%).")
        elif self.model is not None:
            verdict, probs = self._classify_cnn(image)
            conf = max(probs.values())
            base = ("тонкие срастания сульфидов" if verdict == "Труднообогатимая"
                    else "крупные компактные вкрапленники")
            note = "" if conf > 0.7 else " ⚠️ Низкая уверенность — рекомендуется проверка эксперта."
            conclusion = (f"Руда {verdict.upper()}: по структуре ({base}). "
                          f"Уверенность модели {conf * 100:.0f}%.{note}")
        else:
            verdict = "Труднообогатимая" if feats["mean_area"] < 1170 else "Рядовая"
            conclusion = f"Руда {verdict.upper()} (базовое правило, модель не загружена)."

        overlay = self._overlay(image, talc_mask, masks["ordinary"], masks["thin"])
        metrics = {
            "verdict": verdict,
            "confidence": round(max(probs.values()), 3) if probs else None,
            "probabilities": probs,
            "talc_percent": round(talc_pct, 2),
            "has_talc_marking": has_marking,
            "sulfide_percent": round(masks["sulfide%"], 2),
            "thin_percent_of_sulfides": round(feats["thin%"], 2),
            "n_inclusions": feats["n_incl"],
            "mean_inclusion_area": round(feats["mean_area"], 1),
            "image_size": f"{w}x{h}",
        }
        return {"verdict": verdict, "conclusion": conclusion,
                "metrics": metrics, "overlay_image": overlay}

    @staticmethod
    def _overlay(image, talc, ordinary, thin):
        out = image.copy()
        color = np.zeros_like(image)
        color[talc > 0] = COLOR_TALC
        color[ordinary > 0] = COLOR_ORDINARY
        color[thin > 0] = COLOR_THIN
        m = (talc > 0) | (ordinary > 0) | (thin > 0)
        blended = cv2.addWeighted(image, 1 - OVERLAY_ALPHA, color, OVERLAY_ALPHA, 0)
        out[m] = blended[m]
        return out


if __name__ == "__main__":
    import sys

    path = sys.argv[1] if len(sys.argv) > 1 else "test.jpg"
    a = ShlifAnalyzer()
    r = a.analyze(path)
    print("ВЕРДИКТ:", r["verdict"])
    print("ЗАКЛЮЧЕНИЕ:", r["conclusion"])
    print("МЕТРИКИ:")
    for k, v in r["metrics"].items():
        print(f"  {k:28s}: {v}")
    if r["overlay_image"] is not None:
        imwrite_unicode("overlay_result.png", r["overlay_image"])
        print("Overlay: overlay_result.png")
