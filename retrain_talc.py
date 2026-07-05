"""
Дообучение (transfer learning) U-Net на новых данных.
Стартует с текущих весов, сохраняет ТОЛЬКО если Dice вырос.
Защита от плохой разметки пользователя.

Запуск на сервере T4:
    python retrain_talc.py --new-data active_learning_dataset --epochs 30
"""
import os, glob, cv2, argparse, shutil, datetime
import numpy as np
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE = 512
BS = 2
LR = 1e-4  # МЕНЬШЕ чем при обучении с нуля (1e-3) — transfer learning

# ---------- U-Net (та же архитектура!) ----------
def conv_block(ci, co):
    return nn.Sequential(
        nn.Conv2d(ci, co, 3, padding=1), nn.BatchNorm2d(co), nn.ReLU(inplace=True),
        nn.Conv2d(co, co, 3, padding=1), nn.BatchNorm2d(co), nn.ReLU(inplace=True))

class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.d1 = conv_block(3, 32); self.d2 = conv_block(32, 64)
        self.d3 = conv_block(64, 128); self.d4 = conv_block(128, 256)
        self.pool = nn.MaxPool2d(2)
        self.bott = conv_block(256, 512)
        self.up4 = nn.ConvTranspose2d(512,256,2,2); self.u4 = conv_block(512,256)
        self.up3 = nn.ConvTranspose2d(256,128,2,2); self.u3 = conv_block(256,128)
        self.up2 = nn.ConvTranspose2d(128,64,2,2); self.u2 = conv_block(128,64)
        self.up1 = nn.ConvTranspose2d(64,32,2,2); self.u1 = conv_block(64,32)
        self.out = nn.Conv2d(32, 1, 1)
    def forward(self, x):
        c1 = self.d1(x); c2 = self.d2(self.pool(c1))
        c3 = self.d3(self.pool(c2)); c4 = self.d4(self.pool(c3))
        b = self.bott(self.pool(c4))
        x = self.u4(torch.cat([self.up4(b), c4],1))
        x = self.u3(torch.cat([self.up3(x), c3],1))
        x = self.u2(torch.cat([self.up2(x), c2],1))
        x = self.u1(torch.cat([self.up1(x), c1],1))
        return self.out(x)

class TalcDS(Dataset):
    def __init__(self, imgs, msks, train=True):
        self.imgs, self.msks, self.train = imgs, msks, train
    def __len__(self): return len(self.imgs)
    def __getitem__(self, i):
        img = cv2.imread(self.imgs[i]); msk = cv2.imread(self.msks[i], 0)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        msk = cv2.resize(msk, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_NEAREST)
        if self.train:
            if np.random.rand() > 0.5: img, msk = img[:, ::-1], msk[:, ::-1]
            if np.random.rand() > 0.5: img, msk = img[::-1], msk[::-1]
            k = np.random.randint(4)
            img, msk = np.rot90(img, k).copy(), np.rot90(msk, k).copy()
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.
        img = (img - [0.485,0.456,0.406]) / [0.229,0.224,0.225]
        img = torch.from_numpy(img.transpose(2,0,1)).float()
        msk = torch.from_numpy((msk > 127).astype(np.float32)).unsqueeze(0)
        return img, msk

def dice_loss(pred, tgt, eps=1):
    pred = torch.sigmoid(pred); inter = (pred*tgt).sum((2,3))
    return (1 - (2*inter+eps)/(pred.sum((2,3))+tgt.sum((2,3))+eps)).mean()

def dice_metric(pred, tgt, eps=1):
    pred = (torch.sigmoid(pred) > 0.5).float(); inter = (pred*tgt).sum((2,3))
    return ((2*inter+eps)/(pred.sum((2,3))+tgt.sum((2,3))+eps)).mean().item()

def collect_pairs(base_data, new_data):
    """Собираем пары (фото+маска) из исходного датасета + новые от эксперта."""
    imgs, msks = [], []
    # исходный датасет
    if base_data and os.path.exists(base_data):
        bi = sorted(glob.glob(f"{base_data}/images/*.png"))
        bm = sorted(glob.glob(f"{base_data}/masks/*.png"))
        imgs += bi; msks += bm
        print(f"Исходный датасет: {len(bi)} пар")
    # новые данные от эксперта (image_X.png + mask_talc_X.png)
    if new_data and os.path.exists(new_data):
        new_imgs = sorted(glob.glob(f"{new_data}/image_*.png"))
        cnt = 0
        for ip in new_imgs:
            name = os.path.basename(ip).replace("image_", "", 1)
            mp = os.path.join(new_data, f"mask_talc_{name}")
            if os.path.exists(mp):
                # берём только НЕПУСТЫЕ маски (защита от каракулей без талька)
                if (cv2.imread(mp, 0) > 127).any():
                    imgs.append(ip); msks.append(mp); cnt += 1
        print(f"Новые валидные пары от эксперта: {cnt}")
    return imgs, msks

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-data", default=os.path.expanduser("~/talc_dataset"))
    ap.add_argument("--new-data", default="active_learning_dataset")
    ap.add_argument("--ckpt", default=os.path.expanduser("~/talc_unet.pth"))
    ap.add_argument("--epochs", type=int, default=30)
    args = ap.parse_args()

    print(f"=== ДООБУЧЕНИЕ U-NET (transfer learning) ===")
    print(f"Device: {DEVICE}")

    imgs, msks = collect_pairs(args.base_data, args.new_data)
    assert len(imgs) > 4, f"Слишком мало данных: {len(imgs)}"
    print(f"Всего пар: {len(imgs)}")

    tr_i, va_i, tr_m, va_m = train_test_split(imgs, msks, test_size=0.2, random_state=42)
    tr_dl = DataLoader(TalcDS(tr_i, tr_m, True), batch_size=BS, shuffle=True)
    va_dl = DataLoader(TalcDS(va_i, va_m, False), batch_size=BS)

    # 1. Загружаем ТЕКУЩИЕ веса (transfer learning, НЕ с нуля)
    model = UNet().to(DEVICE)
    model.load_state_dict(torch.load(args.ckpt, map_location=DEVICE, weights_only=True))
    print(f"Загружены текущие веса: {args.ckpt}")

    # 2. Baseline Dice ДО дообучения
    def eval_dice():
        model.eval(); ds = []
        with torch.no_grad():
            for x, y in va_dl:
                ds.append(dice_metric(model(x.to(DEVICE)), y.to(DEVICE)))
        return float(np.mean(ds))

    baseline = eval_dice()
    print(f"Baseline Dice (до дообучения): {baseline:.3f}")

    # 3. Дообучение
    bce = nn.BCEWithLogitsLoss()
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    best = baseline
    best_state = {k: v.clone() for k, v in model.state_dict().items()}

    for ep in range(args.epochs):
        model.train(); tl = 0
        for x, y in tr_dl:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            out = model(x)
            loss = bce(out, y) + dice_loss(out, y)
            loss.backward(); opt.step(); tl += loss.item()
        d = eval_dice()
        mark = ""
        if d > best:
            best = d
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            mark = " ← лучше!"
        print(f"Эпоха {ep+1}/{args.epochs}: loss={tl/len(tr_dl):.3f} Dice={d:.3f}{mark}")

    # 4. SAFE-SAVE: сохраняем ТОЛЬКО если стало лучше
    if best > baseline:
        # бэкап старой модели
        backup = args.ckpt.replace(".pth", f"_backup_{datetime.datetime.now():%Y%m%d_%H%M%S}.pth")
        shutil.copy(args.ckpt, backup)
        model.load_state_dict(best_state)
        torch.save(model.state_dict(), args.ckpt)
        print(f"\n✅ УЛУЧШЕНИЕ: {baseline:.3f} → {best:.3f}. Модель обновлена.")
        print(f"   Бэкап старой: {backup}")
    else:
        print(f"\n⚠️ УЛУЧШЕНИЯ НЕТ: baseline={baseline:.3f}, best={best:.3f}.")
        print(f"   Модель НЕ изменена (защита от плохих данных).")

if __name__ == "__main__":
    main()