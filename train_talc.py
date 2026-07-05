"""U-Net сегментация талька. GPU (NVIDIA L4). 42 пары + аугментация."""
import os, glob, cv2, numpy as np
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

DATA = os.path.expanduser("~/talc_dataset")
DEVICE = "cuda"
IMG_SIZE = 512
EPOCHS = 200
BS = 2
LR = 1e-3
CKPT = os.path.expanduser("~/talc_unet.pth")

# ---------- данные ----------
imgs = sorted(glob.glob(f"{DATA}/images/*.png"))
msks = sorted(glob.glob(f"{DATA}/masks/*.png"))
assert len(imgs) == len(msks) and len(imgs) > 0, f"imgs={len(imgs)} msks={len(msks)}"
print(f"Пар: {len(imgs)}", flush=True)

# проверим, что маски не пустые (тальк есть)
nonempty = sum(1 for m in msks if (cv2.imread(m, 0) > 127).any())
print(f"Масок с тальком (непустых): {nonempty}/{len(msks)}", flush=True)

tr_i, va_i, tr_m, va_m = train_test_split(imgs, msks, test_size=0.2, random_state=42)
print(f"train={len(tr_i)} val={len(va_i)}", flush=True)

class TalcDS(Dataset):
    def __init__(self, imgs, msks, train=True):
        self.imgs, self.msks, self.train = imgs, msks, train
    def __len__(self): return len(self.imgs)
    def __getitem__(self, i):
        img = cv2.imread(self.imgs[i])
        msk = cv2.imread(self.msks[i], 0)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        msk = cv2.resize(msk, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_NEAREST)
        if self.train:
            if np.random.rand() > 0.5: img, msk = img[:, ::-1], msk[:, ::-1]
            if np.random.rand() > 0.5: img, msk = img[::-1], msk[::-1]
            k = np.random.randint(4)
            img, msk = np.rot90(img, k).copy(), np.rot90(msk, k).copy()
            # лёгкая яркость
            if np.random.rand() > 0.5:
                f = np.random.uniform(0.8, 1.2)
                img = np.clip(img.astype(np.float32) * f, 0, 255).astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.
        img = (img - [0.485,0.456,0.406]) / [0.229,0.224,0.225]
        img = torch.from_numpy(img.transpose(2,0,1)).float()
        msk = torch.from_numpy((msk > 127).astype(np.float32)).unsqueeze(0)
        return img, msk

tr_dl = DataLoader(TalcDS(tr_i, tr_m, True),  batch_size=BS, shuffle=True, num_workers=2)
va_dl = DataLoader(TalcDS(va_i, va_m, False), batch_size=BS, num_workers=2)

# ---------- U-Net ----------
def conv_block(ci, co):
    return nn.Sequential(
        nn.Conv2d(ci, co, 3, padding=1), nn.BatchNorm2d(co), nn.ReLU(inplace=True),
        nn.Conv2d(co, co, 3, padding=1), nn.BatchNorm2d(co), nn.ReLU(inplace=True))

class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.d1 = conv_block(3, 32);   self.d2 = conv_block(32, 64)
        self.d3 = conv_block(64, 128); self.d4 = conv_block(128, 256)
        self.pool = nn.MaxPool2d(2)
        self.bott = conv_block(256, 512)
        self.up4 = nn.ConvTranspose2d(512,256,2,2); self.u4 = conv_block(512,256)
        self.up3 = nn.ConvTranspose2d(256,128,2,2); self.u3 = conv_block(256,128)
        self.up2 = nn.ConvTranspose2d(128,64,2,2);  self.u2 = conv_block(128,64)
        self.up1 = nn.ConvTranspose2d(64,32,2,2);   self.u1 = conv_block(64,32)
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

model = UNet().to(DEVICE)

# ---------- loss + метрика ----------
def dice_loss(pred, tgt, eps=1):
    pred = torch.sigmoid(pred)
    inter = (pred*tgt).sum((2,3))
    return (1 - (2*inter+eps)/(pred.sum((2,3))+tgt.sum((2,3))+eps)).mean()

def dice_metric(pred, tgt, eps=1):
    pred = (torch.sigmoid(pred) > 0.5).float()
    inter = (pred*tgt).sum((2,3))
    return ((2*inter+eps)/(pred.sum((2,3))+tgt.sum((2,3))+eps)).mean().item()

bce = nn.BCEWithLogitsLoss()
opt = torch.optim.Adam(model.parameters(), lr=LR)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS)

# ---------- обучение ----------
best = 0
for ep in range(EPOCHS):
    model.train(); tr_loss = 0
    for x, y in tr_dl:
        x, y = x.to(DEVICE), y.to(DEVICE)
        opt.zero_grad()
        out = model(x)
        loss = bce(out, y) + dice_loss(out, y)
        loss.backward(); opt.step()
        tr_loss += loss.item()
    sched.step()
    model.eval(); dices = []
    with torch.no_grad():
        for x, y in va_dl:
            out = model(x.to(DEVICE))
            dices.append(dice_metric(out, y.to(DEVICE)))
    d = float(np.mean(dices))
    mark = ""
    if d > best:
        best = d
        torch.save(model.state_dict(), CKPT)
        mark = " ← saved"
    print(f"Эпоха {ep+1:2d}/{EPOCHS}: loss={tr_loss/len(tr_dl):.3f} val Dice={d:.3f}{mark}", flush=True)

print(f"\nЛучший Dice: {best:.3f} → {CKPT}", flush=True)
