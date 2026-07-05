"""Инференс U-Net: фото → heatmap талька. Проверка на нескольких фото."""
import os, glob, cv2, numpy as np, torch, torch.nn as nn

DEVICE = "cuda"
IMG_SIZE = 512   # ВАЖНО: то же, что при обучении! (если менял на 512)
CKPT = os.path.expanduser("~/talc_unet.pth")
DATA = os.path.expanduser("~/talc_dataset")

# --- та же архитектура ---
def conv_block(ci, co):
    return nn.Sequential(
        nn.Conv2d(ci, co, 3, padding=1), nn.BatchNorm2d(co), nn.ReLU(inplace=True),
        nn.Conv2d(co, co, 3, padding=1), nn.BatchNorm2d(co), nn.ReLU(inplace=True))
class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.d1=conv_block(3,32); self.d2=conv_block(32,64)
        self.d3=conv_block(64,128); self.d4=conv_block(128,256)
        self.pool=nn.MaxPool2d(2); self.bott=conv_block(256,512)
        self.up4=nn.ConvTranspose2d(512,256,2,2); self.u4=conv_block(512,256)
        self.up3=nn.ConvTranspose2d(256,128,2,2); self.u3=conv_block(256,128)
        self.up2=nn.ConvTranspose2d(128,64,2,2); self.u2=conv_block(128,64)
        self.up1=nn.ConvTranspose2d(64,32,2,2); self.u1=conv_block(64,32)
        self.out=nn.Conv2d(32,1,1)
    def forward(self,x):
        c1=self.d1(x); c2=self.d2(self.pool(c1))
        c3=self.d3(self.pool(c2)); c4=self.d4(self.pool(c3))
        b=self.bott(self.pool(c4))
        x=self.u4(torch.cat([self.up4(b),c4],1))
        x=self.u3(torch.cat([self.up3(x),c3],1))
        x=self.u2(torch.cat([self.up2(x),c2],1))
        x=self.u1(torch.cat([self.up1(x),c1],1))
        return self.out(x)

model = UNet().to(DEVICE)
model.load_state_dict(torch.load(CKPT, map_location=DEVICE))
model.eval()

def predict_heatmap(img_bgr):
    h, w = img_bgr.shape[:2]
    inp = cv2.resize(img_bgr, (IMG_SIZE, IMG_SIZE))
    inp = cv2.cvtColor(inp, cv2.COLOR_BGR2RGB).astype(np.float32)/255.
    inp = (inp - [0.485,0.456,0.406]) / [0.229,0.224,0.225]
    t = torch.from_numpy(inp.transpose(2,0,1)).float().unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        prob = torch.sigmoid(model(t))[0,0].cpu().numpy()
    return cv2.resize(prob, (w, h))   # 0..1 карта вероятности талька

# тест на первых 4 фото → сохраняем визуализации
os.makedirs(os.path.expanduser("~/talc_test"), exist_ok=True)
for p in sorted(glob.glob(f"{DATA}/images/*.png"))[:4]:
    img = cv2.imread(p)
    heat = predict_heatmap(img)
    pct = (heat > 0.5).mean() * 100
    # цветная heatmap поверх фото
    hm = cv2.applyColorMap((heat*255).astype(np.uint8), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(img, 0.6, hm, 0.4, 0)
    name = os.path.basename(p)
    out = os.path.expanduser(f"~/talc_test/{name}")
    cv2.imwrite(out, np.hstack([img, overlay]))
    print(f"{name}: талька {pct:.1f}% → {out}", flush=True)
print("Готово. Скачай ~/talc_test/*.png и посмотри.")
