"""摄像头实时预览 + 模型预测 — 调参/观察两用
按键:
  e/E = 曝光 +/-     b/B = 亮度 +/-     c/C = 对比度 +/-
  w = 切换白平衡      a = 切换自动曝光
  s = 保存截图        q = 退出
"""

import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
import json
import os
import time

from model import resnet34

# ===== 配置 =====
CAMERA_INDEX = 1
MODEL_NAME = "EBT"
N_CLASSES = 2
USE_CAP_DSHOW = os.name == "nt"

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 加载模型
with open(f"pths/{MODEL_NAME}.json") as f:
    class_indict = json.load(f)

model = resnet34(num_classes=N_CLASSES).to(device)
model.load_state_dict(torch.load(f"pths/{MODEL_NAME}Net.pth", map_location=device))
model.eval()

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# ===== 打开摄像头 =====
backend = cv2.CAP_DSHOW if USE_CAP_DSHOW else cv2.CAP_ANY
cap = cv2.VideoCapture(CAMERA_INDEX, backend)
if not cap.isOpened():
    print(f"无法打开摄像头 {CAMERA_INDEX}")
    exit(1)

# 先用出厂默认（不设任何参数），摄像头自动调节
# 快捷键可随时手动开关自动/调参
cap.set(cv2.CAP_PROP_AUTO_WB, 1)
cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75)

awb_on = True
aec_on = True
exp_val = 0
bright_val = 0
contrast_val = 0

predict_every = 3
frame_count = 0
pred_class = "---"
pred_prob = 0.0
all_probs = [0.0] * N_CLASSES

window_name = f"Camera {CAMERA_INDEX} — {MODEL_NAME} 实时预测"
cv2.namedWindow(window_name)

print(f"摄像头 {CAMERA_INDEX} 已打开 | 模型: {MODEL_NAME} | 类别: {class_indict}")
print("按键: e/E=曝光  b/B=亮度  c/C=对比度  w=白平衡  a=自动曝光  s=截图  q=退出")

while True:
    ret, frame = cap.read()
    if not ret:
        print("无法读取画面")
        break

    frame_count += 1
    h, w = frame.shape[:2]

    # 模型预测（每隔 predict_every 帧）
    if frame_count % predict_every == 0:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = transform(Image.fromarray(rgb)).unsqueeze(0).to(device)
        with torch.no_grad():
            out = torch.softmax(model(img).squeeze(), dim=0).tolist()
        pred_idx = max(range(len(out)), key=lambda i: out[i])
        pred_class = class_indict[str(pred_idx)]
        pred_prob = out[pred_idx]
        all_probs = out

    # 中心 ROI 颜色
    cx, cy = w // 2, h // 2
    roi = frame[cy-25:cy+25, cx-25:cx+25]
    avg_bgr = np.mean(roi, axis=(0, 1)).astype(int)
    b, g, r = avg_bgr[0], avg_bgr[1], avg_bgr[2]

    # 中心标记
    cv2.rectangle(frame, (cx-25, cy-25), (cx+25, cy+25), (0, 255, 255), 2)
    cv2.line(frame, (cx-10, cy), (cx+10, cy), (0, 255, 255), 1)
    cv2.line(frame, (cx, cy-10), (cx, cy+10), (0, 255, 255), 1)

    # 信息叠加
    lines = [
        f"Model: {pred_class}  ({pred_prob*100:.1f}%)",
        f"RGB: ({r}, {g}, {b})",
        f"EXP={exp_val}  BRIGHT={bright_val}  CONT={contrast_val}",
        f"AWB={'ON' if awb_on else 'OFF'}  AEC={'ON' if aec_on else 'OFF'}",
    ]
    for i, txt in enumerate(lines):
        cv2.putText(frame, txt, (10, 30 + i * 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1)

    # 各类别概率
    y0 = 130
    for k, v in class_indict.items():
        pct = all_probs[int(k)] * 100
        color = (0, 255, 0) if class_indict[k] == pred_class else (200, 200, 200)
        cv2.putText(frame, f"  {v}: {pct:.1f}%", (10, y0),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        y0 += 20

    # 底部操作提示
    cv2.putText(frame, "e/E=曝光  b/B=亮度  c/C=对比度  w=白平衡  a=自动曝光  s=截图  q=退出",
                (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)

    cv2.imshow(window_name, frame)

    key = cv2.waitKey(30) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('s'):
        ts = time.strftime("%Y%m%d_%H%M%S")
        os.makedirs("Output", exist_ok=True)
        path = f"Output/preview_{ts}.jpg"
        cv2.imwrite(path, frame)
        print(f"截图保存: {path}")
    elif key == ord('e'):
        exp_val = min(exp_val + 1, 20)
        cap.set(cv2.CAP_PROP_EXPOSURE, exp_val)
        print(f"曝光: {exp_val}")
    elif key == ord('E'):
        exp_val = max(exp_val - 1, -10)
        cap.set(cv2.CAP_PROP_EXPOSURE, exp_val)
        print(f"曝光: {exp_val}")
    elif key == ord('b'):
        bright_val = min(bright_val + 10, 255)
        cap.set(cv2.CAP_PROP_BRIGHTNESS, bright_val)
        print(f"亮度: {bright_val}")
    elif key == ord('B'):
        bright_val = max(bright_val - 10, -255)
        cap.set(cv2.CAP_PROP_BRIGHTNESS, bright_val)
        print(f"亮度: {bright_val}")
    elif key == ord('c'):
        contrast_val = min(contrast_val + 5, 255)
        cap.set(cv2.CAP_PROP_CONTRAST, contrast_val)
        print(f"对比度: {contrast_val}")
    elif key == ord('C'):
        contrast_val = max(contrast_val - 5, 0)
        cap.set(cv2.CAP_PROP_CONTRAST, contrast_val)
        print(f"对比度: {contrast_val}")
    elif key == ord('w'):
        awb_on = not awb_on
        cap.set(cv2.CAP_PROP_AUTO_WB, 1 if awb_on else 0)
        print(f"白平衡: {'ON' if awb_on else 'OFF'}")
    elif key == ord('a'):
        aec_on = not aec_on
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75 if aec_on else 0)
        print(f"自动曝光: {'ON' if aec_on else 'OFF'}")

cap.release()
cv2.destroyAllWindows()
