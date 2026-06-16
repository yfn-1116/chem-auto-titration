"""全面诊断摄像头参数状态 v3 — 只读不修改，不使用DSHOW"""

import cv2
import numpy as np
import time

CAMERA_INDEX = 1

PROPS = [
    ("BRIGHTNESS", cv2.CAP_PROP_BRIGHTNESS),
    ("CONTRAST", cv2.CAP_PROP_CONTRAST),
    ("SATURATION", cv2.CAP_PROP_SATURATION),
    ("HUE", cv2.CAP_PROP_HUE),
    ("GAIN", cv2.CAP_PROP_GAIN),
    ("EXPOSURE", cv2.CAP_PROP_EXPOSURE),
    ("WHITE_BALANCE_BLUE_U", cv2.CAP_PROP_WHITE_BALANCE_BLUE_U),
    ("WHITE_BALANCE_RED_V", cv2.CAP_PROP_WHITE_BALANCE_RED_V),
    ("SHARPNESS", cv2.CAP_PROP_SHARPNESS),
    ("AUTO_WB", cv2.CAP_PROP_AUTO_WB),
    ("AUTO_EXPOSURE", cv2.CAP_PROP_AUTO_EXPOSURE),
    ("TEMPERATURE", cv2.CAP_PROP_TEMPERATURE),
    ("BACKLIGHT", cv2.CAP_PROP_BACKLIGHT),
    ("FPS", cv2.CAP_PROP_FPS),
    ("FRAME_WIDTH", cv2.CAP_PROP_FRAME_WIDTH),
    ("FRAME_HEIGHT", cv2.CAP_PROP_FRAME_HEIGHT),
]

def read_frame_stats(cap, label):
    ret, frame = cap.read()
    if not ret or frame is None:
        print(f"  [{label}] 无法读取画面")
        return None
    h, w = frame.shape[:2]
    print(f"  [{label}] 分辨率: {w}x{h}")
    print(f"  [{label}] 像素统计:")
    print(f"    整体:  mean={np.mean(frame):.1f}  min={np.min(frame)}  max={np.max(frame)}")
    for ch, name in enumerate(['B', 'G', 'R']):
        print(f"    {name}通道:  mean={np.mean(frame[:,:,ch]):.1f}  min={np.min(frame[:,:,ch])}  max={np.max(frame[:,:,ch])}")
    return frame

def read_props(cap, label):
    print(f"\n--- [{label}] 参数 ---")
    for name, pid in PROPS:
        try:
            val = cap.get(pid)
            if val is not None:
                print(f"  {name:25s} = {val}")
        except:
            pass

print("="*60)
print("  摄像头参数全面诊断 v3")
print("  摄像头 index =", CAMERA_INDEX)
print("  使用默认后端 (非DSHOW)")
print("="*60)

# === 1. 出厂默认状态 ===
print("\n\n>>> 1. 出厂默认状态 (打开后不修改任何参数)")
cap = cv2.VideoCapture(CAMERA_INDEX)
if not cap.isOpened():
    print("  无法打开摄像头")
    exit(1)
read_props(cap, "默认")
read_frame_stats(cap, "默认")
cap.release()
time.sleep(0.5)

# === 2. 测试 AEC + AWB 全自动 ===
print("\n\n>>> 2. 恢复 AEC=ON + AWB=ON")
cap = cv2.VideoCapture(CAMERA_INDEX)
if cap.isOpened():
    cap.set(cv2.CAP_PROP_AUTO_WB, 1)
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75)
    read_props(cap, "AEC=ON AWB=ON")
    read_frame_stats(cap, "AEC=ON AWB=ON")
    cap.release()
time.sleep(0.5)

# === 3. 测试 AEC=ON 但 AWB=OFF ===
print("\n\n>>> 3. AEC=ON + AWB=OFF")
cap = cv2.VideoCapture(CAMERA_INDEX)
if cap.isOpened():
    cap.set(cv2.CAP_PROP_AUTO_WB, 0)
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75)
    read_props(cap, "AEC=ON AWB=OFF")
    read_frame_stats(cap, "AEC=ON AWB=OFF")
    cap.release()
time.sleep(0.5)

# === 4. 逐项参数扫描 ===
print("\n\n>>> 4. 单独调整曝光值")
cap = cv2.VideoCapture(CAMERA_INDEX)
if cap.isOpened():
    for exp in range(-14, 5, 2):
        cap.set(cv2.CAP_PROP_EXPOSURE, exp)
        time.sleep(0.1)
        ret, frame = cap.read()
        if ret:
            print(f"  EXPOSURE={exp:3d}  mean={np.mean(frame):.1f}  min={np.min(frame)}  max={np.max(frame)}")
    cap.release()
time.sleep(0.5)

print("\n>>> 5. 单独调整增益(Gain)")
cap = cv2.VideoCapture(CAMERA_INDEX)
if cap.isOpened():
    for gain in [64, 48, 32, 16, 8, 4, 2, 1]:
        cap.set(cv2.CAP_PROP_GAIN, gain)
        time.sleep(0.1)
        ret, frame = cap.read()
        if ret:
            print(f"  GAIN={gain:3d}  mean={np.mean(frame):.1f}")
    cap.release()
time.sleep(0.5)

print("\n>>> 6. 单独调整亮度")
cap = cv2.VideoCapture(CAMERA_INDEX)
if cap.isOpened():
    for bright in range(0, -201, -40):
        cap.set(cv2.CAP_PROP_BRIGHTNESS, bright)
        cap.set(cv2.CAP_PROP_EXPOSURE, -6)
        time.sleep(0.1)
        ret, frame = cap.read()
        if ret:
            print(f"  BRIGHTNESS={bright:4d}  mean={np.mean(frame):.1f}")
    cap.release()
time.sleep(0.5)

print("\n>>> 7. 单独降低饱和度")
cap = cv2.VideoCapture(CAMERA_INDEX)
if cap.isOpened():
    for sat in [64, 32, 16, 8, 0]:
        cap.set(cv2.CAP_PROP_SATURATION, sat)
        time.sleep(0.1)
        ret, frame = cap.read()
        if ret:
            print(f"  SATURATION={sat:3d}  mean={np.mean(frame):.1f}")
    cap.release()
time.sleep(0.5)

# === 8. 连续帧稳定性 ===
print("\n\n>>> 8. 连续30帧稳定性测试")
cap = cv2.VideoCapture(CAMERA_INDEX)
if cap.isOpened():
    for i in range(30):
        ret, frame = cap.read()
        if ret:
            m = np.mean(frame)
            if i < 5 or i >= 25:
                print(f"  第{i+1:2d}帧: mean={m:.1f}")
    cap.release()

print("\n\n诊断完毕，所有信息已采集，未做任何修改")
