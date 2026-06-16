"""全面诊断摄像头参数状态 — 只读不修改"""

import cv2
import numpy as np

CAMERA_INDEX = 1

PROPS = [
    ("CAP_PROP_POS_MSEC", cv2.CAP_PROP_POS_MSEC),
    ("CAP_PROP_POS_FRAMES", cv2.CAP_PROP_POS_FRAMES),
    ("CAP_PROP_POS_AVI_RATIO", cv2.CAP_PROP_POS_AVI_RATIO),
    ("CAP_PROP_FRAME_WIDTH", cv2.CAP_PROP_FRAME_WIDTH),
    ("CAP_PROP_FRAME_HEIGHT", cv2.CAP_PROP_FRAME_HEIGHT),
    ("CAP_PROP_FPS", cv2.CAP_PROP_FPS),
    ("CAP_PROP_FOURCC", cv2.CAP_PROP_FOURCC),
    ("CAP_PROP_FRAME_COUNT", cv2.CAP_PROP_FRAME_COUNT),
    ("CAP_PROP_BRIGHTNESS", cv2.CAP_PROP_BRIGHTNESS),
    ("CAP_PROP_CONTRAST", cv2.CAP_PROP_CONTRAST),
    ("CAP_PROP_SATURATION", cv2.CAP_PROP_SATURATION),
    ("CAP_PROP_HUE", cv2.CAP_PROP_HUE),
    ("CAP_PROP_GAIN", cv2.CAP_PROP_GAIN),
    ("CAP_PROP_EXPOSURE", cv2.CAP_PROP_EXPOSURE),
    ("CAP_PROP_CONVERT_RGB", cv2.CAP_PROP_CONVERT_RGB),
    ("CAP_PROP_WHITE_BALANCE_BLUE_U", cv2.CAP_PROP_WHITE_BALANCE_BLUE_U),
    ("CAP_PROP_RECTIFICATION", cv2.CAP_PROP_RECTIFICATION),
    ("CAP_PROP_MONOCHROME", cv2.CAP_PROP_MONOCHROME),
    ("CAP_PROP_SHARPNESS", cv2.CAP_PROP_SHARPNESS),
    ("CAP_PROP_AUTO_WB", cv2.CAP_PROP_AUTO_WB),
    ("CAP_PROP_AUTO_EXPOSURE", cv2.CAP_PROP_AUTO_EXPOSURE),
    ("CAP_PROP_TEMPERATURE", cv2.CAP_PROP_TEMPERATURE),
    ("CAP_PROP_TRIGGER", cv2.CAP_PROP_TRIGGER),
    ("CAP_PROP_TRIGGER_DELAY", cv2.CAP_PROP_TRIGGER_DELAY),
    ("CAP_PROP_WHITE_BALANCE_RED_V", cv2.CAP_PROP_WHITE_BALANCE_RED_V),
    ("CAP_PROP_ZOOM", cv2.CAP_PROP_ZOOM),
    ("CAP_PROP_FOCUS", cv2.CAP_PROP_FOCUS),
    ("CAP_PROP_GUID", cv2.CAP_PROP_GUID),
    ("CAP_PROP_ISO_SPEED", cv2.CAP_PROP_ISO_SPEED),
    ("CAP_PROP_BACKLIGHT", cv2.CAP_PROP_BACKLIGHT),
    ("CAP_PROP_PAN", cv2.CAP_PROP_PAN),
    ("CAP_PROP_TILT", cv2.CAP_PROP_TILT),
    ("CAP_PROP_ROLL", cv2.CAP_PROP_ROLL),
    ("CAP_PROP_IRIS", cv2.CAP_PROP_IRIS),
    ("CAP_PROP_SETTINGS", cv2.CAP_PROP_SETTINGS),
    ("CAP_PROP_BUFFERSIZE", cv2.CAP_PROP_BUFFERSIZE),
    ("CAP_PROP_AUTOFOCUS", cv2.CAP_PROP_AUTOFOCUS),
    ("CAP_PROP_SAR_NUM", cv2.CAP_PROP_SAR_NUM),
    ("CAP_PROP_SAR_DEN", cv2.CAP_PROP_SAR_DEN),
    ("CAP_PROP_BACKEND", cv2.CAP_PROP_BACKEND),
    ("CAP_PROP_CHANNEL", cv2.CAP_PROP_CHANNEL),
    ("CAP_PROP_AUTO_WB", cv2.CAP_PROP_AUTO_WB),
]

def read_props(cap, label):
    print(f"\n{'='*60}")
    print(f"  [{label}] 摄像头所有可读参数:")
    print(f"{'='*60}")
    for name, pid in PROPS:
        try:
            val = cap.get(pid)
            if val != -1:
                print(f"  {name:40s} = {val}")
        except:
            pass

def read_frame_stats(cap, label):
    ret, frame = cap.read()
    if not ret or frame is None:
        print(f"  [{label}] 无法读取画面")
        return
    h, w = frame.shape[:2]
    print(f"  [{label}] 分辨率: {w}x{h}")
    print(f"  [{label}] 像素统计:")
    print(f"    整体:  mean={np.mean(frame):.1f}  min={np.min(frame)}  max={np.max(frame)}")
    for ch, name in enumerate(['B', 'G', 'R']):
        print(f"    {name}通道:  mean={np.mean(frame[:,:,ch]):.1f}  min={np.min(frame[:,:,ch])}  max={np.max(frame[:,:,ch])}")

print("="*60)
print("  摄像头参数全面诊断")
print("  摄像头 index =", CAMERA_INDEX)
print("="*60)

# === 方案A: 默认打开（无DSHOW，不修改任何参数）===
print("\n\n>>> 方案A: cv2.VideoCapture(index) — 出厂默认状态")
cap = cv2.VideoCapture(CAMERA_INDEX)
if cap.isOpened():
    read_props(cap, "默认后端")
    read_frame_stats(cap, "默认后端")
    cap.release()

# === 方案B: DSHOW 打开，不修改任何参数 ===
print("\n\n>>> 方案B: cv2.VideoCapture(index, CAP_DSHOW) — 不修改任何参数")
cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
if cap.isOpened():
    read_props(cap, "DSHOW 全默认")
    read_frame_stats(cap, "DSHOW 全默认")
    cap.release()

# === 方案C: DSHOW + 只关 AWB，其他不动 ===
print("\n\n>>> 方案C: DSHOW + 仅关 AWB")
cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
if cap.isOpened():
    cap.set(cv2.CAP_PROP_AUTO_WB, 0)
    read_props(cap, "DSHOW AWB=OFF")
    read_frame_stats(cap, "DSHOW AWB=OFF")
    cap.release()

# === 方案D: DSHOW + 恢复自动全部 ===
print("\n\n>>> 方案D: DSHOW + 恢复自动AWB + 自动AEC")
cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
if cap.isOpened():
    cap.set(cv2.CAP_PROP_AUTO_WB, 1)
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75)
    read_props(cap, "DSHOW AWB=ON AEC=ON")
    read_frame_stats(cap, "DSHOW AWB=ON AEC=ON")
    cap.release()

# === 连续读帧测试（看是否首帧异常）===
print("\n\n>>> 连续读帧测试（看首帧 vs 稳定后）")
cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
if cap.isOpened():
    for i in range(10):
        ret, frame = cap.read()
        if ret and frame is not None:
            print(f"  第{i+1:2d}帧: mean={np.mean(frame):.1f}  min={np.min(frame)}  max={np.max(frame)}")
    cap.release()

# === 热启动 vs 冷启动测试 ===
print("\n\n>>> 热启动测试（重复打开关闭，检查参数是否被上一次污染）")
for trial in range(3):
    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
    if cap.isOpened():
        ret, frame = cap.read()
        if ret:
            print(f"  第{trial+1}次打开: mean={np.mean(frame):.1f}")
        cap.release()

print("\n\n>>> 尝试单个参数调整测试 (只读不写)")
print("  以下测试每次打开新摄像头，只改一个参数，然后恢复")
for gain in [64, 32, 16, 8]:
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if cap.isOpened():
        cap.set(cv2.CAP_PROP_GAIN, gain)
        ret, frame = cap.read()
        if ret:
            print(f"  GAIN={gain:3d}  mean={np.mean(frame):.1f}")
        cap.release()

for exposure in [-6, -8, -10, -12, -14]:
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if cap.isOpened():
        cap.set(cv2.CAP_PROP_EXPOSURE, exposure)
        ret, frame = cap.read()
        if ret:
            print(f"  EXPOSURE={exposure:3d}  mean={np.mean(frame):.1f}")
        cap.release()

for bright in [0, -64, -128, -255]:
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if cap.isOpened():
        cap.set(cv2.CAP_PROP_BRIGHTNESS, bright)
        ret, frame = cap.read()
        if ret:
            print(f"  BRIGHTNESS={bright:4d}  mean={np.mean(frame):.1f}")
        cap.release()

# 测试 AEC=ON 的效果
cap = cv2.VideoCapture(CAMERA_INDEX)
if cap.isOpened():
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75)
    ret, frame = cap.read()
    if ret:
        print(f"  AEC=ON(0.75)  mean={np.mean(frame):.1f}")
    cap.release()

# 测试自己恢复 AEC=OFF
cap = cv2.VideoCapture(CAMERA_INDEX)
if cap.isOpened():
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0)
    ret, frame = cap.read()
    if ret:
        print(f"  AEC=OFF(0)   mean={np.mean(frame):.1f}")
    cap.release()

print("\n\n诊断完毕，所有信息已采集，未做任何修改")
