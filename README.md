# Mlabs AI Titration System v1.5

基于计算机视觉和深度学习的自动滴定实验系统。使用 ResNet34 实时分析滴定液颜色变化，自动控制注射泵完成精确滴定。支持**视觉滴定**（颜色变化检测）和**电位滴定**（电压测量）两种模式。

---

## 功能特点

- **实时颜色识别** — ResNet34 通过 USB 摄像头实时分类溶液颜色（蓝色/紫色/酒红色）
- **三级状态机** — FAST → SLOW → STOP 三段式控制，滑动窗口平滑 + 多帧连续确认防抖动
- **注射泵自动化** — CH340 串口通信、自动补液、体积校准、舵机控制
- **电位滴定模式** — 可选电压读取，支持曲线拟合和二阶导终点分析
- **人工纠错** — 快捷键修正分类结果，保存标注图片用于模型续训
- **数据记录** — 自动保存滴定曲线图、实验 JSON 数据、终点图像
- **平台上传** — 可选集成竞赛平台（jingsai.mools.net）数据上传

---

## 项目结构

```
chem-auto-titration/
├── titration.py           # 主入口（独立模式 + --upload 平台模式）
├── model.py               # ResNet34 / ResNet50 / ResNeXt 模型定义
├── Find_COM.py            # 串口自动检测（CH340 → USB串行 → 任意串口）
├── requirements.txt       # Python 依赖
├── README.md
├── login/
│   └── .gitkeep           # 登录信息文件（已 gitignore）
└── tools/
    ├── camera_preview.py       # 摄像头实时预览 + 模型推理
    ├── camera_diagnose_full.py # 全面摄像头参数诊断
    ├── camera_diagnose_v3.py   # 摄像头诊断（非 DShow 后端）
    ├── pump_diagnose.py        # 注射泵通信测试
    ├── pump_diagnose_v2.py     # 注射泵完整协议模拟
    ├── continue_train.py       # 3分类模型续训（酒红/紫/蓝）
    └── continue_train_ebt.py   # 2分类模型续训（蓝/紫）
```

---

## 快速开始

### 环境要求

- Python 3.8+
- PyTorch 1.9+
- USB 摄像头
- 注射泵（串口通信，可选 — 无硬件也可运行）

### 安装

```bash
pip install -r requirements.txt
```

### 准备模型

将训练好的权重文件和类别映射放入 `pths/`：

```
pths/
├── EBTNet.pth              # 2分类模型（紫色/蓝色）
├── EBT.json                # 类别索引映射
├── Color_Model_Net.pth     # 3分类模型（酒红/紫/蓝）
└── Color_Model_.json       # 类别索引映射
```

### 运行

```bash
# 独立模式（无需登录平台）
python titration.py

# 完整模式（含平台登录和数据上传）
python titration.py --upload
```

---

## 工作原理

### 滴定流程

```
摄像头采集 → 图像预处理 → ResNet34 推理 → 颜色分类
                                                  ↓
                         ┌────────────────────────────────┐
                         │  状态机                         │
                         │  ┌──────┐    ┌──────┐    ┌──┐  │
                         │  │ FAST │───→│ SLOW │───→│STOP│ │
                         │  └──────┘    └──────┘    └──┘  │
                         │  (快速滴定)  (缓慢滴定)  (终点)  │
                         └────────────────────────────────┘
                                                  ↓
                                      注射泵控制
```

### 状态机逻辑

1. **FAST 阶段** — 以 `QUICK_SPEED`（如 0.3 ml/步）快速滴定。监控蓝色概率，若 `p_blue > BLUE_THRESHOLD` 持续 `CONFIRM_FRAMES` 帧，转入 STOP。若检测到紫色（过渡色），转入 SLOW。
2. **SLOW 阶段** — 以 `SLOW_SPEED`（如 0.08 ml/步）缓慢滴定。双重判定逻辑：绝对阈值判定（`p_blue > 0.70`）或相对优势判定（blue 超过 purple 0.15 且 blue > 0.30）。超过 25 轮强制 STOP。
3. **STOP 阶段** — 记录终点体积、保存终点图像，可选进行过量滴定。

### 防抖动策略

- **滑动窗口** — 对最近 `SMOOTH_WINDOW` 帧的预测结果取平均
- **连续确认** — 状态切换前需要 `CONFIRM_FRAMES` 次连续阳性检测

---

## 参数配置

在 `titration.py` → `run_titration_standalone()` 中修改：

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `CAMERA_INDEX` | 摄像头编号（0=内置，1=USB） | 1 |
| `MODEL_NAME` | `pths/` 下模型文件名前缀 | `"EBT"` |
| `N_CLASSES` | 输出类别数 | 2 |
| `TRANSITION_CLASS` | 过渡色类别名（可选） | `None` |
| `OVERDOSE_COUNT` | 终点后继续滴定的次数 | 0 |
| `VOLUME_PAR` | 体积校正系数 | 0.913 |
| `FINAL_VOLUME` | 最大目标体积（ml） | 100 |
| `TYPE` | 模式：`"Vision"` / `"Potential"` / `"All"` | `"Vision"` |
| `QUICK_SPEED` | 快速滴定步长（ml） | 0.3 |
| `SLOW_SPEED` | 慢速滴定步长（ml） | 0.08 |
| `BLUE_THRESHOLD` | 蓝色概率阈值 | 0.40 |
| `SMOOTH_WINDOW` | 滑动平均窗口大小 | 3 |
| `CONFIRM_FRAMES` | 状态切换确认帧数 | 2 |

---

## 模型训练

提供两个训练脚本（位于 `tools/`）：

```bash
# 训练 EBT 2分类模型（紫色 vs 蓝色）
python tools/continue_train_ebt.py

# 训练 Color_Model_ 3分类模型（酒红 vs 紫 vs 蓝）
python tools/continue_train.py
```

### 数据组织

```
data_ebt/                 # EBT 2分类训练数据
├── purple/               # 紫色终点图片
└── blue/                 # 蓝色终点图片

data_3cls/                # Color_Model_ 3分类训练数据
├── wine_red/             # 滴定前酒红色图片
├── purple/               # 过渡色紫色图片
└── blue/                 # 终点蓝色图片
```

训练时自动合并 `_archive/`（历史数据）和 `feedback/`（人工纠错数据）。

### 训练特性

- 加权采样解决类别不平衡
- 数据增强（随机裁剪、翻转、旋转、颜色抖动、模糊）
- AdamW 优化器 + ReduceLROnPlateau 调度器
- 标签平滑 + 早停机制
- 类别加权交叉熵损失

---

## 滴定过程按键

| 按键 | 功能 |
|------|------|
| `1` / `F1` | 保存当前帧为酒红色反馈 |
| `2` / `F2` | 保存当前帧为紫色反馈 |
| `3` / `F3` | 保存当前帧为蓝色反馈 |
| `q` | 退出实验 |

---

## 工具脚本说明

| 脚本 | 用途 |
|------|------|
| `tools/camera_preview.py` | 摄像头实时预览 + ResNet34 实时预测 |
| `tools/camera_diagnose_full.py` | 读取摄像头所有 OpenCV 参数 |
| `tools/camera_diagnose_v3.py` | 摄像头诊断（不使用 DShow 后端） |
| `tools/pump_diagnose.py` | 快速串口通信测试 |
| `tools/pump_diagnose_v2.py` | 完整注射泵协议模拟 + 状态位解析 |
| `tools/continue_train.py` | 3分类颜色模型续训 |
| `tools/continue_train_ebt.py` | 2分类 EBT 模型续训 |

---

## 硬件要求

- **摄像头**：OpenCV 支持的任意 USB 摄像头（Windows 上使用 DShow 后端）
- **注射泵**：支持 `IP|{speed},{direction}\n` 协议、115200 波特率的串口注射泵
- **串口适配器**：CH340 USB 转串口模块（自动检测）

---

## 说明

本项目为 Mlabs 竞赛开发。模型权重和数据集不包含在此仓库中。
