# Mlabs AI Titration System v1.5

基于计算机视觉和深度学习的自动滴定实验系统。使用 ResNet34 实时分析滴定液颜色变化，自动控制注射泵完成精确滴定。

## 系统架构

```
滴定过程:
  摄像头 → 图像采集 → ResNet34推理 → 颜色分类 → 状态机判断 → 注射泵控制
                                                                   ↓
                                                       [FAST] 快速滴定
                                                       [SLOW] 缓慢滴定
                                                       [STOP] 终点停止
```

- **视觉识别**: ResNet34 实时分类溶液颜色 (blue/purple/wine_red)
- **状态机**: FAST → SLOW → STOP 三级控制，平滑窗口+连续帧确认防抖动
- **注射泵控制**: 串口通信，自动补液，体积校正
- **电位检测**: 可选电位滴定模式，支持电压曲线二阶导分析
- **数据保存**: 自动保存滴定曲线图、实验数据 JSON、终点图像

## 快速开始

### 环境要求

- Python 3.8+
- PyTorch 1.9+
- USB 摄像头
- 注射泵 (串口通信, 可选)

### 安装

```bash
pip install -r requirements.txt
```

### 准备模型

将训练好的模型权重放入 `pths/` 目录:

```
pths/
├── EBNet.pth          # 2分类模型 (blue/purple)
├── EBT.json           # 类别映射文件
├── Color_Model_Net.pth  # 3分类模型 (wine_red/purple/blue)
└── Color_Model_.json    # 类别映射文件
```

### 运行

```bash
# 独立模式 (无需登录平台)
python titration.py

# 完整模式 (含平台登录和数据上传)
python titration.py --upload
```

## 参数配置

在 `titration.py` 的 `run_titration_standalone()` 函数中修改:

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `CAMERA_INDEX` | 摄像头编号 (0=内置, 1=USB) | 1 |
| `MODEL_NAME` | 模型名称 | "EBT" |
| `QUICK_SPEED` | 快速滴定速度 (ml/次) | 0.3 |
| `SLOW_SPEED` | 慢速滴定速度 (ml/次) | 0.08 |
| `BLUE_THRESHOLD` | 蓝色概率阈值 | 0.40 |
| `VOLUME_PAR` | 体积校正系数 | 0.913 |

## 模型训练

项目提供了独立的训练脚本:

```bash
# 训练 EBT 2分类模型
python tools/continue_train_ebt.py

# 训练 Color_Model_ 3分类模型
python tools/continue_train.py
```

将标注数据 (wine_red/purple/blue 三类) 放入 `data_3cls/` 或 `data_ebt/` 目录即可开始训练。

## 工具脚本

| 脚本 | 用途 |
|------|------|
| `tools/camera_preview.py` | 摄像头实时预览 + 模型预测 |
| `tools/camera_diagnose_full.py` | 全面诊断摄像头参数 |
| `tools/pump_diagnose_v2.py` | 注射泵通信诊断 |
| `tools/continue_train.py` | 3分类模型续训 |
| `tools/continue_train_ebt.py` | 2分类模型续训 |

## 按键控制

滴定过程中:

- `1` (或 `F1`) - 保存当前帧为 wine_red 反馈
- `2` (或 `F2`) - 保存当前帧为 purple 反馈
- `3` (或 `F3`) - 保存当前帧为 blue 反馈
- `q` - 退出实验
