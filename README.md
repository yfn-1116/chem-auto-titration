# Mlabs AI Titration System v1.5

> 基于计算机视觉和深度学习的自动滴定实验系统
> AI-powered automatic titration system using computer vision and deep learning

Automate chemical titration experiments with real-time color recognition via ResNet34 and precise syringe pump control. Supports both **vision-based** (color change detection) and **potentiometric** (voltage measurement) titration modes.

---

## Features

- **Real-time Color Recognition** — ResNet34 classifies solution color (blue / purple / wine_red) via USB camera feed
- **State Machine Control** — Three-phase FAST → SLOW → STOP logic with sliding-window smoothing and multi-frame confirmation to prevent jitter
- **Syringe Pump Automation** — Serial communication (CH340), auto-refill, volume calibration, servo control
- **Potentiometric Mode** — Optional voltage readout with curve fitting and second-derivative endpoint analysis
- **Human-in-the-loop Feedback** — Hotkey misclassification correction saves labeled images for retraining
- **Data Logging** — Auto-saves titration curve plots, experiment JSON records, and endpoint images
- **Platform Upload** — Optional integration with competition platform (jingsai.mools.net)

---

## Project Structure

```
chem-auto-titration/
├── titration.py           # Main entry point (standalone + --upload mode)
├── model.py               # ResNet34 / ResNet50 / ResNeXt model definition
├── Find_COM.py            # Serial port auto-detection (CH340 → USB → fallback)
├── requirements.txt       # Python dependencies
├── README.md
├── login/
│   └── .gitkeep           # Credentials file (gitignored)
└── tools/
    ├── camera_preview.py       # Live camera feed + real-time model inference
    ├── camera_diagnose_full.py # Comprehensive camera parameter diagnostics
    ├── camera_diagnose_v3.py   # Camera diagnostics (non-DShow backend)
    ├── pump_diagnose.py        # Syringe pump communication test
    ├── pump_diagnose_v2.py     # Full pump protocol simulation
    ├── continue_train.py       # Retrain 3-class model (wine_red/purple/blue)
    └── continue_train_ebt.py   # Retrain 2-class model (blue/purple)
```

---

## Quick Start

### Prerequisites

- Python 3.8+
- PyTorch 1.9+
- USB camera
- Syringe pump with serial interface (optional, system runs without it)

### Installation

```bash
pip install -r requirements.txt
```

### Prepare Model

Place trained weights and class mapping in `pths/`:

```
pths/
├── EBTNet.pth              # 2-class model (purple / blue)
├── EBT.json                # Class index mapping
├── Color_Model_Net.pth     # 3-class model (wine_red / purple / blue)
└── Color_Model_.json       # Class index mapping
```

### Run

```bash
# Standalone mode (no platform login required)
python titration.py

# Full mode with platform login and data upload
python titration.py --upload
```

---

## How It Works

### Titration Pipeline

```
Camera Capture → Image Preprocessing → ResNet34 Inference → Color Classification
                                                                    ↓
                                           ┌────────────────────────────────┐
                                           │  State Machine                │
                                           │  ┌──────┐    ┌──────┐    ┌──┐ │
                                           │  │ FAST  │───→│ SLOW  │───→│STOP│
                                           │  └──────┘    └──────┘    └──┘ │
                                           │  (titrate    (titrate    (end)  │
                                           │   quickly)   slowly)           │
                                           └────────────────────────────────┘
                                                                    ↓
                                                        Syringe Pump Control
```

### State Machine Logic

1. **FAST phase** — Titrates at `QUICK_SPEED` (e.g., 0.3 ml/step). Monitors blue probability. If `p_blue > BLUE_THRESHOLD` for `CONFIRM_FRAMES` consecutive frames, transitions to STOP. If purple (transition color) is detected, transitions to SLOW.
2. **SLOW phase** — Titrates at `SLOW_SPEED` (e.g., 0.08 ml/step). Dual-criteria endpoint detection: absolute threshold OR relative dominance (blue > purple by 0.15). Falls back to STOP after 25 slow cycles.
3. **STOP** — Records endpoint volume, saves final image, optionally continues for overdose titration.

### Smoothing & Anti-jitter

- **Sliding window** — Averages predictions over the last `SMOOTH_WINDOW` frames.
- **Consecutive confirmation** — Requires `CONFIRM_FRAMES` consecutive positive detections before state transition.

---

## Configuration

Edit parameters in `titration.py` → `run_titration_standalone()`:

| Parameter | Description | Default |
|-----------|-------------|---------|
| `CAMERA_INDEX` | Camera device index (0=built-in, 1=USB) | 1 |
| `MODEL_NAME` | Model weights basename in `pths/` | `"EBT"` |
| `N_CLASSES` | Number of output classes | 2 |
| `TRANSITION_CLASS` | Transition color name (optional) | `None` |
| `OVERDOSE_COUNT` | Extra titrations after endpoint | 0 |
| `VOLUME_PAR` | Volume calibration factor | 0.913 |
| `FINAL_VOLUME` | Max target volume (ml) | 100 |
| `TYPE` | Mode: `"Vision"`, `"Potential"`, or `"All"` | `"Vision"` |
| `QUICK_SPEED` | Fast titration step (ml) | 0.3 |
| `SLOW_SPEED` | Slow titration step (ml) | 0.08 |
| `BLUE_THRESHOLD` | Blue probability threshold | 0.40 |
| `SMOOTH_WINDOW` | Sliding window size | 3 |
| `CONFIRM_FRAMES` | Consecutive frames for confirmation | 2 |

---

## Model Training

Two training scripts are available in `tools/`:

```bash
# Train 2-class EBT model (purple vs blue)
python tools/continue_train_ebt.py

# Train 3-class Color_Model (wine_red vs purple vs blue)
python tools/continue_train.py
```

### Data Organization

Place labeled images in the following structure:

```
data_ebt/                 # For EBT 2-class training
├── purple/               # Purple endpoint images
└── blue/                 # Blue endpoint images

data_3cls/                # For Color_Model 3-class training
├── wine_red/             # Pre-transition red images
├── purple/               # Transition purple images
└── blue/                 # Endpoint blue images
```

Additional data sources (`_archive/`, `feedback/`) are automatically merged during training.

### Training Features

- Weighted sampling for class imbalance
- Data augmentation (random crop, flip, rotation, color jitter, blur)
- AdamW optimizer with ReduceLROnPlateau scheduler
- Label smoothing + early stopping
- Class-weighted CrossEntropy loss

---

## Key Controls During Titration

| Key | Action |
|-----|--------|
| `1` / `F1` | Save frame as wine_red feedback |
| `2` / `F2` | Save frame as purple feedback |
| `3` / `F3` | Save frame as blue feedback |
| `q` | Quit experiment |

---

## Tool Scripts

| Script | Purpose |
|--------|---------|
| `tools/camera_preview.py` | Live camera preview with real-time ResNet34 predictions |
| `tools/camera_diagnose_full.py` | Comprehensive camera parameter readout (all OpenCV props) |
| `tools/camera_diagnose_v3.py` | Camera diagnostics without DShow backend |
| `tools/pump_diagnose.py` | Quick serial communication test |
| `tools/pump_diagnose_v2.py` | Full pump protocol simulation + status bit parsing |
| `tools/continue_train.py` | Retrain 3-class color model |
| `tools/continue_train_ebt.py` | Retrain 2-class EBT model |

---

## Hardware Requirements

- **Camera**: Any USB camera supported by OpenCV (tested with DShow backend on Windows)
- **Syringe Pump**: Serial-controlled pump supporting `IP|{speed},{direction}\n` protocol at 115200 baud
- **Serial Adapter**: CH340 USB-to-serial adapter (auto-detected)

---

## License

This project is developed for the Mlabs competition. Model weights and datasets are not included in this repository.
