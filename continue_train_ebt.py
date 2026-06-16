"""继续训练 EBT (2分类: blue/purple, wine_red→purple)
整合所有数据源，加载已有权重，设置防过拟合机制
"""
import os
import sys
import json
import random
import shutil
from collections import Counter

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from tqdm import tqdm

from model import resnet34

CWD = os.path.dirname(os.path.abspath(__file__))
PTHS_DIR = os.path.join(CWD, "pths")
DATA_DIR = os.path.join(CWD, "data_ebt")

DATA_SOURCES = {
    "rowdata": os.path.join(CWD, "_archive", "Rowdata"),
    "caiji": os.path.join(CWD, "_archive", "采集数据"),
    "caiji_raw": os.path.join(CWD, "_archive", "采集数据_raw"),
    "feedback": os.path.join(CWD, "feedback"),
}

CLASSES_3 = ["wine_red", "purple", "blue"]


def collect_all_data(target_dir):
    if os.path.exists(target_dir):
        shutil.rmtree(target_dir)

    wine_red_dir = os.path.join(target_dir, "wine_red")
    purple_dir = os.path.join(target_dir, "purple")
    blue_dir = os.path.join(target_dir, "blue")
    os.makedirs(wine_red_dir, exist_ok=True)
    os.makedirs(purple_dir, exist_ok=True)
    os.makedirs(blue_dir, exist_ok=True)

    total = 0
    for source_name, source_path in DATA_SOURCES.items():
        for cls_name in CLASSES_3:
            src_cls_dir = os.path.join(source_path, cls_name)
            if not os.path.isdir(src_cls_dir):
                continue
            files = [
                f for f in os.listdir(src_cls_dir)
                if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))
            ]
            dst_dir = purple_dir if cls_name == "wine_red" else os.path.join(target_dir, cls_name)
            for f in files:
                dst = os.path.join(dst_dir, f)
                if not os.path.exists(dst):
                    shutil.copy2(os.path.join(src_cls_dir, f), dst)
            total += len(files)
            cls_dst = "purple (原wine_red)" if cls_name == "wine_red" else cls_name
            print(f"  [{source_name}] {cls_name}→{cls_dst}: {len(files)} 张")
    print(f"  共收集 {total} 张图片")

    for cls_name in ["blue", "purple"]:
        d = os.path.join(target_dir, cls_name)
        cnt = len([f for f in os.listdir(d) if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))])
        print(f"  最终 {cls_name}: {cnt} 张")
    return total


def split_train_val(data_dir, val_ratio=0.15, seed=42):
    random.seed(seed)
    train_dir = os.path.join(data_dir, "train")
    val_dir = os.path.join(data_dir, "val")

    for cls_name in ["blue", "purple"]:
        src = os.path.join(data_dir, cls_name)
        files = [f for f in os.listdir(src) if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))]
        random.shuffle(files)
        n_val = max(1, int(len(files) * val_ratio))
        val_files = set(files[:n_val])
        train_files = files[n_val:]

        os.makedirs(os.path.join(train_dir, cls_name), exist_ok=True)
        os.makedirs(os.path.join(val_dir, cls_name), exist_ok=True)

        for f in train_files:
            shutil.move(os.path.join(src, f), os.path.join(train_dir, cls_name, f))
        for f in val_files:
            shutil.move(os.path.join(src, f), os.path.join(val_dir, cls_name, f))

        print(f"  {cls_name}: train={len(train_files)}, val={len(val_files)}")

    for d in ["wine_red", "blue", "purple"]:
        p = os.path.join(data_dir, d)
        if os.path.isdir(p):
            shutil.rmtree(p)
    print(f"  分割完成: train/2类, val/2类")


def get_transforms():
    normalize = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.75, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.3, contrast=0.25, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        normalize,
    ])
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        normalize,
    ])
    return train_transform, val_transform


def compute_class_weights(dataset):
    labels = [label for _, label in dataset]
    counter = Counter(labels)
    n_classes = len(counter)
    n_samples = len(labels)
    weights = [n_samples / (n_classes * counter[i]) for i in range(n_classes)]
    return torch.tensor(weights, dtype=torch.float)


def evaluate(net, loader, device, n_classes):
    net.eval()
    correct = 0
    total = 0
    per_class_correct = torch.zeros(n_classes)
    per_class_total = torch.zeros(n_classes)
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            preds = outputs.argmax(dim=1)
            correct += torch.eq(preds, labels).sum().item()
            total += labels.size(0)
            for c in range(n_classes):
                mask = labels == c
                per_class_total[c] += mask.sum().item()
                per_class_correct[c] += (preds[mask] == c).sum().item()
    acc = correct / total if total > 0 else 0.0
    recall = [
        round((per_class_correct[c] / per_class_total[c]).item(), 4)
        if per_class_total[c] > 0 else 0.0
        for c in range(n_classes)
    ]
    return acc, recall


def main():
    print("=" * 60)
    print("EBT 2分类续训 (蓝/紫, wine_red→purple)")
    print("=" * 60)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

    # 1. 收集所有数据
    print("\n[1/4] 整合所有标注数据 (wine_red→purple)...")
    total = collect_all_data(DATA_DIR)
    if total == 0:
        print("没有数据，退出")
        return

    # 2. 分割 train/val
    print(f"\n[2/4] 分割训练集/验证集 (val_ratio=0.15)...")
    split_train_val(DATA_DIR, val_ratio=0.15)

    # 3. 数据加载
    print(f"\n[3/4] 加载数据...")
    train_transform, val_transform = get_transforms()
    train_dataset = datasets.ImageFolder(
        root=os.path.join(DATA_DIR, "train"),
        transform=train_transform
    )
    val_dataset = datasets.ImageFolder(
        root=os.path.join(DATA_DIR, "val"),
        transform=val_transform
    )
    class_to_idx = train_dataset.class_to_idx
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    n_classes = len(class_to_idx)
    print(f"  类别映射: {class_to_idx}")
    print(f"  训练集: {len(train_dataset)} | 验证集: {len(val_dataset)}")

    batch_size = 32
    nw = min(os.cpu_count() or 4, 8)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=nw, pin_memory=torch.cuda.is_available()
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=nw, pin_memory=torch.cuda.is_available()
    )

    # 4. 模型
    print(f"\n[4/4] 加载模型 + 训练...")
    net = resnet34(num_classes=n_classes).to(device)
    pretrained_path = os.path.join(PTHS_DIR, "EBTNet.pth")
    if os.path.exists(pretrained_path):
        print(f"  加载已有权重: {pretrained_path}")
        state_dict = torch.load(pretrained_path, map_location=device)
        net.load_state_dict(state_dict, strict=False)
    else:
        print("  未找到已有权重，从头训练")

    # 5. 保存类别映射
    json_path = os.path.join(PTHS_DIR, "EBT.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(idx_to_class, f, indent=4, ensure_ascii=False)
    print(f"  类别映射已保存: {json_path}")

    # ============ 训练配置 ============
    EPOCHS = 60
    LR = 0.00005
    WEIGHT_DECAY = 1e-4
    LABEL_SMOOTHING = 0.1

    class_weights = compute_class_weights(train_dataset)
    print(f"  类别权重: {class_weights.tolist()}")
    class_weights = class_weights.to(device)
    loss_function = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=LABEL_SMOOTHING)

    optimizer = optim.AdamW(net.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=4, min_lr=1e-6
    )

    early_stop_patience = 12
    best_acc = 0.0
    epochs_no_improve = 0
    save_path = os.path.join(PTHS_DIR, "EBTNet.pth")
    train_steps = len(train_loader)

    print(f"\n  开始训练 (epochs={EPOCHS}, lr={LR}, weight_decay={WEIGHT_DECAY})")
    print(f"  label_smoothing={LABEL_SMOOTHING}, early_stop_patience={early_stop_patience}\n")

    for epoch in range(EPOCHS):
        net.train()
        running_loss = 0.0
        train_bar = tqdm(train_loader, file=sys.stdout, desc=f"Epoch [{epoch + 1}/{EPOCHS}]")
        for images, labels in train_bar:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = net(images)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            train_bar.set_postfix(loss=f"{loss.item():.3f}")

        val_acc, val_recall = evaluate(net, val_loader, device, n_classes)
        avg_loss = running_loss / train_steps
        scheduler.step(val_acc)
        current_lr = optimizer.param_groups[0]["lr"]

        recall_str = ", ".join([f"{idx_to_class[i]}: {r:.4f}" for i, r in enumerate(val_recall)])
        print(f"  loss: {avg_loss:.4f} - val_acc: {val_acc:.4f} - recall: {{{recall_str}}} - lr: {current_lr:.6f}")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(net.state_dict(), save_path)
            epochs_no_improve = 0
            print(f"  >>> 保存最佳模型 (acc: {best_acc:.4f}) <<<")
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= early_stop_patience:
            print(f"\n早停触发! 连续 {early_stop_patience} 轮无提升, best_acc={best_acc:.4f}")
            break

        if current_lr <= 1e-6 and epoch > 20:
            print(f"\n学习率已降至最低，停止训练")
            break

    print(f"\n{'=' * 60}")
    print(f"训练完成！最佳验证准确率: {best_acc:.4f}")
    print(f"模型已保存: {save_path}")


if __name__ == "__main__":
    main()
