import argparse
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.preprocessing import load_ptbxl_metadata, get_train_val_test_splits
from src.data.dataset import ECGDataset
from src.models.simple_cnn1d import SimpleCNN1D
from src.models.xresnet1d import xresnet1d50, xresnet1d101
from src.evaluation.metrics import evaluate_predictions, find_optimal_thresholds

def train_epoch(model, loader, criterion, optimizer, scheduler, device):
    model.train()
    total_loss = 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        if scheduler:
            scheduler.step()
        total_loss += loss.item() * len(x)
    return total_loss / len(loader.dataset)

def predict_logits(model, loader, device):
    model.eval()
    all_logits = []
    all_targets = []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            logits = model(x)
            all_logits.append(logits.cpu().numpy())
            all_targets.append(y.numpy())
    return np.concatenate(all_logits, axis=0), np.concatenate(all_targets, axis=0)

def main():
    parser = argparse.ArgumentParser(description="EKG Model Egitimi (12 veya 1 Derivasyon)")
    parser.add_argument("--model", type=str, default="simple_cnn", choices=["simple_cnn", "xresnet50", "xresnet101"])
    parser.add_argument("--leads", type=str, default="12", choices=["12", "1"])
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--save_name", type=str, default=None)
    args = parser.parse_args()

    in_channels = int(args.leads)
    if args.save_name is None:
        args.save_name = f"{args.model}_{args.leads}lead"

    print(f"=== Egitim Basliyor: Model={args.model}, Derivasyon={args.leads}, Cihaz={args.device} ===")

    # 1. Veri Hazirligi
    df, base_dir = load_ptbxl_metadata("data/raw/ptb-xl")
    train_df, val_df, test_df = get_train_val_test_splits(df)

    train_npz = "data/processed/train_data.npz"
    val_npz = "data/processed/val_data.npz"
    test_npz = "data/processed/test_data.npz"

    train_ds = ECGDataset(train_df, base_dir, leads=args.leads, npz_path=train_npz if os.path.exists(train_npz) else None)
    val_ds = ECGDataset(val_df, base_dir, leads=args.leads, npz_path=val_npz if os.path.exists(val_npz) else None)
    test_ds = ECGDataset(test_df, base_dir, leads=args.leads, npz_path=test_npz if os.path.exists(test_npz) else None)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0 if os.path.exists(train_npz) else 2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0 if os.path.exists(val_npz) else 2)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0 if os.path.exists(test_npz) else 2)

    # 2. Model Kurulumu
    if args.model == "simple_cnn":
        model = SimpleCNN1D(in_channels=in_channels, num_classes=5)
    elif args.model == "xresnet50":
        model = xresnet1d50(in_channels=in_channels, num_classes=5)
    elif args.model == "xresnet101":
        model = xresnet1d101(in_channels=in_channels, num_classes=5)
    model.to(args.device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.lr,
        steps_per_epoch=len(train_loader),
        epochs=args.epochs,
        pct_start=0.2,
    )

    # 3. Egitim Dongusu
    best_val_loss = float("inf")
    save_dir = "results/checkpoints"
    os.makedirs(save_dir, exist_ok=True)
    best_model_path = os.path.join(save_dir, f"{args.save_name}_best.pt")

    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, scheduler, args.device)
        val_logits, val_targets = predict_logits(model, val_loader, args.device)
        val_loss = criterion(torch.tensor(val_logits), torch.tensor(val_targets)).item()

        val_probs = 1.0 / (1.0 + np.exp(-val_logits))
        val_metrics = evaluate_predictions(val_probs, val_targets)

        print(
            f"Epoch [{epoch:02d}/{args.epochs:02d}] "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val AUROC: {val_metrics['macro_auroc']:.4f} | "
            f"Val ECE: {val_metrics['macro_ece']:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_model_path)
            print(f"  --> En iyi model agirliklari kaydedildi: {best_model_path}")

    # 4. En Iyi Modeli Yukleme ve Test Sonuclari
    print(f"\nEn iyi agirliklar yukleniyor ({best_model_path})...")
    model.load_state_dict(torch.load(best_model_path))

    # Fold 9 (Val/Calib) ve Fold 10 (Test) tahminlerini kaydet
    val_logits, val_targets = predict_logits(model, val_loader, args.device)
    test_logits, test_targets = predict_logits(model, test_loader, args.device)

    val_probs = 1.0 / (1.0 + np.exp(-val_logits))
    test_probs = 1.0 / (1.0 + np.exp(-test_logits))

    best_thresholds = find_optimal_thresholds(val_probs, val_targets)
    test_metrics = evaluate_predictions(test_probs, test_targets, thresholds=best_thresholds)

    print("\n=== Test Kumesi (Fold 10) Sonuclari (Kalibrasyonsuz) ===")
    print(f"Macro AUROC : {test_metrics['macro_auroc']:.4f}")
    print(f"Macro F1    : {test_metrics['macro_f1']:.4f}")
    print(f"Macro ECE   : {test_metrics['macro_ece']:.4f}")
    print(f"Macro Brier : {test_metrics['macro_brier']:.4f}")

    # Ciktilari kaydet (.npz)
    pred_save_path = os.path.join(save_dir, f"{args.save_name}_predictions.npz")
    np.savez_compressed(
        pred_save_path,
        val_logits=val_logits,
        val_targets=val_targets,
        test_logits=test_logits,
        test_targets=test_targets,
        best_thresholds=best_thresholds,
    )
    print(f"Tahminler ve logits kaydedildi: {pred_save_path}")

if __name__ == "__main__":
    main()
