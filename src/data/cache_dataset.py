import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import numpy as np
from tqdm import tqdm
from src.data.preprocessing import load_ptbxl_metadata, get_train_val_test_splits
from src.data.dataset import ECGDataset

def cache_split(dataset, save_path):
    print(f"\n{save_path} olusturuluyor ({len(dataset)} kayit)...")
    signals = []
    labels = []
    
    for i in tqdm(range(len(dataset)), desc=os.path.basename(save_path)):
        x, y = dataset[i]
        signals.append(x.numpy())
        labels.append(y.numpy())
        
    x_arr = np.stack(signals, axis=0) # (N, 12, 1000)
    y_arr = np.stack(labels, axis=0)  # (N, 5)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    np.savez_compressed(save_path, signals=x_arr, labels=y_arr)
    print(f"Kaydedildi: {save_path} ({x_arr.shape}, {y_arr.shape})")

def main():
    raw_dir = "data/raw/ptb-xl"
    processed_dir = "data/processed"
    
    df, base_dir = load_ptbxl_metadata(raw_dir)
    train_df, val_df, test_df = get_train_val_test_splits(df)

    print("Veri kumesi numpy on-onbelleklemeye baslaniyor (12 derivasyon, 100 Hz)...")
    # 12 derivasyon yuklenir; 1-lead modeli egitilirken signals[:, 0:1, :] alinir
    train_ds = ECGDataset(train_df, base_dir, leads="12")
    val_ds = ECGDataset(val_df, base_dir, leads="12")
    test_ds = ECGDataset(test_df, base_dir, leads="12")

    cache_split(val_ds, os.path.join(processed_dir, "val_data.npz"))
    cache_split(test_ds, os.path.join(processed_dir, "test_data.npz"))
    cache_split(train_ds, os.path.join(processed_dir, "train_data.npz"))

    print("\nTum bolumler (Train, Val, Test) yuksek hizli .npz onbellegine alindi!")

if __name__ == "__main__":
    main()
