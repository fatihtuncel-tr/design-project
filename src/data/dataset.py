import os
import numpy as np
import torch
from torch.utils.data import Dataset
import wfdb

SUPERCLASSES = ["NORM", "MI", "STTC", "CD", "HYP"]

class ECGDataset(Dataset):
    def __init__(self, df, base_dir, leads="12", normalize=True, preload_signals=False, npz_path=None):
        """
        PTB-XL EKG Dataset sinifi.
        
        Args:
            df (pd.DataFrame): Metadata dataframe (preprocessing.py ciktisi).
            base_dir (str): PTB-XL veri kumesinin temel dizini.
            leads (str): '12' (tum 12 derivasyon) veya '1' (yalniz Lead I - akilli saat standardi).
            normalize (bool): Sinyali z-score ile normalize etme (mean=0, std=1).
            preload_signals (bool): Hizli egitim icin RAM'e yukleme.
            npz_path (str, optional): Onceden olusturulmus .npz onbellek dosyasi.
        """
        self.df = df.reset_index()
        self.base_dir = base_dir
        self.leads = leads
        self.normalize = normalize
        self.preload_signals = preload_signals

        # Cok etiketli hedef vektorleri (5 super-sinif)
        self.labels = self.df[SUPERCLASSES].values.astype(np.float32)
        self.filenames = self.df["filename_lr"].values
        self.preloaded_data = None

        if npz_path and os.path.exists(npz_path):
            data = np.load(npz_path)
            raw_sigs = data["signals"] # (N, 12, 1000)
            leads_str = str(self.leads)
            if leads_str == "1":
                self.preloaded_data = raw_sigs[:, 0:1, :]
            elif leads_str == "6":
                self.preloaded_data = raw_sigs[:, 0:6, :]
            elif leads_str == "2":
                self.preloaded_data = raw_sigs[:, 0:2, :]
            else:
                self.preloaded_data = raw_sigs
            print(f"Hizli onbellekten {len(self.preloaded_data)} kayit yuklendi: {npz_path} (leads={leads_str})")
        elif self.preload_signals:
            print(f"Toplam {len(self.filenames)} sinyal RAM'e on yukleniyor...")
            signals = []
            for fn in self.filenames:
                signals.append(self._load_signal(fn))
            self.preloaded_data = np.stack(signals, axis=0) # (N, C, 1000)

    def _load_signal(self, filename_lr):
        full_path = os.path.join(self.base_dir, filename_lr)
        # wfdb ile oku (1000, 12)
        signal, _ = wfdb.rdsamp(full_path)
        # (12, 1000) sekline getir
        signal = signal.T

        if self.leads == "1":
            # Yalnizca Lead I (indeks 0) -> (1, 1000)
            signal = signal[0:1, :]
        elif self.leads == "6":
            # Ekstremite derivasyonlari (I, II, III, aVR, aVL, aVF) -> (6, 1000)
            signal = signal[0:6, :]
        elif self.leads == "2":
            # Derivasyon I ve II -> (2, 1000)
            signal = signal[0:2, :]

        if self.normalize:
            # Her derivasyon kanali icin z-score normalizasyonu
            mean = np.nanmean(signal, axis=1, keepdims=True)
            std = np.nanstd(signal, axis=1, keepdims=True)
            std[std == 0] = 1.0
            signal = (signal - mean) / std

        # NaN kontrolu
        signal = np.nan_to_num(signal, nan=0.0)
        return signal.astype(np.float32)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if self.preloaded_data is not None:
            x = self.preloaded_data[idx]
        else:
            x = self._load_signal(self.filenames[idx])
        
        y = self.labels[idx]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)
