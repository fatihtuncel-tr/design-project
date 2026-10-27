import numpy as np
from sklearn.metrics import roc_auc_score, f1_score, brier_score_loss

def compute_binary_ece(probs: np.ndarray, targets: np.ndarray, n_bins: int = 15):
    """
    Tek bir ikili sinif icin Beklenen Kalibrasyon Hatasi (ECE).
    
    Args:
        probs (np.ndarray): 1B dizi, tahmin edilen olasiliklar [0, 1].
        targets (np.ndarray): 1B dizi, gercek ikili etiketler (0 veya 1).
        n_bins (int): Kutu sayisi (varsayilan 15).
    
    Returns:
        float: ECE skoru [0, 1].
        dict: Guvenilirlik diyagrami cizimi icin kutu istatistikleri.
    """
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n_total = len(probs)
    
    bin_accs = []
    bin_confs = []
    bin_counts = []

    for i in range(n_bins):
        bin_lower = bin_edges[i]
        bin_upper = bin_edges[i + 1]
        
        # Son kutu icin ust sinir dahil
        if i == n_bins - 1:
            in_bin = (probs >= bin_lower) & (probs <= bin_upper)
        else:
            in_bin = (probs >= bin_lower) & (probs < bin_upper)
            
        count = np.sum(in_bin)
        bin_counts.append(count)
        
        if count > 0:
            bin_acc = np.mean(targets[in_bin])
            bin_conf = np.mean(probs[in_bin])
            bin_accs.append(bin_acc)
            bin_confs.append(bin_conf)
            ece += (count / n_total) * np.abs(bin_acc - bin_conf)
        else:
            bin_accs.append(0.0)
            bin_confs.append((bin_lower + bin_upper) / 2.0)
            
    diagram_data = {
        "bin_edges": bin_edges,
        "bin_accs": np.array(bin_accs),
        "bin_confs": np.array(bin_confs),
        "bin_counts": np.array(bin_counts),
    }
    return float(ece), diagram_data

def evaluate_predictions(probs: np.ndarray, targets: np.ndarray, thresholds: np.ndarray = None, n_bins: int = 15):
    """
    Cok etiketli tahminleri AUROC, Macro-F1, ECE (sinif basina ortalama) ve Brier ile degerlendirir.
    
    Args:
        probs (np.ndarray): (N, C) boyutunda tahmin edilen olasiliklar.
        targets (np.ndarray): (N, C) boyutunda gercek ikili etiketler.
        thresholds (np.ndarray): (C,) boyutunda sinif basina esik degerleri (None ise 0.5 alinir).
        n_bins (int): ECE kutu sayisi (15).
        
    Returns:
        dict: Tum metriklerin sonuclari.
    """
    n_samples, n_classes = probs.shape
    if thresholds is None:
        thresholds = np.full(n_classes, 0.5)

    class_aurocs = []
    class_f1s = []
    class_eces = []
    class_briers = []
    class_diagrams = []

    for c in range(n_classes):
        y_true = targets[:, c]
        y_prob = probs[:, c]

        # AUROC (yalniz her iki sinif da mevcutsa)
        if len(np.unique(y_true)) > 1:
            auc = roc_auc_score(y_true, y_prob)
            class_aurocs.append(auc)
        else:
            class_aurocs.append(np.nan)

        # F1
        y_pred = (y_prob >= thresholds[c]).astype(int)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        class_f1s.append(f1)

        # ECE
        ece, diag = compute_binary_ece(y_prob, y_true, n_bins=n_bins)
        class_eces.append(ece)
        class_diagrams.append(diag)

        # Brier
        brier = brier_score_loss(y_true, y_prob)
        class_briers.append(brier)

    return {
        "macro_auroc": float(np.nanmean(class_aurocs)),
        "macro_f1": float(np.nanmean(class_f1s)),
        "macro_ece": float(np.nanmean(class_eces)),
        "macro_brier": float(np.nanmean(class_briers)),
        "class_aurocs": class_aurocs,
        "class_f1s": class_f1s,
        "class_eces": class_eces,
        "class_briers": class_briers,
        "class_diagrams": class_diagrams,
    }

def find_optimal_thresholds(probs: np.ndarray, targets: np.ndarray):
    """
    Kat 9 (val) verisi uzerinde her sinif icin F1 skorunu maksimize eden esik degerleri bulur.
    """
    n_classes = probs.shape[1]
    best_thresholds = np.zeros(n_classes)
    candidate_thresholds = np.linspace(0.1, 0.9, 81)

    for c in range(n_classes):
        y_true = targets[:, c]
        y_prob = probs[:, c]
        best_f1 = -1.0
        best_th = 0.5

        for th in candidate_thresholds:
            y_pred = (y_prob >= th).astype(int)
            f1 = f1_score(y_true, y_pred, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_th = th
        best_thresholds[c] = best_th

    return best_thresholds
