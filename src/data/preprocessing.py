import ast
import os
import numpy as np
import pandas as pd

SUPERCLASSES = ["NORM", "MI", "STTC", "CD", "HYP"]

def load_ptbxl_metadata(data_dir: str):
    """
    PTB-XL metadata CSV dosyalarini yukler ve tanisal super-sinif etiketlerini cikarir.
    """
    # Veri dizini icindeki olasi ptb-xl klasoru yapisi kontrolu
    if os.path.exists(os.path.join(data_dir, "ptbxl_database.csv")):
        base_dir = data_dir
    else:
        # Zip acildiginda alt klasor icine yerlesmis olabilir
        found = False
        for root, dirs, files in os.walk(data_dir):
            if "ptbxl_database.csv" in files:
                base_dir = root
                found = True
                break
        if not found:
            raise FileNotFoundError(f"ptbxl_database.csv bulunamadi: {data_dir}")

    db_path = os.path.join(base_dir, "ptbxl_database.csv")
    scp_path = os.path.join(base_dir, "scp_statements.csv")

    df = pd.read_csv(db_path, index_col="ecg_id")
    df.scp_codes = df.scp_codes.apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)

    # SCP ifadeleri ve super-sinif haritalamasi
    scp_df = pd.read_csv(scp_path, index_col=0)
    diag_scp = scp_df[scp_df.diagnostic == 1]

    def aggregate_diagnostic(y_dict):
        classes = set()
        for key in y_dict.keys():
            if key in diag_scp.index:
                diag_class = diag_scp.loc[key].diagnostic_class
                if isinstance(diag_class, str) and diag_class in SUPERCLASSES:
                    classes.add(diag_class)
        return list(classes)

    df["diagnostic_superclass"] = df.scp_codes.apply(aggregate_diagnostic)

    # Cok etiketli binary matris (0 veya 1)
    for c in SUPERCLASSES:
        df[c] = df["diagnostic_superclass"].apply(lambda classes: 1 if c in classes else 0)

    # Yas gruplama (<40, 40-60, 60-75, 75+)
    def assign_age_group(age):
        if pd.isna(age):
            return "Unknown"
        elif age < 40:
            return "<40"
        elif age <= 60:
            return "40-60"
        elif age <= 75:
            return "60-75"
        else:
            return "75+"

    df["age_group"] = df["age"].apply(assign_age_group)
    df["base_dir"] = base_dir

    return df, base_dir

def get_train_val_test_splits(df: pd.DataFrame):
    """
    PTB-XL resmi kat dagilimi:
    Kat 1-8: Egitim (Train)
    Kat 9: Dogrulama ve Kalibrasyon (Val & Calibration)
    Kat 10: Test (Test)
    """
    train_df = df[df.strat_fold <= 8].copy()
    val_df = df[df.strat_fold == 9].copy()
    test_df = df[df.strat_fold == 10].copy()
    return train_df, val_df, test_df
