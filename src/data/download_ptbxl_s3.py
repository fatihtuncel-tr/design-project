import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from src.data.preprocessing import load_ptbxl_metadata

S3_BASE = "https://physionet-open.s3.amazonaws.com/ptb-xl/1.0.3/"

def download_record_pair(session, s3_base, base_dir, filename_lr):
    """Her kayit icin .dat ve .hea dosyalarini indirir."""
    for ext in [".dat", ".hea"]:
        rel_path = filename_lr + ext
        target_path = os.path.join(base_dir, rel_path)
        if os.path.exists(target_path) and os.path.getsize(target_path) > 0:
            continue
        os.makedirs(os.path.dirname(target_path), exist_ok=True)
        url = s3_base + rel_path
        resp = session.get(url, timeout=20)
        resp.raise_for_status()
        with open(target_path, "wb") as f:
            f.write(resp.content)
    return filename_lr

def download_all_records(max_workers=24):
    df, base_dir = load_ptbxl_metadata("data/raw/ptb-xl")
    filenames = df["filename_lr"].tolist()
    print(f"Toplam {len(filenames)} EKG kaydi S3 uzerinden paralel indiriliyor...")

    # Thread-local session kullanarak baglanti havuzunu optimize edelim
    import threading
    thread_local = threading.local()

    def get_session():
        if not hasattr(thread_local, "session"):
            thread_local.session = requests.Session()
            adapter = requests.adapters.HTTPAdapter(pool_connections=max_workers, pool_maxsize=max_workers)
            thread_local.session.mount("https://", adapter)
        return thread_local.session

    def worker_task(fn):
        session = get_session()
        return download_record_pair(session, S3_BASE, base_dir, fn)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(worker_task, fn) for fn in filenames]
        for _ in tqdm(as_completed(futures), total=len(futures), desc="PTB-XL Records100"):
            pass

    print("Tum 100 Hz EKG kayitlari basariyla tamamlandi!")

if __name__ == "__main__":
    download_all_records(max_workers=24)
