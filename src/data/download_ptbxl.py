import os
import sys
import zipfile
import requests
from tqdm import tqdm

URL = "https://physionet.org/static/published-projects/ptb-xl/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3.zip"
TARGET_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../data/raw"))
ZIP_PATH = os.path.join(TARGET_DIR, "ptbxl_1.0.3.zip")
EXTRACT_DIR = os.path.join(TARGET_DIR, "ptb-xl")

def download_file(url: str, dest_path: str):
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    
    if os.path.exists(dest_path):
        remote_size = int(requests.head(url, allow_redirects=True).headers.get("content-length", 0))
        local_size = os.path.getsize(dest_path)
        if local_size == remote_size and remote_size > 0:
            print(f"Zip dosyasi zaten tam indirilmis ({local_size / 1e6:.1f} MB): {dest_path}")
            return
        else:
            print(f"Mevcut dosya eksik veya farkli boyutta. Yeniden indiriliyor...")

    print(f"Indiriliyor: {url}")
    response = requests.get(url, stream=True)
    response.raise_for_status()
    total_size = int(response.headers.get("content-length", 0))

    with open(dest_path, "wb") as f, tqdm(
        desc="PTB-XL Indirme",
        total=total_size,
        unit="iB",
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for chunk in response.iter_content(chunk_size=1024 * 1024): # 1MB chunks
            if chunk:
                size = f.write(chunk)
                bar.update(size)
    print("Indirme tamamlandi!")

def extract_essential(zip_path: str, extract_to: str):
    print("Zip arsivi aciliyor...")
    os.makedirs(extract_to, exist_ok=True)
    
    with zipfile.ZipFile(zip_path, "r") as z:
        members = z.namelist()
        # Sadece 100Hz kayitlari ve metadata CSV dosyalarini ayiklayalim (500Hz gerekmedigi icin disk ve zaman tasarrufu)
        essential_members = [
            m for m in members if ("records100" in m or m.endswith(".csv") or "scp_statements" in m)
        ]
        print(f"Toplam {len(essential_members)} gerekli dosya ayiklaniyor...")
        for member in tqdm(essential_members, desc="Ayiklama"):
            z.extract(member, extract_to)
    print(f"Veriler basariyla ayiklandi: {extract_to}")

if __name__ == "__main__":
    download_file(URL, ZIP_PATH)
    extract_essential(ZIP_PATH, EXTRACT_DIR)
