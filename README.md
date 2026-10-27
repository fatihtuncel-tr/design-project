# Tasarım Projesi: Derivasyon Sayısı Azaltılmış EKG Sınıflandırmasında Belirsizlik Kalibrasyonu ve Yaş Grubu Adaleti

Bu dizin, **Tasarım Projesi Ara Raporu** kapsamında tamamlanan ilk aşama çalışmalarını (Veri Hazırlığı, Sinyal Ön-İşleme, Model Mimarileri ve Altyapı) içermektedir.

---

## 📁 Dizin Yapısı (Ara Sürüm)

* **`IEEE_Proje_Ara_Raporu_1.docx`**: IEEE iki sütunlu konferans formatında hazırlanmış Ara Rapor (Teslim dokümanı).
* **`IEEE_Proje_Ara_Raporu_1.md`**: Raporun Markdown metin sürümü.
* **`Design Project proje konusu.docx`**: Proje öneri ve konu şartnamesi.
* **`requirements.txt`**: Projede kullanılan Python kütüphaneleri ve sürüm gereksinimleri (PyTorch, WFDB, Pandas vb.).
* **`data/`**:
  * `raw/ptb-xl/`: Orijinal PTB-XL veri seti metadata tabloları (`ptbxl_database.csv`, `scp_statements.csv`) ve 100 Hz EKG kayıtları.
  * `processed/`: 100 Hz olarak normalize edilip 10 katlı tabakalı yapıya göre önbelleğe alınmış eğitim, doğrulama ve test dizileri (`train_data.npz`, `val_data.npz`, `test_data.npz`).
* **`src/`**:
  * `data/`: Sinyal ön-işleme (`preprocessing.py`), PyTorch Dataset sınıfı (`dataset.py`) ve disk önbellekleme aracı (`cache_dataset.py`).
  * `models/`: Geliştirilen 1B derin öğrenme modelleri (`simple_cnn1d.py` ve `xresnet1d.py`).
  * `evaluation/`: Değerlendirme ve kalibrasyon metrik formülleri (`metrics.py`).
* **`experiments/`**:
  * `train.py`: Bir sonraki aşamada model eğitimlerini (12, 6, 2, 1 derivasyon) yürütmek üzere hazırlanan eğitim betiği.

---

## 🚀 Doğrulama ve Çalıştırma

### 1. Kütüphanelerin Yüklenmesi
```bash
pip install -r requirements.txt
```

### 2. Veri Boru Hattının Test Edilmesi
```bash
python -c "from src.data.preprocessing import load_ptbxl_metadata; df, _ = load_ptbxl_metadata('data/raw/ptb-xl'); print('Toplam kayit:', len(df)); print(df['diagnostic_superclass'].value_counts().head())"
```

### 3. Model Mimarilerinin Test Edilmesi
```bash
python -c "import torch; from src.models.simple_cnn1d import SimpleCNN1D; from src.models.xresnet1d import xresnet1d50; x=torch.randn(2, 12, 1000); m1=SimpleCNN1D(12, 5); m2=xresnet1d50(12, 5); print('CNN Cikis:', m1(x).shape); print('ResNet Cikis:', m2(x).shape)"
```

---

## 📌 Sıradaki Aşamalar
Proje takvimine göre sıradaki aşamada `experiments/train.py` kullanılarak modeller eğitilecek; ardından validasyon katında Sıcaklık Ölçekleme (Temperature Scaling) ve MC Dropout ile kalibrasyon deneyleri yürütülecektir.
