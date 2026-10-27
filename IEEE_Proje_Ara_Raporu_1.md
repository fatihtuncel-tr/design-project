# Derivasyon Sayısı Azaltılmış EKG Sınıflandırmasında Belirsizlik Kalibrasyonu ve Yaş Grubu Adaleti: Ara Rapor

**Ders:** Tasarım Projesi (Design Project)  
**Rapor Türü:** Ara Rapor (Dönem İçi Gelişme Raporu)  
**Tarih:** 2026-10-27  

---

### Proje Ekibi (2 Öğrenci)

| 1. Öğrenci Adı Soyadı | 2. Öğrenci Adı Soyadı |
| :---: | :---: |
| **Öğrenci No:** 20XXXXXXX<br>Bilgisayar Mühendisliği Bölümü<br>[Üniversite Adı]<br>ogrenci1@universite.edu.tr | **Öğrenci No:** 20XXXXXXX<br>Bilgisayar Mühendisliği Bölümü<br>[Üniversite Adı]<br>ogrenci2@universite.edu.tr |

---

## Özet (*Abstract*)
*Özet*—Akıllı saatler ve giyilebilir sağlık cihazları günlük yaşamda kalp ritmi takibi için giderek daha fazla kullanılmaktadır. Ancak bu cihazlar, hastanelerde kullanılan 12 derivasyonlu standart EKG cihazlarının aksine genellikle yalnızca tek bir derivasyon (Lead I) üzerinden sinyal alabilmektedir. Derin öğrenme modelleri derivasyon sayısı azaldığında yalnızca doğruluk kaybı yaşamakla kalmamakta, aynı zamanda tahminlerine aşırı güvenerek yanlış teşhislerde bile yüksek olasılıklar üretebilmektedir. Bu durum, özellikle kalp sinyalleri daha değişken olan 75 yaş ve üzeri yaşlı hastalarda ciddi bir güvenilirlik sorunu yaratmaktadır. Bu projede amacımız, derivasyon sayısı 12'den 1'e düşürüldüğünde modellerin kalibrasyon hatasının (ECE) ve yaş grupları arasındaki güvenilirlik farkının nasıl etkilendiğini incelemek ve kalibrasyon yöntemleriyle bu farkı azaltmaktır. Bu ara raporda, projenin ilk adımı olan veri hazırlığı ve altyapı çalışmaları sunulmaktadır. Çalışma kapsamında; Python, PyTorch ve WFDB kütüphaneleriyle geliştirme ortamı kurulmuş, 21.799 kayıttan oluşan PTB-XL veri seti indirilmiştir. Sinyaller 100 Hz frekansa getirilmiş, taban çizgisi gürültüleri temizlenmiş ve z-score yöntemiyle normalize edilmiştir. Tanı etiketleri 5 ana kardiyak sınıfa (NORM, MI, STTC, CD, HYP) eşlenmiş ve hastalar 4 yaş grubuna ayrılmıştır. Veri seti eğitim, doğrulama ve test olarak 10 katlı tabakalı yapıya bölünmüş, disk okuma hızını artırmak için .npz formatında önbelleğe alınmıştır. Ayrıca kullanılacak SimpleCNN1D ve xResNet1d modelleri kodlanmıştır. Eğitim ve kalibrasyon deneyleri sonraki aşamada gerçekleştirilecektir.

*Anahtar Kelimeler*—EKG Sınıflandırma, Derivasyon Azaltma, Model Kalibrasyonu, ECE, Yaş Grupları, PTB-XL, PyTorch, Ara Rapor.

---

## I. Giriş ve Motivasyon (*Introduction*)
Kardiyovasküler hastalıkların erken evrede belirlenmesinde EKG sinyallerinin düzenli takibi hayati önem taşır. Son yıllarda akıllı saatler (Apple Watch, Withings vb.) ve göğüs bantları sayesinde kullanıcılar hastaneye gitmeye gerek kalmadan günlük hayatlarında tek kanallı EKG kaydı alabilmektedir [1]. Fakat hastanelerde tanı için kullanılan altın standart EKG cihazları kalbin elektriksel hareketini 12 farklı uzamsal açıdan kaydeder. Giyilebilir cihazlarda tek derivasyona inildiğinde sinyaldeki uzamsal bilgi miktarı önemli ölçüde azalmaktadır.

Yapay zeka modelleri literatürde çoğunlukla sadece genel doğruluk veya AUROC oranlarıyla değerlendirilir. Ancak sağlık uygulamalarında bir modelin doğru tahmin yapması kadar, ne zaman yanıldığını veya ne kadar emin olduğunu bilmesi de gereklidir [8]. Kalibre edilmemiş derin öğrenme modelleri, eksik bilgi içeren tek derivasyonlu sinyallerde yanlış tanı koyarken bile %95 gibi yüksek bir güven skoru verebilmektedir (aşırı özgüven sorunu). Literatürdeki son çalışmalar, derivasyon azalmasının yarattığı performans kaybının tüm yaş gruplarında eşit olmadığını, özellikle 75 yaş üstü geriatrik hastalarda kaybın gençlere göre çok daha yüksek olduğunu göstermektedir [2].

Projemizde şu temel araştırma sorularına odaklanıyoruz:
1. Derivasyon sayısı 12'den 1'e düştüğünde modelin kalibrasyon hatası (ECE) doğruluğundan daha hızlı mı bozuluyor?
2. Sıcaklık ölçekleme ve Monte Carlo Dropout yöntemleri tek derivasyonlu modellerin aşırı güvenini ne kadar düzeltebilir?
3. Kalibrasyon bozulması yaş gruplarında (<40, 40-60, 60-75, 75+) ne kadar farklılık gösteriyor?

Bu ara raporda; çalışma ortamının kurulması, PTB-XL veri setinin indirilip ön-işlenmesi, etiket eşleme, yaş gruplama, hızlı veri önbellekleme ve model mimarilerinin kodlanması anlatılmaktadır.

---

## II. İlgili Çalışmalar (*Related Work*)
EKG sınıflandırmasında derin öğrenme alanında öne çıkan temel çalışmalar şunlardır:

* **PTB-XL Veri Seti ve Kıyaslamalar:** Wagner ve ark. (2020), 18.885 hastadan alınan 21.837 adet 12 derivasyonlu klinik EKG kaydını içeren açık PTB-XL veri setini yayınlamıştır [1]. Strodthoff ve ark. (2020) ise bu veri seti üzerinde xResNet1d ve CNN modellerini test ederek 10 katlı resmi bölümlemeyi standart hale getirmiştir [6].
* **Derivasyon Azaltma ve Yaş Disparitesi:** Tiruwa ve ark. (2026), derivasyon sayısı 1'e indirildiğinde 75+ yaş grubunda doğruluğun 14 puandan fazla düştüğünü, genç grupta ise bu düşüşün 1 puanın altında kaldığını belirtmiştir [2]. Ancak bu çalışmada model güveni ve kalibrasyon incelenmemiştir.
* **Model Kalibrasyonu ve Belirsizlik:** Guo ve ark. (2017), derin ağların aşırı özgüvenli olduğunu ve validasyon verisi üzerinden yapılan Sıcaklık Ölçeklemenin (Temperature Scaling) bu sapmayı etkili biçimde düzelttiğini göstermiştir [8]. Gal ve Ghahramani (2016) ise test anında dropout katmanlarını aktif bırakarak (MC Dropout) modelin belirsizliğinin hesaplanabileceğini ortaya koymuştur [9].

---

## III. Geliştirme Ortamı ve Kütüphaneler
Proje çalışmaları Python 3.10 programlama dili kullanılarak yürütülmektedir. Derin öğrenme modelleri ve GPU optimizasyonu için PyTorch tercih edilmiştir. Kullandığımız başlıca kütüphaneler ve projedeki görevleri Tablo I'de özetlenmiştir:

#### TABLO I: Projede Kullanılan Temel Kütüphaneler ve Kullanım Amaçları
| Kütüphane | Sürüm | Kullanım Amacı |
| :--- | :---: | :--- |
| **PyTorch (`torch`, `nn`)** | `>= 2.0.0` | 1B konvolüsyonel modeller, GPU eğitimi ve kayıp fonksiyonları |
| **WFDB (`wfdb`)** | `>= 4.1.0` | PhysioNet formatındaki EKG sinyallerinin (.dat/.hea) doğrudan okunması |
| **Pandas & NumPy** | `>= 2.0.0` | Hasta demografisi, tanı etiketleri ve sinyal matris işlemleri |
| **TorchMetrics** | `>= 1.0.0` | Çok etiketli AUROC, Macro F1 ve sınıflandırma başarım ölçümü |
| **Scikit-Learn / SciPy** | `>= 1.2.0` | Katmanlı veri örnekleme, sinyal filtreleme ve z-score normalizasyonu |
| **Seaborn / Matplotlib** | `>= 0.12.0` | Kalibrasyon eğrileri ve yaş grubu grafiklerinin çizimi |

---

## IV. Veri Seti Hazırlığı ve Ön-İşleme (Ara Rapor Kapsamı)
Projenin bu ilk aşamasında üzerinde en çok çalışılan bölüm EKG verilerinin indirilmesi, filtrelenmesi, kardiyolojik sınıflara eşlenmesi ve eğitime hazır hale getirilmesidir.

### A. PTB-XL Veri Seti
Çalışmada PhysioNet platformundaki açık erişimli **PTB-XL (v1.0.3)** veri seti kullanılmıştır [1]. Veri setinde 18.885 hastadan alınmış 21.799 adet 10 saniyelik 12 derivasyonlu EKG bulunmaktadır. Her kaydın yanında hastanın yaşı, cinsiyeti ve uzman hekimler tarafından girilmiş SCP-ECG tanı kodları yer almaktadır.

### B. Sinyal Ön-İşleme Adımları
Ham sinyallerin doğrudan modele verilmesi gürültü ve boyut nedeniyle uygun değildir. Bu nedenle `src/data/preprocessing.py` modülü içinde şu adımlar uygulanmıştır:
1. **100 Hz Yeniden Örnekleme:** Orijinal 500 Hz kayıtlar 100 Hz frekansa getirilerek 10 saniyelik sinyal tam 1.000 veri noktasına indirgenmiştir. Bu işlem hesaplama yükünü azaltırken dalga formunun temel morfolojisini korur.
2. **Kanal Başına Z-Score Normalizasyonu:** Farklı hastaların cilt direnci ve voltaj seviyelerini eşitlemek için her derivasyon kanalı kendi ortalaması ve standart sapmasıyla normalize edilmiştir:  
   `x_norm = (x - mean) / std`
3. **Taban Çizgisi Filtreleme ve Temizlik:** Solunum ve hareket kaynaklı taban çizgisi kaymaları filtrelenmiş, eksik veya hatalı değerler temizlenmiştir.

### C. Derivasyon Senaryoları
Farklı cihaz türlerini temsil etmek amacıyla 4 ayrı derivasyon seçeneği hazırlanmıştır:
* **12 Derivasyon:** Hastanedeki tam klinik kayıt (I, II, III, aVR, aVL, aVF, V1-V6).
* **6 Derivasyon:** Kol ve bacak elektrotları (ekstremite derivasyonları).
* **2 Derivasyon:** Göğüs bantları (Derivasyon I ve II).
* **1 Derivasyon:** Akıllı saatlerdeki tek kanal standardı (Derivasyon I).

### D. Tanısal Etiket Eşleme (5 Süper Sınıf)
PTB-XL içerisindeki onlarca farklı klinik tanı kodu, uluslararası kıyaslama standardına uygun olarak 5 ana sınıfta toplanmıştır:
1. **NORM:** Normal EKG bulguları.
2. **MI:** Miyokard Enfarktüsü (Kalp krizi bulguları).
3. **STTC:** ST ve T dalgası morfolojik değişiklikleri.
4. **CD:** İletim bozuklukları (bloklar).
5. **HYP:** Ventrikül veya atriyal hipertrofi (kalp kası kalınlaşması).

Bir hastada aynı anda birden fazla durum bulunabildiği için (örneğin hem MI hem CD), hedef değişkenler çok etiketli (multi-label) ikili vektör şeklinde yapılandırılmıştır.

### E. Yaş Grupları ve Tabakalı Bölümleme
Yaşlı hastalardaki durumu inceleyebilmek için hastalar 4 ayrı yaş grubuna ayrılmıştır:
* `< 40 Yaş` (Genç yetişkinler)
* `40 – 60 Yaş` (Orta yaş)
* `60 – 75 Yaş` (İleri yaş)
* `75+ Yaş` (Geriatrik grup - morfolojik varyansı en yüksek kohort)

Veri seti, PTB-XL'in resmi 10 katlı tabakalı yapısına göre bölünmüştür (Tablo II):

#### TABLO II: PTB-XL Veri Seti Kat Dağılımı ve Kullanım Amacı
| Bölüm | Katlar | Kayıt Sayısı | Kullanım Amacı |
| :--- | :---: | :---: | :--- |
| **Eğitim (Train)** | Kat 1 – 8 | 17.418 | Model parametrelerinin eğitilmesi |
| **Doğrulama (Val)** | Kat 9 | 2.183 | Sıcaklık ölçekleme ve F1 eşik ayarı |
| **Test (Hold-out)** | Kat 10 | 2.198 | Bağımsız test ve yaş grubu adalet analizi |
| **Toplam Veri** | Kat 1 – 10 | **21.799** | *75+ yaş kayıt sayısı: 4.980* |

### F. Hızlı Bellek Önbellekleme (`.npz` Dosyaları)
21.799 adet EKG dosyasının her eğitim turunda diskten tek tek okunması eğitimi aşırı yavaşlatmaktadır. Bu darboğazı çözmek için `src/data/cache_dataset.py` aracı yazılmış ve tüm sinyaller önceden işlenerek sıkıştırılmış NumPy dizileri halinde diske kaydedilmiştir:
* `train_data.npz` (478.6 MB, 17.418 kayıt)
* `val_data.npz` (61.1 MB, 2.183 kayıt)
* `test_data.npz` (61.7 MB, 2.198 kayıt)

Bu sayede veri yükleme hızı yaklaşık 15 kat artırılmış ve eğitim süresi kısaltılmıştır.

---

## V. Model Mimarileri ve Metrikler
Veri hazırlığı tamamlandıktan sonra, bir sonraki aşamada eğitilmek üzere iki farklı 1B derin öğrenme modeli kodlanmıştır:

1. **SimpleCNN1D:** 4 konvolüsyonel bloktan oluşan hafif bir modeldir (~300.000 parametre). Her blokta Conv1d, BatchNorm1d, ReLU, MaxPool1d ve Dropout(0.2) yer alır. Düşük işlem gücüne sahip cihazlar için referans modeldir.
2. **xResNet1d:** EKG zaman serilerinde literatürde en çok kullanılan artık bağlantılı (residual) derin ağ mimarisidir (~1.800.000 parametre) [6].

### Değerlendirme Metrikleri
Eğitimde kayıp fonksiyonu olarak çok etiketli ikili çapraz entropi (BCEWithLogitsLoss) kullanılacaktır. Modellerin başarımını ve güvenilirliğini ölçmek için şu metrikler hesaplanacaktır:
* **Ayırt Edicilik:** Macro AUROC ve Macro F1 skoru.
* **Beklenen Kalibrasyon Hatası (ECE):** Tahmin olasılıkları 15 eşit aralığa bölünerek her kutudaki ortalama güven ile gerçek doğruluk arasındaki fark toplanır:

```
ECE = Σ (|B_m| / N) · |acc(B_m) - conf(B_m)|
```

* **Brier Skoru:** Tahmin edilen olasılıklar ile gerçek etiketler arasındaki ortalama karesel hata:

```
Brier = (1 / N) · Σ (p_i - y_i)²
```

* **Yaş Grubu Adalet Açığı (ΔECE):** 75 yaş ve üzeri grup ile 40 yaş altı genç grup arasındaki kalibrasyon hatası farkı:

```
ΔECE = ECE(75+) - ECE(<40)
```

---

## VI. İş Planı ve Sıradaki Çalışmalar
Proje 14 haftalık takvime göre yürütülmektedir. Bu ara rapor itibarıyla ilk iş paketi tamamlanmış olup sıradaki adımlar Tablo III'te gösterilmiştir:

#### TABLO III: Proje Takvimi ve İlerleme Durumu
| İş Paketi | Kapsam ve Görevler | Süreç | Mevcut Durum |
| :--- | :--- | :---: | :---: |
| **İP1: Veri ve Altyapı** | PTB-XL indirme, 100Hz ön-işleme, etiket eşleme, .npz cache | Hafta 1 – 3 | **Tamamlandı** |
| **İP2: Model Eğitimleri** | 12, 6, 2 ve 1 derivasyon için model eğitimleri ve AUROC ölçümü | Hafta 4 – 7 | **Sıradaki Aşama** |
| **İP3: Kalibrasyon Deneyleri** | Kat 9 ile Sıcaklık Ölçekleme ve testte MC Dropout uygulanması | Hafta 7 – 9 | Planlandı |
| **İP4: Dış Veri ve Alan Kayması** | CinC 2021 ve Chapman veri setlerinde test, konformal tahmin | Hafta 9 – 11 | Planlandı |
| **İP5: Yaş Adaleti ve Final Raporu** | Yaş grupları analizi (ΔECE), hekime yönlendirme ve final raporu | Hafta 12 – 14 | Planlandı |

### Sonraki Aşamada Yapılacak Çalışmalar
1. İP2 kapsamında 12, 6, 2 ve 1 derivasyon modelleri eğitilecek ve derivasyon azaldığında test kümesindeki ayrım gücü (AUROC, F1) ölçülecektir.
2. İP3 kapsamında Kat 9 validasyon logitleri kullanılarak Sıcaklık Ölçekleme parametreleri optimize edilecek; ardından MC Dropout ile modellerin kalibrasyon hatası (ECE) hesaplanacaktır.
3. İP4 ve İP5 aşamalarında ise modellerin dış veri setlerinde (CinC 2021, Chapman) performansı incelenecek ve 75 yaş üstü hastalar ile genç hastalar arasındaki adalet farkı (ΔECE) karşılaştırılacaktır.

---

## VII. Sonuç ve Mevcut Durum
Tasarım Projesi'nin bu ara rapor döneminde hedeflenen altyapı ve veri hazırlığı çalışmaları tamamlanmıştır. Çalışma ortamı kurulmuş, PTB-XL veri seti temizlenip 100 Hz'e dönüştürülmüş, 5 tanı sınıfı ve 4 yaş grubu tanımlanmıştır. Hızlı bellek önbellekleri (.npz) oluşturulmuş ve model mimarileri kodlanarak eğitim aşamasına hazır hale getirilmiştir. Sıradaki süreçte modellerin eğitimi ve kalibrasyon deneylerine geçilecektir.

---

## Teşekkür (*Acknowledgment*)
Bu çalışma Tasarım Projesi dersi kapsamında hazırlanmıştır. PTB-XL veri setini sağlayan PhysioNet platformuna ve çalışmamızda bize yol gösteren ders danışmanımıza teşekkür ederiz.

---

## Kaynaklar (*References*)
* [1] P. Wagner et al., "PTB-XL, a large publicly available electrocardiography dataset," *Scientific Data*, vol. 7, no. 1, p. 154, 2020.
* [2] K. R. Tiruwa et al., "Wearable-Grade Lead Reduction Disproportionately Degrades ECG AI Performance in Elderly Patients: Evidence from PTB-XL and MIT-BIH," *openRxiv preprint*, 2026.
* [3] H. Asgharnezhad et al., "UCTECG-Net: Uncertainty-aware Convolution Transformer ECG Network for Arrhythmia Detection," *arXiv:2602.16216*, 2026.
* [4] N. Strodthoff et al., "PTB-XL+, a comprehensive electrocardiographic feature dataset," *Scientific Data*, vol. 10, no. 1, p. 279, 2023.
* [5] M. A. Reyna et al., "Issues in the automated classification of multilead ecgs using heterogeneous labels and populations," *Physiological Measurement*, vol. 43, no. 8, p. 085002, 2022.
* [6] N. Strodthoff et al., "Deep Learning for ECG Analysis: Benchmarks and Insights from PTB-XL," *IEEE Journal of Biomedical and Health Informatics*, vol. 25, no. 5, pp. 1519-1528, 2020.
* [7] A. N. Angelopoulos and S. Bates, "A Gentle Introduction to Conformal Prediction and Distribution-Free Uncertainty Quantification," *arXiv:2107.07511*, 2021.
* [8] C. Guo, G. Pleiss, Y. Sun, and K. Q. Weinberger, "On Calibration of Modern Neural Networks," in *Proc. 34th Int. Conf. Machine Learning (ICML)*, pp. 1321-1330, 2017.
* [9] Y. Gal and Z. Ghahramani, "Dropout as a Bayesian approximation: Representing model uncertainty in deep learning," in *Proc. 33rd Int. Conf. Machine Learning (ICML)*, pp. 1050-1059, 2016.
* [10] A. L. Goldberger et al., "PhysioBank, PhysioToolkit, and PhysioNet: Components of a new research resource for complex physiologic signals," *Circulation*, vol. 101, no. 23, pp. e215-e220, 2000.
