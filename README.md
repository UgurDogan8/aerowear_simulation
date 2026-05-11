# AeroWear Simülasyon Projesi

Bu proje, AeroWear adlı yakaya takılabilir akıllı hava kalitesi rozeti için Python tabanlı bir eğitim/proje simülasyonudur. Simülasyon; PM2.5, VOC/IAQ, sıcaklık, nem, yazılımsal kalibrasyon, olay-tetiklemeli örnekleme, BLE veri aktarımı, alarm üretimi ve pil tüketimi davranışlarını tek komutla üretir.

Bu çalışma gerçek klinik karar sistemi değildir; tıbbi teşhis, tedavi veya kişisel sağlık önerisi amacıyla kullanılmamalıdır.

## Projenin Amacı

AeroWear_TEYDEP1507 dokümanındaki hedeflerle uyumlu olarak, düşük maliyetli sensörlerden gelen ham verinin sıcaklık, nem, drift ve gürültü etkileri altında nasıl bozulabileceğini; doğrusal regresyon tabanlı kalibrasyonun hata metriklerini ne kadar iyileştirebileceğini; risk durumuna göre örnekleme ve BLE aktarım frekansının nasıl değişebileceğini göstermek amaçlanmıştır.

Simülasyon özellikle şu başlıkları kapsar:

- PM2.5 ve VOC mikro-ortam verisi üretimi
- Ham sensör gürültüsü, sıcaklık/nem sapması ve drift modeli
- LinearRegression ile PM2.5 kalibrasyonu
- Green / Yellow / Red risk sınıflandırması
- Alarm yorgunluğunu azaltan 60 saniyelik tekrar sınırlaması
- LOW_POWER, NORMAL ve HIGH_FREQUENCY örnekleme modları
- BLE gecikme ve paket kaybı simülasyonu
- 250 mAh pil ile 8 saatlik kullanım değerlendirmesi

## Kurulum

Python 3.10 veya üzeri gereklidir.

```bash
python -m pip install -r requirements.txt
```

## Çalıştırma

Proje klasöründe şu komutu çalıştırın:

```bash
python aerowear_simulation.py
```

Kod çalıştığında `outputs/` klasörü otomatik oluşturulur ve CSV, JSON, PNG çıktıları kaydedilir.

## Çıktı Dosyaları

- `outputs/aerowear_simulation_data.csv`: Saniyelik simülasyon verisi
- `outputs/simulation_summary.json`: Temel metriklerin JSON özeti
- `outputs/pm25_reference_raw_calibrated.png`: Referans, ham ve kalibre PM2.5 karşılaştırması
- `outputs/calibration_error_comparison.png`: Kalibrasyon öncesi/sonrası MAE ve RMSE bar grafiği
- `outputs/risk_level_timeline.png`: Green/Yellow/Red risk zaman çizelgesi
- `outputs/sampling_mode_timeline.png`: Örnekleme modu zaman çizelgesi
- `outputs/battery_percentage.png`: Pil yüzdesi değişimi
- `outputs/ble_latency_and_packet_loss.png`: BLE gecikmesi ve kayıp paketler
- `simulation_report.md`: Hocaya teslim edilebilecek kısa akademik rapor

## Simülasyonda Kullanılan Varsayımlar

- Toplam süre 8 saat, zaman çözünürlüğü 1 saniyedir.
- PM2.5 ve VOC referans değerleri gerçek ortam değerleri gibi kabul edilir.
- Ham sensör verisi; gain hatası, Gaussian gürültü, sıcaklık/nem sapması ve zamana bağlı drift içerir.
- PM2.5 kalibrasyonu için ilk %70 zaman dilimi eğitim, son %30 zaman dilimi test olarak ayrılır.
- Risk seviyesi PM2.5 ve VOC risklerinin maksimumudur.
- BLE paketleri yalnızca örnekleme yapılan saniyelerde gönderilir.
- Pil tüketimi çalışma modu, BLE gönderimi ve alarm durumuna göre saniyelik hesaplanır.

## Sonuçların Yorumlanması

`simulation_summary.json` dosyasında kalibrasyon öncesi/sonrası hata metrikleri, paket kaybı oranı, BLE gecikmesi, alarm gecikmesi, riskte geçirilen süreler ve kalan pil yüzdesi bulunur.

Kalibrasyon sonrası MAE veya RMSE iyileşmesinin %20 üzerinde olması, yazılımsal telafinin ham sensör hatasını azaltabildiğini gösterir. Paket kaybı oranının %5 altında kalması BLE aktarım modelinin hedefle uyumlu olduğunu; 8 saat sonunda pil kalması ise simülasyon düzeyinde günlük kullanım hedefinin sağlandığını gösterir.
