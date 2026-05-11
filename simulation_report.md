# AeroWear Yazılım Tabanlı Sensör, Kalibrasyon, BLE ve Pil Tüketimi Simülasyonu

## 1. Amaç

Bu çalışmanın amacı, AeroWear adlı yakaya takılabilir hava kalitesi rozeti için gerçek donanım üretimi öncesinde yazılım tabanlı bir ön değerlendirme yapmaktır. Simülasyon; PM2.5, VOC/IAQ, sıcaklık, nem, kalibrasyon, BLE veri aktarımı, alarm davranışı ve pil tüketimi bileşenlerini birlikte ele alır.

Bu çalışma gerçek klinik karar sistemi değildir; tıbbi teşhis veya tedavi önerisi üretmez.

## 2. Simülasyon Senaryosu

Toplam 8 saatlik kullanım 1 saniye çözünürlükle modellenmiştir. Mikro-ortam akışı normal iç ortam, trafik yakın çevresi, temizlik ürünü kullanımı, yemek pişirme/kapalı ortam ve kampüs/açık alan yürüyüşü bloklarından oluşur.

TEYDEP proje dokümanındaki hedeflere uygun olarak sistem; düşük maliyetli PM2.5 ve VOC sensörleri, sıcaklık/nem telafisi, olay-tetiklemeli örnekleme, BLE aktarımı ve günlük kullanım için pil yönetimi varsayımlarıyla modellenmiştir.

## 3. Kullanılan Değişkenler

Temel değişkenler `reference_pm25`, `raw_pm25`, `calibrated_pm25`, `reference_voc`, `raw_voc`, `temperature_c`, `humidity_percent`, `final_risk`, `sampling_mode`, `ble_latency_ms`, `ble_packet_lost`, `current_consumption_mA` ve `battery_percent` alanlarıdır. Referans değerler gerçek ortam değeri gibi kabul edilmiş; ham sensör verisine gain hatası, Gaussian gürültü, sıcaklık/nem biası ve zamana bağlı drift eklenmiştir.

## 4. Kalibrasyon Yaklaşımı

PM2.5 kalibrasyonu için LinearRegression modeli kullanılmıştır. Özellikler `raw_pm25`, `temperature_c`, `humidity_percent` ve `elapsed_hours`; hedef değişken ise `reference_pm25` olarak belirlenmiştir. Veri zamana göre ayrılmış, ilk %70 eğitim ve son %30 test olarak kullanılmıştır.

Test kümesinde MAE 10.16 değerinden 3.08 değerine, RMSE ise 11.60 değerinden 3.73 değerine düşmüştür. MAE iyileşmesi %69.7, RMSE iyileşmesi %67.9 olarak hesaplanmıştır.

## 5. Olay-Tetiklemeli Örnekleme Mantığı

Risk seviyesi Green olduğunda LOW_POWER modunda 60 saniyede bir, Yellow olduğunda NORMAL modunda 10 saniyede bir, Red olduğunda HIGH_FREQUENCY modunda her saniye örnekleme yapılmıştır. Bu yaklaşım, yüksek riskte daha sık veri üretirken düşük riskte pil tüketimini azaltmayı amaçlar.

Simülasyonda Green süresi 16868 saniye, Yellow süresi 4951 saniye ve Red süresi 6981 saniyedir.

## 6. BLE Veri Aktarım Simülasyonu

BLE paketi yalnızca örnekleme yapılan saniyelerde gönderilmiş kabul edilmiştir. Toplam 10828 BLE paketi üretilmiş, paket kaybı oranı %3.52 olarak bulunmuştur. Ortalama BLE gecikmesi 302.7 ms, 95. yüzdelik gecikme 492.5 ms olarak hesaplanmıştır.

## 7. Pil Tüketimi Modeli

Pil kapasitesi 250 mAh kabul edilmiştir. LOW_POWER, NORMAL ve HIGH_FREQUENCY modları için sırasıyla 6 mA, 14 mA ve 28 mA temel akım tüketimi kullanılmıştır. BLE gönderimi olan saniyelerde +4 mA, alarm üretilen saniyelerde +8 mA eklenmiştir.

8 saat sonunda kalan pil yüzdesi %54.1; ortalama akıma göre tahmini çalışma süresi 17.4 saattir.

## 8. Bulgular

Kalibrasyon sonrası hata metriklerinde hedeflenen %20 iyileşme eşiği aşılmıştır. BLE paket kaybı hedeflenen %5 sınırının altında kalmıştır. Alarm algoritması toplam 424 bildirim üretmiş, ortalama alarm gecikmesi 0.49 saniye ve maksimum alarm gecikmesi 0.88 saniye olmuştur.

Üretilen grafikler `outputs/` klasöründe yer almaktadır: PM2.5 karşılaştırması, hata karşılaştırması, risk zaman çizelgesi, örnekleme modu, pil yüzdesi ve BLE gecikme/kayıp grafiği.

## 9. Sonuç ve Yorum

8 saatlik kullanım hedefi simülasyon düzeyinde sağlandı. Kalibrasyon, ham sensör verisindeki sıcaklık/nem kaynaklı sapma ve drift etkilerini azaltarak ölçüm doğruluğunu belirgin şekilde iyileştirmiştir. Olay-tetiklemeli örnekleme, riskli anlarda yüksek frekanslı izlemeye geçerken düşük riskli sürelerde enerji tüketimini sınırlamıştır.

Bu simülasyon, gerçek donanım üretimi öncesinde AeroWear sisteminin sensör verisi işleme, kalibrasyon, risk sınıflandırması, BLE veri aktarımı ve enerji tüketimi davranışlarının ön değerlendirmesini sağlamaktadır.

## 10. Sınırlılıklar

Bu çalışma sentetik veri üretimine dayanır ve klinik validasyon yerine geçmez. Sensör yaşlanması, gövde hava akışı, gerçek BLE parazitleri, kullanıcı hareketi, konum değişimi ve kişiye özel tıbbi eşikler basitleştirilmiş olarak temsil edilmiştir. Gerçek prototip aşamasında referans cihaz eşleşmesi, kontrollü ortam testi ve saha doğrulaması gereklidir.
