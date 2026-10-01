# Derin Öğrenme Kullanarak Türk İşaret Dili için Sürekli İşaret Tanıma ve Çeviri Sistemi

> **2209/A - Üniversite Öğrencileri Araştırma Projeleri Destek Programı**

Bu proje, Türk İşaret Dili (TİD) için sürekli işaret dili tanıma ve çeviri sistemi geliştirmeyi amaçlamaktadır. Geliştirilen sistem, standart bir akıllı telefon kamerasından gelen video girdisiyle çalışarak işaretleri tanımayı, metne çevirmeyi ve sesli Türkçe çıktı üretmeyi hedeflemektedir.

## Genel Bilgiler

| Alan | Bilgi |
|---|---|
| **Projenin Konusu** | Derin Öğrenme Kullanarak Türk İşaret Dili için Sürekli İşaret Tanıma ve Çeviri Sistemi |
| **Proje Yürütücüsü** | Jacques Asinyo |
| **Danışman** | Salih Bayar |
| **Proje Başlangıç ve Bitiş Tarihleri** | 2024 Güz - 2025 |
| **Sonuç Raporu Tarihi** | 17/07/2026 |

## Giriş

Bu proje, Türk İşaret Dili (TİD) için sürekli işaret dili tanıma ve çeviri sistemi geliştirmeyi amaçlamaktadır. Sistem, standart akıllı telefon kamerasından gelen video girişiyle çalışacak ve işaretleri doğru bir şekilde tanıyıp metin ve konuşmaya çevirecektir.

## Proje Konusu Hakkında Kısa Bilgi

Projenin temel amacı, işaret dilinin gerçek zamanlı çevirilerini sağlayarak Türkiye'deki sağır ve işitme engelli topluluk için erişilebilirliği artırmaktır.

İzole işaretleri ele alan sistemlerden farklı olarak bu proje, **sürekli işaret tanımaya** odaklanmaktadır.

Rapor kapsamında sistemin şu işaretleri tanıması sağlanmıştır:

- **5 temel kelime/ifade:** `merhaba`, `benim`, `adım`, `senin adın ne`, `soru`
- Raporda **3 harf** ifadesi kullanılmasına rağmen `J`, `A`, `C`, `K` olmak üzere **4 harf** listelenmiştir.
- Çevrilmiş metinden doğal sesli Türkçe konuşma üretmek için **Google Text-to-Speech (gTTS)** entegre edilmiştir.

## Rapor Dönemlerinde Yapılan Çalışmalar

- Belirlenen her bir kelime için **30 saniyelik 30 adet video kaydı** gerçekleştirilerek projeye özel bir veri seti oluşturulmuştur.
- Model performansını artırmak için **MediaPipe** kütüphanesinin el ve yüz anahtar noktası (`keypoint`) tespit yeteneklerinden faydalanılmıştır. Bu sayede parmak ve el pozisyonları gibi kritik bilgiler çıkarılmıştır.
- Çıkarılan veriler, modelin işleyebilmesi için **NumPy** kullanılarak veri dizilerine dönüştürülmüştür.
- Görüntü işleme süreçlerinde **OpenCV** kullanılmıştır.
- Model geliştirme sürecinde çeşitli derin öğrenme modelleri uygulanmış ve optimize edilmiştir. Raporda örnek olarak **Bidirectional CNN** belirtilmiştir.
- Sınırlı veri miktarı nedeniyle **Transformer** modeli beklenen performansı vermemiştir.
- Bu nedenle el konumlarını saptamak için **Evrişimli Sinir Ağı (CNN)**, zaman içerisindeki hareketleri işlemek için ise **Uzun Kısa Vadeli Bellek (LSTM)** mimarilerinin kombinasyonu tercih edilmiştir.

## Kullanılan Teknolojiler ve Yöntemler

- MediaPipe
- NumPy
- OpenCV
- CNN
- LSTM
- Transformer
- Bidirectional CNN
- Google Text-to-Speech (`gTTS`)
- Google Cloud Platform
- NVIDIA T4 GPU

## Sonuç

Yapılan çalışmalar sonucunda, Türk İşaret Dili'ne ait belirlenen kelimeleri ve harfleri kameradan gerçek zamanlı olarak algılayabilen bir **CNN-LSTM modeli** eğitilmiştir.

Eğitim verileri kullanılarak oluşturulan mimariye **text-to-speech** özelliği entegre edilmiş ve sistemin tanınan işaretleri eş zamanlı olarak sesli şekilde dile getirmesi sağlanmıştır.

## Çıktılar

Çalışan sistemin çıktısı ve sürece ait ilk sonuçlar video olarak paylaşılmıştır.


- **LinkedIn:** [Proje demosunu görüntüle](https://www.linkedin.com/feed/update/urn:li:activity:7347246085528084480/)


**Toplam:** 7.708,5 TL

## Proje Ekibi

**Proje Yürütücüsü:** Jacques Asinyo  
**Danışman:** Salih Bayar

## Rapor Tarihi

**17/07/2026**
