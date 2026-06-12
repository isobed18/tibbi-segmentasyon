# Tıbbi Görüntü Segmentasyonu Karşılaştırmalı Deney Projesi

Bu çalışma, CNN tabanlı U-Net ailesi ile Transformer/hibrit segmentasyon mimarilerini aynı makinede karşılaştırmak için hazırlanmış yeniden üretilebilir deney altyapısıdır.

## Dizinler

- Kod, konfigürasyon ve rapor taslakları: `C:\Users\ishak\tibbi-segmentasyon`
- Ham veri, işlenmiş veri, checkpoint ve çalışma çıktıları: `D:\tibbi-segmentasyon`

## İlk Komutlar

```powershell
python scripts\download_data.py --all
python scripts\make_manifest.py
python scripts\run_experiments.py --preset smoke
python scripts\run_experiments.py --preset standard
python scripts\generate_report.py
```

`smoke` modu kod ve CUDA hattını hızlı doğrular. `standard` modu erişilebilen gerçek veri setleri üzerinde modelleri eğitir ve sonuçları `D:\tibbi-segmentasyon\runs` altına yazar.

## Karşılaştırılan Model Aileleri

- CNN: MONAI U-Net
- CNN dikkatli varyant: MONAI Attention U-Net
- Transformer/hibrit: MONAI SwinUNETR

## Ölçülen Metrikler

- Dice
- IoU
- Parametre sayısı
- Yaklaşık FLOPs
- Eğitim süresi
- Çıkarım gecikmesi/FPS
- Tepe GPU bellek kullanımı

