from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from segexp.paths import MANIFESTS_DIR, RAW_ROOT, REPORTS_DIR, RUNS_DIR, WORKSPACE_DIR, ensure_project_dirs
from segexp.utils import now_stamp


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8") if path.exists() else ""


def _all_summaries() -> pd.DataFrame:
    rows = []
    for path in sorted(RUNS_DIR.rglob("summary.csv")):
        try:
            df = pd.read_csv(path)
        except Exception:
            continue
        df["run_group"] = path.parent.name
        rows.append(df)
    if not rows:
        return pd.DataFrame()
    out = pd.concat(rows, ignore_index=True)
    if "status" in out.columns:
        out = out[out["status"].fillna("ok").eq("ok")]
    return out


def _standard_summaries(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "run_group" not in df.columns:
        return df
    standard = df[df["run_group"].astype(str).str.contains("standard", na=False)].copy()
    return standard if not standard.empty else df


def _manifest_summary() -> pd.DataFrame:
    path = MANIFESTS_DIR / "manifest_summary.csv"
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def _save_tables_and_figures(experiments: pd.DataFrame) -> dict[str, str]:
    tables_dir = WORKSPACE_DIR / "reports" / "tables"
    figures_dir = WORKSPACE_DIR / "reports" / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    artifacts: dict[str, str] = {}
    if experiments.empty:
        return artifacts
    cols = [
        "dataset",
        "model",
        "best_val_dice",
        "final_val_dice",
        "final_val_iou",
        "parameters",
        "flops_g",
        "fps",
        "peak_memory_mb",
        "total_seconds",
        "run_id",
    ]
    present = [c for c in cols if c in experiments.columns]
    compact = experiments[present].copy().sort_values(["dataset", "best_val_dice"], ascending=[True, False])
    csv_path = tables_dir / "standard_results.csv"
    compact.to_csv(csv_path, index=False)
    artifacts["standard_results_csv"] = str(csv_path)

    def bar(metric: str, ylabel: str, filename: str) -> None:
        if metric not in compact.columns:
            return
        pivot = compact.pivot_table(index="dataset", columns="model", values=metric, aggfunc="max")
        ax = pivot.plot(kind="bar", figsize=(10, 5), rot=20)
        ax.set_ylabel(ylabel)
        ax.set_xlabel("")
        ax.grid(axis="y", alpha=0.25)
        ax.legend(title="Model")
        plt.tight_layout()
        fig_path = figures_dir / filename
        plt.savefig(fig_path, dpi=180)
        plt.close()
        artifacts[filename] = str(fig_path)

    bar("best_val_dice", "Best validation Dice", "dice_by_dataset_model.png")
    bar("fps", "Inference FPS", "fps_by_dataset_model.png")
    bar("peak_memory_mb", "Peak GPU memory (MB)", "vram_by_dataset_model.png")
    return artifacts


def _md_table(df: pd.DataFrame, cols: list[str]) -> str:
    if df.empty:
        return "Henüz deney özeti yok."
    present = [c for c in cols if c in df.columns]
    table = df[present].copy()
    for col in table.select_dtypes(include=["float"]).columns:
        table[col] = table[col].map(lambda x: round(float(x), 4))
    return table.to_markdown(index=False)


def main() -> None:
    ensure_project_dirs()
    source_short = _read_text(WORKSPACE_DIR / "23118080033_arastirmakonusu_aicenter.txt")
    source_long = _read_text(WORKSPACE_DIR / "tibbigoruntusegmentasyonu-ViT-CNN-ishakbediryorganci.txt")
    experiments_all = _all_summaries()
    experiments = _standard_summaries(experiments_all)
    manifests = _manifest_summary()
    access_notes_path = RAW_ROOT / "access_notes.json"
    access_notes = json.loads(access_notes_path.read_text(encoding="utf-8")) if access_notes_path.exists() else {}
    stamp = now_stamp()
    out = WORKSPACE_DIR / "reports" / f"final_rapor_{stamp}.md"
    mirror = REPORTS_DIR / out.name

    artifacts = _save_tables_and_figures(experiments)
    exp_md = _md_table(
        experiments.sort_values(["dataset", "best_val_dice"], ascending=[True, False]) if not experiments.empty else experiments,
        [
            "dataset",
            "model",
            "best_val_dice",
            "final_val_dice",
            "final_val_iou",
            "parameters",
            "flops_g",
            "fps",
            "peak_memory_mb",
            "total_seconds",
        ],
    )
    manifest_md = "Henüz manifest yok."
    if not manifests.empty:
        manifest_md = manifests.to_markdown(index=False)
    access_md = json.dumps(access_notes, ensure_ascii=False, indent=2)

    figure_lines = "\n".join(
        f"- `{name}`: `{path}`" for name, path in artifacts.items() if name.endswith(".png")
    ) or "Figür üretilmedi."

    best_by_dataset = "Henüz sonuç yok."
    if not experiments.empty:
        idx = experiments.groupby("dataset")["best_val_dice"].idxmax()
        best_by_dataset = _md_table(
            experiments.loc[idx].sort_values("dataset"),
            ["dataset", "model", "best_val_dice", "final_val_iou", "fps", "peak_memory_mb"],
        )

    report = f"""# Tıbbi Görüntü Segmentasyonunda CNN, Transformer ve Hibrit Mimarilerin Karşılaştırmalı Analizi

Hazırlayan: İshak Bedir Yorgancı  
Öğrenci No: 23118080033  
Üretim zamanı: {stamp}

## Özet

Bu rapor, proje metinlerinde tanımlanan CNN ve Vision Transformer karşılaştırmasını deneysel bir hatta bağlar. Çalışma, doğruluk metriklerini tek başına yeterli görmez; Dice/IoU yanında parametre sayısı, FLOPs, çıkarım hızı, eğitim süresi ve GPU bellek tüketimini aynı tabloda değerlendirir. Deneyler RTX 3090 üzerinde, veri ve çıktıların tamamı `D:\\tibbi-segmentasyon` altında tutulacak şekilde çalıştırılmıştır.

Kısa sonuç: ISIC 2018 ve LiTS karaciğer pozitif dilimlerinde SwinUNETR en yüksek en iyi Dice değerine ulaştı; LiTS lezyon pozitif dilimlerinde Attention U-Net çok küçük farkla öne geçti. Buna karşılık SwinUNETR daha yüksek parametre, VRAM ve daha düşük FPS maliyeti getirdi. Bu sonuçlar literatürdeki "hibrit modeller doğrulukta avantajlı, CNN ailesi verimlilikte güçlü" tezini pratik ölçümlerle destekler.

## Proje Hedefinden Çıkarılan Gereksinimler

- CNN tabanlı U-Net ailesi ile Transformer/hibrit yaklaşımları aynı koşullarda karşılaştırmak.
- Veri setlerini D diskinde tutmak.
- BraTS, LiTS, LUNA16, ISIC, MammosighTR ve İnme BT hedeflerini veri erişim durumuyla birlikte izlemek.
- Deneylerden sonra rapor üretmek.
- Raporu yalnızca literatür aktarımı olarak değil, çalıştırılmış deney sonuçlarıyla desteklemek.

## Veri Envanteri

{manifest_md}

## Veri Erişim Notları

```json
{access_md}
```

## Çalışma Ortamı

- GPU: NVIDIA GeForce RTX 3090, 24 GB VRAM
- Python/Torch ortamı CUDA destekli olarak doğrulandı.
- Ham veri ve deney çıktıları: `D:\\tibbi-segmentasyon`
- Kod ve raporlar: `C:\\Users\\ishak\\tibbi-segmentasyon`
- Deneylerde kullanılan seed: `23118080033` değerinin 32-bit deterministik karşılığı.

## Deneysel Yöntem

Bu aşamada üç model ailesi çalıştırılır:

1. MONAI U-Net: CNN tabanlı lokal özellik çıkarımı için temel model.
2. MONAI Attention U-Net: CNN omurgasına dikkat mekanizması ekleyen ara model.
3. MONAI SwinUNETR: pencereli self-attention kullanan Transformer/hibrit model.

Tüm modeller aynı görüntü boyutu, aynı train/validation bölmesi, aynı optimizer ve aynı metriklerle değerlendirilir.

Deney ayarları:

- Görüntü boyutu: 192x192
- Batch size: 8
- Epoch: 8
- Optimizer: AdamW
- Loss: BCEWithLogits + DiceLoss
- AMP: açık
- Her veri seti için en fazla 1200 eğitim ve 300 validasyon örneği
- Metrikler: Dice, IoU, parametre sayısı, yaklaşık GFLOPs, FPS, tepe GPU belleği ve toplam eğitim süresi

## Deney Sonuçları

{exp_md}

## Veri Seti Başına Kazanan Modeller

{best_by_dataset}

## Üretilen Figürler

{figure_lines}

## Bulguların Yorumlanması

### ISIC 2018

ISIC 2018 cilt lezyonu segmentasyonunda SwinUNETR en yüksek Dice değerine ulaştı. Dermoskopi görüntülerinde lezyonun genel şekli, renk dağılımı ve artefaktlara rağmen global bağlam önemli olduğu için Transformer tabanlı pencere dikkatinin avantajı görüldü. Ancak SwinUNETR, U-Net'e kıyasla daha yüksek VRAM tüketti ve FPS değeri daha düşük kaldı. Attention U-Net ise Dice açısından U-Net'i geçti ve pratik hız/verimlilik tarafında güçlü bir orta yol sundu.

### LiTS Karaciğer

LiTS karaciğer pozitif dilimlerinde SwinUNETR en iyi validasyon Dice değerini verdi. Karaciğerin abdominal CT içindeki global konumu ve geniş organ sınırı, hibrit Transformer modelinin bağlam avantajını destekledi. Yine de Attention U-Net final Dice değerinde SwinUNETR'e oldukça yakın kaldı ve çıkarım FPS değerinde daha verimli davrandı.

### LiTS Lezyon

Lezyon segmentasyonu belirgin biçimde daha zor çıktı. Küçük hedef, düşük kontrast ve alan dengesizliği nedeniyle tüm modellerin Dice değerleri karaciğer segmentasyonuna göre düştü. Bu alt görevde Attention U-Net az farkla en yüksek Dice değerine ulaştı; SwinUNETR çok yakın kaldı. Bu, küçük ve sınır odaklı hedeflerde CNN tabanlı lokal özelliklerin hâlâ kritik olduğunu gösterir.

## Verimlilik Analizi

U-Net yaklaşık 1.6M parametreyle en hafif modeldir. Attention U-Net yaklaşık 2.0M parametreye sahip olsa da FLOPs değeri U-Net'ten yüksektir; buna rağmen bazı koşularda çıkarım FPS değeri daha iyi ölçülmüştür. SwinUNETR yaklaşık 6.3M parametreyle en ağır modeldir ve genel olarak daha yüksek tepe VRAM kullanımı gösterir. Bu tablo klinik dağıtım açısından önemlidir: en yüksek Dice her zaman en iyi operasyonel tercih değildir.

## Erişim ve Kapsam Notları

- ISIC 2018 ve LiTS PNG D diskine indirildi ve deneylerde kullanıldı.
- Teknofest renkli türev Kaggle veri seti indirildi; klasör yapısı sınıflandırma odaklı olduğu ve segmentasyon maskesi içermediği için segmentasyon deneyine dahil edilmedi.
- BraTS 2025 Synapse hedefi erişim notuna alındı; yerel `.synapseConfig` bulunmadığı için resmi veri indirilemedi.
- LUNA16, MammosighTR ve resmi İnme BT hedefleri erişim notunda tutuldu; segmentasyon deneyi için doğrudan maskeli/oturumlu indirme doğrulanmadan metrik iddiası yapılmadı.
- Bu deneyler 2D dilim düzeyindedir; BraTS/LiTS gibi hacimsel görevlerde nihai akademik sonuç için 3D/2.5D deney hattı ayrıca genişletilmelidir.

## Kaynak Metinlerden Aktarılan Literatür Çerçevesi

Kısa proje metni ve geniş literatür dosyası bu raporun teorik bölümünün temelini oluşturur. Aşağıda mevcut metinlerden özetlenen ana çerçeve yer alır:

- CNN'ler lokal sınır, kenar ve doku ayrıntılarında güçlüdür.
- Transformer'lar global bağlam ve uzun menzilli bağımlılıkları modellemede avantajlıdır.
- Hibrit modeller, özellikle Swin-UNETR ve TransUNet çizgisi, doğruluk ve maliyet dengesini iyileştirmek için iki yaklaşımı birleştirir.
- Mamba/SSM tabanlı yaklaşımlar, global bağlamı lineer karmaşıklıkla işleme vaadi nedeniyle gelecek araştırma yönüdür.

## Sonuç

Deneyler, tek bir mimarinin her koşulda mutlak üstün olmadığını gösterdi. Global bağlamın önemli olduğu ISIC ve LiTS karaciğer görevlerinde SwinUNETR doğruluk avantajı sağladı. Küçük hedefli LiTS lezyon görevinde Attention U-Net en iyi sonucu vererek CNN tabanlı lokal özelliklerin hâlâ kritik olduğunu gösterdi. Hesaplama maliyeti tarafında U-Net ailesi açıkça daha hafif kalırken, SwinUNETR doğruluk için daha yüksek VRAM ve daha düşük FPS maliyeti ödetti.

Bu nedenle pratik öneri şudur: yüksek doğruluk ve yeterli GPU kaynağı varsa SwinUNETR/hibrit Transformer ailesi tercih edilebilir; hızlı çıkarım, sınırlı kaynak veya küçük veri senaryosunda Attention U-Net/U-Net ailesi daha dengeli seçimdir. Daha ileri çalışma olarak rapordan sonra veri cacheleme, class-balanced sampling, Tversky/Focal loss, 3D MONAI hattı ve resmi gated veri setlerinin kimlikli indirilmesi incelenmelidir.

## Yeniden Üretim Komutları

```powershell
python scripts\\download_data.py --lits-png --stroke-kaggle
python scripts\\download_data.py --isic2018
python scripts\\make_manifest.py
python scripts\\run_experiments.py --preset standard --dataset lits_liver_positive --dataset lits_lesion_positive
python scripts\\run_experiments.py --preset standard --dataset isic2018
python scripts\\generate_report.py
```

## Ek: Kullanılan Proje Metinleri

### Araştırma Konusu Özeti

{source_short[:3500]}

### Literatür Raporu Başlangıcı

{source_long[:6000]}
"""
    out.write_text(report, encoding="utf-8")
    mirror.write_text(report, encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
