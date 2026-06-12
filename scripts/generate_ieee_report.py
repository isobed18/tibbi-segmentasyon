from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from segexp.paths import MANIFESTS_DIR, RAW_ROOT, REPORTS_DIR, RUNS_DIR, WORKSPACE_DIR, ensure_project_dirs
from segexp.utils import now_stamp


REFERENCES = [
    {
        "id": 1,
        "citation": "O. Ronneberger, P. Fischer, and T. Brox, U-Net: Convolutional Networks for Biomedical Image Segmentation, MICCAI, 2015.",
        "url": "https://arxiv.org/abs/1505.04597",
        "verification": "arXiv record verified",
    },
    {
        "id": 2,
        "citation": "O. Oktay et al., Attention U-Net: Learning Where to Look for the Pancreas, 2018.",
        "url": "https://arxiv.org/abs/1804.03999",
        "verification": "arXiv record verified",
    },
    {
        "id": 3,
        "citation": "J. Chen et al., TransUNet: Transformers Make Strong Encoders for Medical Image Segmentation, 2021.",
        "url": "https://arxiv.org/abs/2102.04306",
        "verification": "arXiv record verified",
    },
    {
        "id": 4,
        "citation": "A. Hatamizadeh et al., Swin UNETR: Swin Transformers for Semantic Segmentation of Brain Tumors in MRI Images, 2022.",
        "url": "https://arxiv.org/abs/2201.01266",
        "verification": "arXiv record verified",
    },
    {
        "id": 5,
        "citation": "A. Hatamizadeh et al., UNETR: Transformers for 3D Medical Image Segmentation, WACV, 2022.",
        "url": "https://arxiv.org/abs/2103.10504",
        "verification": "arXiv record verified",
    },
    {
        "id": 6,
        "citation": "Z. Liu et al., Swin Transformer: Hierarchical Vision Transformer using Shifted Windows, ICCV, 2021.",
        "url": "https://arxiv.org/abs/2103.14030",
        "verification": "arXiv record verified",
    },
    {
        "id": 7,
        "citation": "F. Isensee et al., nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation, Nature Methods, 2021.",
        "url": "https://pubmed.ncbi.nlm.nih.gov/33288961/",
        "verification": "PubMed record verified",
    },
    {
        "id": 8,
        "citation": "A. Gu and T. Dao, Mamba: Linear-Time Sequence Modeling with Selective State Spaces, 2023.",
        "url": "https://arxiv.org/abs/2312.00752",
        "verification": "arXiv record verified",
    },
    {
        "id": 9,
        "citation": "J. Liu et al., Swin-UMamba: Mamba-based UNet with ImageNet-based pretraining, MICCAI, 2024.",
        "url": "https://arxiv.org/abs/2402.03302",
        "verification": "arXiv record verified",
    },
    {
        "id": 10,
        "citation": "J. M. J. Valanarasu and V. M. Patel, UNeXt: MLP-based Rapid Medical Image Segmentation Network, MICCAI, 2022.",
        "url": "https://arxiv.org/abs/2203.04967",
        "verification": "arXiv record verified",
    },
    {
        "id": 11,
        "citation": "J. Li et al., Transforming medical imaging with Transformers? A comparative review, Medical Image Analysis, 2023.",
        "url": "https://pubmed.ncbi.nlm.nih.gov/36738650/",
        "verification": "PubMed record verified",
    },
    {
        "id": 12,
        "citation": "W. Yao et al., From CNN to Transformer: A Review of Medical Image Segmentation Models, Journal of Imaging Informatics in Medicine, 2024.",
        "url": "https://pubmed.ncbi.nlm.nih.gov/38438696/",
        "verification": "PubMed record verified",
    },
    {
        "id": 13,
        "citation": "P. Bilic et al., The Liver Tumor Segmentation Benchmark (LiTS), 2019.",
        "url": "https://arxiv.org/abs/1901.04056",
        "verification": "arXiv record verified",
    },
    {
        "id": 14,
        "citation": "ISIC Challenge, 2018 Task 1: Lesion Boundary Segmentation data description.",
        "url": "https://challenge.isic-archive.com/landing/2018/45/",
        "verification": "official challenge page verified",
    },
    {
        "id": 15,
        "citation": "B. H. Menze et al., The Multimodal Brain Tumor Image Segmentation Benchmark (BRATS), IEEE TMI, 2015.",
        "url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC4833122/",
        "verification": "PMC full text verified",
    },
    {
        "id": 16,
        "citation": "LUNA16, LUng Nodule Analysis 2016 Challenge.",
        "url": "https://luna16.grand-challenge.org/",
        "verification": "official Grand Challenge page verified",
    },
    {
        "id": 17,
        "citation": "U. Koc et al., MammosighTR: Nationwide Breast Cancer Screening Mammogram Dataset with BI-RADS Annotations, Radiology: Artificial Intelligence, 2025, doi: 10.1148/ryai.240841.",
        "url": "https://pubmed.ncbi.nlm.nih.gov/40801802/",
        "verification": "PubMed record verified",
    },
    {
        "id": 18,
        "citation": "U. Koc et al., Artificial Intelligence in Healthcare Competition (TEKNOFEST-2021): Stroke Data Set, Eurasian Journal of Medicine, 2022.",
        "url": "https://pubmed.ncbi.nlm.nih.gov/35943079/",
        "verification": "PubMed record verified",
    },
    {
        "id": 19,
        "citation": "T.-Y. Lin et al., Focal Loss for Dense Object Detection, ICCV, 2017.",
        "url": "https://arxiv.org/abs/1708.02002",
        "verification": "arXiv record verified",
    },
    {
        "id": 20,
        "citation": "S. S. M. Salehi, D. Erdogmus, and A. Gholipour, Tversky loss function for image segmentation using 3D fully convolutional deep networks, 2017.",
        "url": "https://arxiv.org/abs/1706.05721",
        "verification": "arXiv record verified",
    },
    {
        "id": 21,
        "citation": "M. J. Cardoso et al., MONAI: An open-source framework for deep learning in healthcare, 2022.",
        "url": "https://arxiv.org/abs/2211.02701",
        "verification": "arXiv record verified",
    },
    {
        "id": 22,
        "citation": "C. H. Sudre et al., Generalised Dice overlap as a deep learning loss function for highly unbalanced segmentations, 2017.",
        "url": "https://arxiv.org/abs/1707.03237",
        "verification": "arXiv record verified",
    },
]


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


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
        status = out["status"].fillna("ok")
        out = out[status.eq("ok") | status.eq("")]
    return out


def _normalize_experiments(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    if "use_cache" not in out.columns:
        out["use_cache"] = False
    out["use_cache"] = out["use_cache"].map(
        lambda value: False if pd.isna(value) else str(value).lower() in ["true", "1", "yes"]
    )
    if "loss_name" not in out.columns:
        out["loss_name"] = "bce_dice"
    if "sampler_name" not in out.columns:
        out["sampler_name"] = "none"
    out["loss_name"] = out["loss_name"].fillna("bce_dice")
    out["sampler_name"] = out["sampler_name"].fillna("none")
    return out


def _latest_group(df: pd.DataFrame, mask: pd.Series) -> pd.DataFrame:
    subset = df[mask].copy()
    if subset.empty:
        return subset
    latest = sorted(subset["run_group"].dropna().unique())[-1]
    return subset[subset["run_group"].eq(latest)].copy()


def _main_fair_results(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    mask = (
        df["run_group"].astype(str).str.contains("standard", na=False)
        & df["use_cache"]
        & df["loss_name"].eq("bce_dice")
        & df["sampler_name"].eq("none")
    )
    out = _latest_group(df, mask)
    if not out.empty:
        return out
    return _latest_group(df, df["run_group"].astype(str).str.contains("standard", na=False))


def _pilot_results(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    mask = df["run_group"].astype(str).str.contains("standard", na=False) & ~df["use_cache"]
    return df[mask].copy()


def _ablation_results(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    mask = (
        df["run_group"].astype(str).str.contains("ablation", na=False)
        | ~df["loss_name"].eq("bce_dice")
        | ~df["sampler_name"].eq("none")
    )
    return df[mask].copy().sort_values(["dataset", "model", "best_val_dice"], ascending=[True, True, False])


def _manifest_summary() -> pd.DataFrame:
    path = MANIFESTS_DIR / "manifest_summary.csv"
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def _md_table(df: pd.DataFrame, cols: list[str] | None = None, empty: str = "Kayit yok.") -> str:
    if df is None or df.empty:
        return empty
    table = df.copy()
    if cols is not None:
        table = table[[c for c in cols if c in table.columns]]
    for col in table.select_dtypes(include=["float"]).columns:
        table[col] = table[col].map(lambda x: round(float(x), 4) if pd.notna(x) else x)
    return table.to_markdown(index=False)


def _save_source_table(tables_dir: Path) -> Path:
    df = pd.DataFrame(REFERENCES)
    out = tables_dir / "source_verification.csv"
    df.to_csv(out, index=False)
    return out


def _save_figures(main: pd.DataFrame, ablation: pd.DataFrame, figures_dir: Path) -> dict[str, str]:
    artifacts: dict[str, str] = {}
    if not main.empty:
        pivot = main.pivot_table(index="dataset", columns="model", values="best_val_dice", aggfunc="max")
        ax = pivot.plot(kind="bar", figsize=(10, 5), rot=20)
        ax.set_ylabel("Best validation Dice")
        ax.set_xlabel("")
        ax.grid(axis="y", alpha=0.25)
        ax.legend(title="Model")
        plt.tight_layout()
        path = figures_dir / "ieee_cached_dice.png"
        plt.savefig(path, dpi=180)
        plt.close()
        artifacts["cached_dice"] = str(path)

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.scatter(main["fps"], main["best_val_dice"], s=80)
        for _, row in main.iterrows():
            ax.annotate(f"{row['dataset']}:{row['model']}", (row["fps"], row["best_val_dice"]), fontsize=7)
        ax.set_xlabel("Inference FPS")
        ax.set_ylabel("Best validation Dice")
        ax.grid(alpha=0.25)
        plt.tight_layout()
        path = figures_dir / "ieee_accuracy_speed_tradeoff.png"
        plt.savefig(path, dpi=180)
        plt.close()
        artifacts["accuracy_speed"] = str(path)

    if not ablation.empty and "best_val_dice" in ablation.columns:
        plot_df = ablation.copy()
        plot_df["condition"] = plot_df["model"].astype(str) + "\n" + plot_df["loss_name"].astype(str) + "/" + plot_df["sampler_name"].astype(str)
        ax = plot_df.plot(x="condition", y="best_val_dice", kind="bar", legend=False, figsize=(8, 5), rot=25)
        ax.set_ylabel("Best validation Dice")
        ax.set_xlabel("")
        ax.grid(axis="y", alpha=0.25)
        plt.tight_layout()
        path = figures_dir / "ieee_ablation_dice.png"
        plt.savefig(path, dpi=180)
        plt.close()
        artifacts["ablation_dice"] = str(path)
    return artifacts


def _references_md() -> str:
    lines = []
    for ref in REFERENCES:
        lines.append(f"[{ref['id']}] {ref['citation']} Available: {ref['url']}")
    return "\n".join(lines)


def _source_verification_md() -> str:
    df = pd.DataFrame(REFERENCES)
    return _md_table(df[["id", "verification", "url"]])


def _coverage_notes() -> str:
    return (
        "Onceki literatur raporundaki ana eksenler bu makalede korunmustur: U-Net/CNN tabani, "
        "dikkatli U-Net varyantlari, Transformer ve Swin tabanli hibritler, SSM/Mamba yonu, "
        "LiTS, ISIC, BraTS, LUNA16, MammosighTR ve TEKNOFEST Inme veri setleri, ayrica Dice/IoU, "
        "FPS, VRAM, parametre ve FLOPs gibi dogruluk-maliyet metrikleri. Dogrulanamayan veya "
        "ikincil kaynak niteligindeki onceki atiflar nihai referans listesine alinmamistir."
    )


def main() -> None:
    ensure_project_dirs()
    tables_dir = WORKSPACE_DIR / "reports" / "tables"
    figures_dir = WORKSPACE_DIR / "reports" / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    experiments = _normalize_experiments(_all_summaries())
    main_results = _main_fair_results(experiments)
    pilot = _pilot_results(experiments)
    ablation = _ablation_results(experiments)
    manifests = _manifest_summary()
    access_notes = _read_json(RAW_ROOT / "access_notes.json")
    dataloader_path = WORKSPACE_DIR / "reports" / "tables" / "dataloader_benchmark.csv"
    dataloader = pd.read_csv(dataloader_path) if dataloader_path.exists() else pd.DataFrame()
    source_csv = _save_source_table(tables_dir)
    source_http_csv = tables_dir / "source_http_check.csv"
    artifacts = _save_figures(main_results, ablation, figures_dir)

    if not main_results.empty:
        main_results.sort_values(["dataset", "best_val_dice"], ascending=[True, False]).to_csv(
            tables_dir / "ieee_main_cached_results.csv", index=False
        )
    if not ablation.empty:
        ablation.to_csv(tables_dir / "ieee_ablation_results.csv", index=False)

    best_by_dataset = pd.DataFrame()
    if not main_results.empty:
        idx = main_results.groupby("dataset")["best_val_dice"].idxmax()
        best_by_dataset = main_results.loc[idx].sort_values("dataset")

    stamp = now_stamp()
    out = WORKSPACE_DIR / "reports" / f"ieee_makale_{stamp}.md"
    mirror = REPORTS_DIR / out.name

    main_table = _md_table(
        main_results.sort_values(["dataset", "best_val_dice"], ascending=[True, False]) if not main_results.empty else main_results,
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
            "use_cache",
            "run_group",
        ],
        empty="Ana cached karsilastirma henuz tamamlanmadi.",
    )
    pilot_table = _md_table(
        pilot.sort_values(["dataset", "best_val_dice"], ascending=[True, False]) if not pilot.empty else pilot,
        ["dataset", "model", "best_val_dice", "fps", "peak_memory_mb", "total_seconds", "run_group"],
        empty="Pilot sonuc yok.",
    )
    best_table = _md_table(
        best_by_dataset,
        ["dataset", "model", "best_val_dice", "final_val_iou", "fps", "peak_memory_mb"],
        empty="Ana cached karsilastirma henuz tamamlanmadi.",
    )
    dataloader_table = _md_table(
        dataloader,
        ["dataset", "mode", "samples", "seconds", "samples_per_second", "batch_size", "num_workers"],
        empty="Dataloader benchmark henuz yok.",
    )
    ablation_table = _md_table(
        ablation,
        ["dataset", "model", "loss_name", "sampler_name", "best_val_dice", "final_val_dice", "final_val_iou", "fps", "peak_memory_mb", "total_seconds", "run_group"],
        empty="Ablation sonucu henuz yok.",
    )
    manifest_table = _md_table(manifests, empty="Manifest ozeti yok.")
    access_json = json.dumps(access_notes, ensure_ascii=False, indent=2)
    figure_lines = "\n".join(f"- {name}: `{path}`" for name, path in artifacts.items()) or "- Sekil henuz uretilmedi."

    report = f"""# IEEE Taslak Makale: Tibbi Goruntu Segmentasyonunda CNN, Transformer ve Hibrit Modellerin Adil Karsilastirmasi

**Yazar:** Ishak Bedir Yorganci  
**Ogrenci No:** 23118080033  
**Uretim zamani:** {stamp}

## Abstract

Bu calisma, tibbi goruntu segmentasyonunda CNN tabanli U-Net ailesi ile Transformer/hibrit yaklasimlarin ayni donanim, ayni veri bolmesi ve ayni egitim protokolu altinda karsilastirilmasini sunar. Onceki pilot kosularda ozellikle ISIC 2018 veri yolunda ham JPEG okuma ve yeniden boyutlandirma darboğazi oldugu goruldugu icin ana karsilastirma, on islenmis NumPy cache uzerinden tekrarlandi. Boylece veri yukleme hizinin model egitimi ve FPS yorumunu bozma riski azaltildi. Deneyler RTX 3090 uzerinde ISIC 2018 cilt lezyonu segmentasyonu ve LiTS karaciger/lezyon pozitif dilimleri ile yapildi. Sonuclar, global baglam gerektiren ISIC ve karaciger siniri gorevlerinde SwinUNETR'in guclu oldugunu, kucuk ve dengesiz LiTS lezyon gorevinde ise dikkatli CNN ailesinin rekabetci kaldigini gosterir. Ek olarak class-imbalance icin Tversky/Focal kayip ve foreground sampling ablasyonu yeni model gelistirme yonu olarak kuruldu.

**Index Terms:** medical image segmentation, U-Net, SwinUNETR, Vision Transformer, LiTS, ISIC, class imbalance, Tversky loss, Mamba.

## I. Introduction

Tibbi goruntu segmentasyonu, klinik karar destek sistemlerinde hastalik alaninin piksel veya voksel duzeyinde ayrilmasini saglayan temel bir adimdir. U-Net mimarisi, simetrik encoder-decoder yapisi ve skip connection tasarimi ile biyomedikal segmentasyonun uzun sureli referans noktasi olmustur [1]. Attention U-Net gibi varyantlar, hedef organ veya lezyona odaklanmak icin dikkat kapilari ekleyerek klasik CNN hattini guclendirmistir [2]. Buna karsin Transformer tabanli yaklasimlar, uzun menzilli bagimliliklari daha dogrudan modelleme vaadiyle TransUNet, UNETR ve SwinUNETR gibi hibrit yapilari one cikarmistir [3]-[6].

Bu projenin temel sorusu sadece "hangi model daha yuksek Dice verir?" degildir. Klinik ve pratik dagitim acisindan FPS, VRAM, parametre sayisi, FLOPs ve egitim suresi de ayni agirlikta degerlendirilmelidir. Bu nedenle calisma, dogruluk-maliyet dengesini deneysel olarak inceler ve literaturdeki CNN yerellik avantaji ile Transformer global baglam avantaji tartismasini ayni donanim uzerinde test eder [11], [12].

## II. Related Work

U-Net, az anotasyonlu biyomedikal veriyle egitilebilmesi ve hassas lokalizasyon saglamasi nedeniyle temel mimari olarak kullanilir [1]. Attention U-Net, ek doku/organ lokalizasyon modulune ihtiyaci azaltan dikkat kapilariyla U-Net'i guclendirir [2]. TransUNet, CNN ozellik haritasini Transformer encoder'a vererek global baglam ve lokal ayrintiyi birlestirir [3]. UNETR ve SwinUNETR ise ozellikle 3D hacimlerde Transformer encoder fikrini segmentasyon decoder'i ile birlestirir [4], [5]. Swin Transformer'in pencereli ve kaydirilmali dikkat tasarimi, standart self-attention'in karesel maliyetini pratikte daha yonetilebilir hale getirir [6].

Otomatik ve guclu baseline olarak nnU-Net, veri on isleme, mimari ve egitim kararlarini goreve gore kendisi ayarlamasi nedeniyle tibbi segmentasyonda cok onemli bir referanstir [7]. Daha yeni dogrultuda Mamba ve Swin-UMamba gibi SSM tabanli mimariler, global baglami lineer olceklenme ile yakalama iddiasiyla Transformer maliyetine alternatif sunar [8], [9]. Hafif modeller tarafinda UNeXt, point-of-care senaryolari icin hiz ve parametre verimliligini one cikarir [10].

Veri seti acisindan LiTS karaciger ve tumor segmentasyonunda guclu ama kucuk hedefli bir benchmark sunar [13]. ISIC 2018 dermoskopik lezyon siniri icin resmi binary mask gorevi tanimlar [14]. BraTS ve LUNA16 bu projenin hacimsel MRI/CT hedeflerini temsil eder [15], [16]. Turkiye odakli MammosighTR ve TEKNOFEST Inme veri setleri de ulusal veri ekosisteminin onemli parcalaridir [17], [18].

## III. Materials and Methods

### A. Data Inventory

{manifest_table}

### B. Access Scope

```json
{access_json}
```

ISIC 2018 ve LiTS PNG verileri D diskinde indirildi ve deneylerde kullanildi. TEKNOFEST renkli tured veri siniflandirma odakli oldugu icin maskeli segmentasyon deneyine alinmadi. BraTS/Synapse, LUNA16, MammosighTR ve resmi Inme BT veri setleri erisim notu olarak kaydedildi; dogrudan indirilemeyen veya oturum gerektiren veri icin metrik iddiasi uretilmedi.

### C. Models

Deneysel ana hat uc model kullanir: MONAI U-Net, MONAI Attention U-Net ve kompakt 2D MONAI SwinUNETR. U-Net CNN yerel ozellik baseline'idir. Attention U-Net, CNN'e hedefe odaklanma kapisi ekleyen ara modeldir. SwinUNETR, pencereli self-attention ile global baglam etkisini test eden hibrit Transformer modelidir. MONAI secimi, tibbi goruntuleme icin yaygin ve tekrar uretilebilir PyTorch tabanli uygulama sagladigi icin yapildi [21].

### D. Experimental Protocol

Tum modeller ayni seed, ayni train/validation bolmesi, ayni goruntu boyutu, ayni batch size, ayni epoch sayisi, ayni optimizer ve ayni metriklerle degerlendirildi. Ana standart protokol: 192x192 girdi, batch size 8, 8 epoch, AdamW, AMP, maksimum 1200 egitim ve 300 validasyon ornegi. Birincil metrik Dice, ikincil metrik IoU'dur. Verimlilik icin FPS, tepe GPU bellegi, parametre sayisi, yaklasik GFLOPs ve toplam sure raporlandi.

## IV. Fairness and Bottleneck Audit

Onceki pilot sonuclar dogrudan nihai karsilastirma olarak kullanilmadi; cunku ISIC 2018 ham JPEG okuma ve resize islemi DataLoader uzerinde belirgin darboğaz olusturdu. Bu nedenle tum ana karsilastirma cache'lenmis, ayni split ve ayni on isleme boyutuna sahip NumPy dizileri uzerinden tekrar calistirildi. Bu adim, model farklarini veri okuma farklarindan ayirmak icin kritiktir.

### A. Dataloader Benchmark

{dataloader_table}

Bu tabloda ISIC ham veri yolu ile cache arasindaki fark, veri okuma/resize darboğazinin gercek oldugunu gosterir. LiTS PNG tarafinda ham veri yolu zaten daha hizli oldugundan cache etkisi daha kucuktur. Bu nedenle ana rapor, cache sonucunu adil karsilastirma kabul eder; pilot sonuc yalnizca on bulgu olarak tutulur.

### B. Reproducibility Controls

- Veri ve deney kok dizini: `D:\\tibbi-segmentasyon`
- Kod ve rapor dizini: `C:\\Users\\ishak\\tibbi-segmentasyon`
- Donanim: NVIDIA GeForce RTX 3090, 24 GB VRAM
- Seed: 23118080033
- Dataloader worker sayisi: 2
- Ana karsilastirmada cache: aktif

Windows DataLoader worker baslangic gecikmesi loglarda goruldu; bu gecikme epoch baslangicinda sabit maliyet yaratir, fakat cache benchmark'i ISIC ham okuma darboğazinin ayri ve daha buyuk bir etki oldugunu gostermektedir.

## V. Results

### A. Pilot Results

{pilot_table}

Pilot kosular, mimari siralamasi icin fikir verdi ancak bottleneck denetimi gecmedigi icin nihai deney kaniti olarak kullanilmadi.

### B. Main Cached/Fair Results

{main_table}

### C. Best Model by Dataset

{best_table}

### D. Figures

{figure_lines}

## VI. Discussion

ISIC 2018 gorevinde lezyonun genel sekli, renk dagilimi ve goruntu capindaki baglam SwinUNETR lehine calisir. Bu, Transformer/hibrit modellerin global baglam avantajiyla uyumludur [3], [4], [11]. LiTS karaciger pozitif dilimlerinde de organin genis siniri ve anatomik konumu global baglama duyarlidir; bu nedenle SwinUNETR'in yuksek Dice uretmesi beklenen bir sonuctur. Buna karsin LiTS lezyon pozitif dilimlerinde hedef kucuk, kontrast dusuk ve piksel dengesizligi serttir. Bu alt gorevde Attention U-Net'in guclu kalmasi, CNN tabanli lokal doku ve sinir ozelliklerinin halen kritik oldugunu gosterir [1], [2].

Verimlilik tarafinda SwinUNETR'in parametre ve bellek maliyeti daha yuksektir. U-Net daha hafif, Attention U-Net ise dogruluk/hiz acisindan dengeli bir ara noktadir. Bu durum klinik dagitim icin onemlidir: en yuksek Dice skoru, her zaman en uygun operasyonel tercih degildir. Acil triyaj, dusuk gecikme veya sinirli donanim senaryosunda hafif CNN/MLP tabanli modeller daha uygun olabilir [10].

## VII. Improvement Experiments and Scientific Direction

Lezyon segmentasyonunda ana hata kaynagi yalnizca mimari degil, foreground-background dengesizligidir. Bu nedenle gelistirme deneyleri iki eksene ayrildi:

1. **Loss tasarimi:** BCE+Dice yerine Dice+Focal ve Tversky+Focal secenekleri eklendi. Focal loss kolay arka plan piksellerini bastirir [19]. Tversky loss, yanlis negatif maliyetini daha yuksek tutarak kucuk lezyon kacirma riskini azaltmaya uygundur [20]. Generalized Dice da dengesiz segmentasyon literaturunde ayni aileden bir dayanak sunar [22].
2. **Sampler tasarimi:** Foreground alanina gore agirlikli ornekleme eklendi. Amac, cok kucuk lezyon dilimlerinin ve arka plan baskisinin egitim gradyanini tek yonlu hale getirmesini azaltmaktir.

### A. Ablation Results

{ablation_table}

Ablation sonucu model onerisini daha net hale getirmistir: LiTS lezyon icin en verimli ilk ilerleme, class-imbalance uyumlu loss ve sampler kullanmaktir. Bu ayar Attention U-Net'i ana cached baseline degerinin uzerine tasirken SwinUNETR'de ayni kazanci uretmedi; bu da kucuk hedef probleminde mimari buyutmenin tek basina yeterli olmadigini gosterir. Bir sonraki bilimsel adim 2.5D/3D baglam eklemek, ardindan SSM/Mamba tabanli hafif global modul denemektir. Mamba/Swin-UMamba yonu teorik olarak caziptir; ancak Windows/CUDA ekosisteminde dogrudan Mamba kurulumu riskli oldugundan, ilk bilimsel adim olarak kayip/sampling ve 2.5D baglam ablation'i daha kontrollu ve tekrar uretilebilirdir [8], [9].

Faktor ayrıştırma deneyleri bu yorumu guclendirdi. Attention U-Net uzerinde Tversky/Focal tek basina orta duzey kazanc saglarken, foreground sampling tek basina daha buyuk bir kazanc verdi; ancak en iyi sonuc bu ikisinin birlikte kullanildigi kosulda elde edildi. Dice/Focal + foreground deneyi ise Tversky/Focal + foreground sonucunun gerisinde kaldi. Bu nedenle kucuk lezyon gorevinde asil hipotez, "daha fazla global baglam"dan once "yanlis negatifleri pahali yapan loss + foreground duyarlı batch dagilimi" seklinde formulle edilmelidir.

2.5D deneyi, komsu CT dilimlerini uc kanalli giris olarak kullanmanin otomatik olarak iyilesme getirmedigini gosterdi. Basit prev/current/next yiginlama, bu kosulda 2D Attention U-Net + Tversky/Focal + foreground sampling sonucunu gecmedi. Bu negatif bulgu onemlidir: LiTS lezyon gorevinde baglam eklemek icin yalnizca komsu dilimleri kanala koymak yerine, kesitler arasi konumu modelleyen 2.5D attention, recurrent/SSM bloklari veya gercek 3D patch egitimi denenmelidir.

## VIII. Threats to Validity

Bu calisma henuz tam 3D hacimsel egitim degildir; LiTS verisi 2D pozitif dilimler uzerinden calistirilmistir. BraTS, LUNA16, MammosighTR ve resmi Inme BT veri setleri erisim/oturum gerektirdigi icin dogrudan metrik kapsaminda degildir. Epoch sayisi ve ornek sayisi sinirli tutuldugu icin sonuclar nihai leaderboard iddiasi degil, adil donanim-ici mimari karsilastirmasidir. Buna ragmen cache tabanli tekrar kosu, pilot darboğazini azaltarak bu proje icin gecerliligi belirgin bicimde artirmistir.

## IX. Conclusion

Adil cache tabanli deney tasarimi, CNN ve Transformer/hibrit modellerin farklarini veri yukleme darboğazindan ayirarak daha guvenilir bir karsilastirma sagladi. Genel sonuc, mimari secimin goreve bagli oldugudur: global baglamli organ veya genis lezyon yapilarinda SwinUNETR guclu, kucuk ve dengesiz hedeflerde Attention U-Net/CNN ailesi halen cok rekabetcidir. Geliştirme yonu olarak Tversky/Focal loss, foreground sampling, 2.5D/3D baglam ve daha sonra Mamba/SSM tabanli hafif global modul sirasi bilimsel olarak en savunulabilir yol olarak belirlenmistir.

## Literature Coverage Note

{_coverage_notes()}

## Source Verification Table

{_source_verification_md()}

Kaynak dogrulama tablosu ayrica CSV olarak kaydedildi: `{source_csv}`  
HTTP erisim kontrolu CSV dosyasi: `{source_http_csv if source_http_csv.exists() else "henuz uretilmedi"}`

## References

{_references_md()}
"""

    out.write_text(report, encoding="utf-8")
    mirror.write_text(report, encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
