<div align="center">

# 🩸 MTFTT — Multi-Task Feature-Token Transformer

### Attention-Guided Anemia Subtyping from Routine CBC Data

*A calibrated, interpretable Transformer for five-class anemia classification, binary screening, and hemoglobin prediction — all in a single forward pass.*

</div>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Architecture](#-architecture)
- [Key Results](#-key-results)
- [Repository Structure](#-repository-structure)
- [Installation](#-installation)
- [Dataset](#-dataset)
- [Usage](#-usage)
- [Training Pipeline](#-training-pipeline)
- [Ablation Study](#-ablation-study)
- [Explainability (XAI)](#-explainability-xai)


---

## 🔬 Overview

**MTFTT (Multi-Task Feature-Token Transformer)** is a deep learning framework designed for automated anemia subtyping from routine Complete Blood Count (CBC) data. It classifies patients into five diagnostic categories:

| Class | Label | Dataset % |
|-------|-------|-----------|
| 0 | No Anemia | 63.7% |
| 1 | HGB Anemia | 6.7% |
| 2 | Iron Deficiency Anemia | 27.3% |
| 3 | Folate Deficiency Anemia | 1.0% |
| 4 | B12 Deficiency Anemia | 1.3% |

### Why MTFTT?

- **Unified framework** — subtype classification, binary screening, HGB regression, and uncertainty estimation in one model
- **Handles extreme imbalance** — 63.7:1 majority-to-minority ratio via SMOTE + curriculum learning + Focal–Dice loss
- **Clinically calibrated** — temperature scaling reduces ECE to 0.42%, making probabilities trustworthy for clinical decisions
- **Interpretable** — four XAI methods converge on features aligned with WHO diagnostic criteria
- **Deployable** — 4.88M parameters, 18.6 MB memory, 0.062 ms per-sample latency

---

## 🏗 Architecture

```
Input (32 features)
│
├── 24 Raw CBC features (WBC, HGB, RBC, MCV, Ferritin, Folate, B12, ...)
└── 8 Engineered indices (Mentzer Index, NLR, PLR, MCHC Deficit, ...)
│
▼
┌─────────────────────────────────┐
│   Feature Tokenization          │
│   Scalar Embedding + ID Embed   │
│   → [B, 32, 256]               │
└─────────────┬───────────────────┘
              ▼
┌─────────────────────────────────┐
│   Transformer Encoder (×6)      │
│   8 heads, d_model=256          │
│   d_ff=1024, GELU, LayerNorm   │
└─────────────┬───────────────────┘
              ▼
┌─────────────────────────────────┐
│   Attention Pooling             │
│   [B, 32, 256] → [B, 256]      │
│   + Clinical Prior Reg.         │
└─────────────┬───────────────────┘
              ▼
┌──────────┬──────────┬──────────┬──────────┐
│ Subtype  │ Binary   │ HGB      │ Variance │
│ (5-cls)  │ (2-cls)  │ (regr.)  │ (σ²)     │
│ Focal-   │ Label-   │ Gaussian │ Softplus │
│ Dice     │ Smooth CE│ NLL      │          │
└──────────┴──────────┴──────────┴──────────┘
              ▼
   Uncertainty-Based Task Weighting
```

### Key Components

| Component | Description |
|-----------|-------------|
| **Feature Tokenization** | Each scalar feature → learnable D-dim embedding via value projection + identity lookup |
| **Transformer Encoder** | 6-layer, 8-head self-attention with GELU FFN, capturing feature–feature interactions |
| **Attention Pooling** | Learned weighted aggregation of token embeddings with clinical prior regularization |
| **Multi-Task Heads** | Four shared-representation heads for joint optimization |
| **Task Weighting** | Automatic uncertainty-based weighting (Kendall et al., 2018) |

---

## 📊 Key Results

### Multi-Class Performance (test set, n=2,296)

| Metric | Value |
|--------|-------|
| **Accuracy** | 98.17% |
| **Balanced Accuracy** | 92.60% |
| **Macro F1** | 91.20% |
| **Weighted F1** | 98.20% |
| **MCC** | 96.46% |
| **Cohen's κ** | 96.45% |
| **ECE (calibrated)** | 0.42% |
| **Brier Score** | 0.027 |

### Per-Class Breakdown

| Class | Precision | Sensitivity | F1 | AUC |
|-------|-----------|-------------|-----|-----|
| No Anemia | 99.66% | 99.73% | 99.69% | 100.0% |
| HGB Anemia | 87.73% | 93.46% | 90.51% | 99.8% |
| Iron Deficiency | 98.86% | 96.66% | 97.75% | 99.9% |
| Folate Deficiency | 80.00% | 86.96% | 83.33% | 99.9% |
| B12 Deficiency | 83.33% | 86.21% | 84.75% | 99.9% |

### Binary Screening (Anemia vs No Anemia)

| Metric | Value |
|--------|-------|
| **Sensitivity** | 99.40% |
| **Specificity** | 99.73% |
| **F1 Score** | 99.46% |
| **ECE (calibrated)** | 0.64% |

### Efficiency

| Property | Value |
|----------|-------|
| Parameters | 4.88M |
| Memory (FP32) | 18.6 MB |
| Inference latency | 0.062 ms/sample |
| Throughput | ~16,047 samples/s |
| Training time | ~15.3 min (single GPU) |

---

## 📁 Repository Structure

```
MTFTT/
│
├── README.md                        # This file
├── LICENSE                          # MIT License
├── requirements.txt                 # Python dependencies
│
├── MTFTT_Proposed_Model.py          # Main model: training + evaluation + XAI
├── MTFTT_Ablation.py                # Ablation study: 11 variant experiments
│
├── SKILICARSLAN_Anemia_DataSet.csv  # Dataset (download from Kaggle)
│
└── Final_MTFTT_OUTPUTS/             # Auto-generated output directory
    └── <run_timestamp>/
        ├── config.json              # Run configuration snapshot
        ├── plots/                   # All visualizations + PDF report
        │   ├── *_PLOTS.pdf          # Combined PDF of all plots
        │   ├── *_cm.png             # Confusion matrices
        │   ├── *_roc_ovr.png        # ROC curves
        │   ├── *_pr_ovr.png         # Precision-Recall curves
        │   ├── *_reliability_*.png  # Calibration reliability diagrams
        │   ├── *_decision_curve.png # DCA plot
        │   ├── *_xai_*.png          # XAI visualizations
        │   └── *_L*_attn_*.png      # Per-layer attention maps
        ├── tables/                  # CSV results and metrics
        │   ├── *_results_table.csv  # Summary metrics
        │   ├── *_confusion_matrix.csv
        │   ├── *_per_class_report.csv
        │   ├── *_per_class_metrics.csv
        │   ├── *_decision_curve.csv
        │   ├── *_attention_global.csv
        │   ├── *_occlusion.csv
        │   ├── *_integrated_gradients_samples.csv
        │   ├── *_rollout_receiver_score.csv
        │   └── ABLATIONS_SUMMARY.csv    # Ablation comparison table
        ├── metrics/                 # JSON summaries and timing
        │   ├── *_summary.json
        │   ├── *_complexity.json
        │   ├── *_timing.json
        │   └── *_history.csv        # Epoch-by-epoch training log
        ├── models/                  # Saved checkpoints
        │   └── *_best.pt
        └── binary/                  # Binary classification outputs
            ├── *_binary_metrics.csv
            ├── *_binary_confusion_matrix.csv
            └── *_binary_probs_calibrated.csv
```

---

## ⚙ Installation

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended, CPU supported)

### Setup

```bash
# Clone the repository
git clone https://github.com/<your-username>/MTFTT.git
cd MTFTT

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate        # Linux/Mac
# venv\Scripts\activate         # Windows

# Install dependencies
pip install -r requirements.txt
```

### requirements.txt

```
torch>=2.0.0
numpy>=1.24.0
pandas>=2.0.0
scikit-learn>=1.3.0
imbalanced-learn>=0.11.0
matplotlib>=3.7.0
tqdm>=4.65.0
```

---

## 📦 Dataset

The model uses the **Kılıçarslan et al. Anemia Disease Dataset** containing 15,300 CBC records from Tokat Gaziosmanpaşa University, Turkey (2013–2018).

**Download:** [Kaggle – Anemia Disease Dataset](https://www.kaggle.com/datasets/serhathoca/anemia-disease)

Place the CSV file in the project root:

```
MTFTT/
└── SKILICARSLAN_Anemia_DataSet.csv
```

**Dataset summary:** 15,300 patients (10,379 female, 4,921 male) | 24 raw CBC features | 5 diagnostic classes

---

## 🚀 Usage

### Train the Main Model

```bash
python MTFTT_Proposed_Model.py
```

This runs the full pipeline: data loading → SMOTE → feature engineering → 3-stage curriculum training → test evaluation → calibration → XAI analysis → plot generation.

### Run Ablation Study

```bash
python MTFTT_Ablation.py
```

Runs the main model + 11 ablation variants and saves a comparison table at `ABLATIONS_SUMMARY.csv`.

### Configuration

All hyperparameters are centralized in the `CFG` dataclass. Key settings:

```python
@dataclass
class CFG:
    # Architecture
    D_MODEL: int = 256          # Transformer hidden dimension
    N_HEADS: int = 8            # Attention heads
    N_LAYERS: int = 6           # Encoder depth
    D_FF: int = 1024            # FFN expansion
    POOLING: str = "attn"       # "attn" or "mean"

    # Training
    LR: float = 2e-4            # Initial learning rate
    BATCH: int = 256            # Training batch size
    E_STAGE1: int = 30          # Stage 1 epochs (classes 0,1,2)
    E_STAGE2: int = 40          # Stage 2 epochs (classes 0,1,2,4)
    E_STAGE3: int = 80          # Stage 3 epochs (all classes)
    PATIENCE: int = 12          # Early stopping patience

    # Loss
    FOCAL_ALPHA: float = 0.20
    FOCAL_GAMMA: float = 1.80
    FOCAL_DICE_MIX: float = 0.5
    LABEL_SMOOTH: float = 0.02

    # Data
    USE_SMOTE: bool = True
    USE_CURRICULUM: bool = True
    USE_AUG: bool = True
    SEED: int = 42
```

---

## 🔄 Training Pipeline

```
1. Data Loading & Validation
   └── Label-flag consistency checks

2. Stratified Split (70/7.5/7.5/15)
   └── Train / Val / Calibration / Test

3. SMOTE (train only, raw feature space)
   └── Adaptive k-neighbors

4. Feature Engineering (8 clinical indices)
   └── Mentzer Index, NLR, PLR, MCHC Deficit, ...

5. StandardScaler (fit on train only)

6. Three-Stage Curriculum Training
   ├── Stage 1: Classes {0, 1, 2}         — 30 epochs, LR ×1.00
   ├── Stage 2: Classes {0, 1, 2, 4}      — 40 epochs, LR ×0.75
   └── Stage 3: All classes {0,1,2,3,4}   — 80 epochs, LR ×0.50
       └── Early stopping on val macro-F1

7. Temperature Scaling (on calibration set)

8. Test Evaluation + XAI Analysis

9. Output Generation (plots, tables, metrics, checkpoints)
```

---

## 🧪 Ablation Study

Systematic evaluation of 11 configurations to isolate each component's contribution:

| Variant | Acc% | Macro-F1% | MCC% | ECE% |
|---------|------|-----------|------|------|
| **Full MTFTT** | **98.17** | **91.20** | **96.46** | **0.42** |
| w/o Curriculum | 96.60 | 89.57 | 93.57 | 1.04 |
| w/o Augmentation | 97.21 | 89.27 | 94.63 | 0.76 |
| w/o Focal-Dice | 98.00 | 91.78 | 96.13 | 0.71 |
| w/o Multi-Task | 98.00 | 91.71 | 96.12 | 0.40 |
| w/o Attn Prior | 97.52 | 89.44 | 95.26 | 0.47 |
| Mean Pooling | 97.60 | 89.19 | 95.37 | 0.63 |
| L=4, H=8, d=256 | 97.69 | 92.27 | 95.56 | 0.44 |
| L=6, H=4, d=256 | 98.04 | 90.74 | 96.21 | 1.28 |
| d=128 | 98.13 | 89.93 | 96.37 | 0.67 |
| d=384 | 97.30 | 88.54 | 94.77 | 0.69 |

**Key findings:** Curriculum learning, attention pooling, and medical augmentation are the most impactful components. The default configuration (L=6, H=8, d=256) offers the best performance–calibration trade-off.

---

## 🔍 Explainability (XAI)

Four complementary attribution methods provide multi-faceted interpretability:

| Method | What it measures | Top-3 Features |
|--------|------------------|----------------|
| **Attention Pooling** | Internal aggregation strategy | PLT, RBC, MPV |
| **Occlusion Sensitivity** | Output impact per feature | HGB, GENDER, TSD |
| **Integrated Gradients** | Gradient-based attribution | HGB, TSD, GENDER |
| **Attention Rollout** | Multi-hop token information flow | HGB, GENDER, TSD |

**Consensus features** (3/4 methods): **HGB**, **GENDER**, **TSD** — directly aligned with WHO anemia diagnostic criteria.

### Attention Specialization

The model shows progressive attention behavior across encoder depth:

- **Layers 1–2:** Sharp feature selection (iron markers, HGB)
- **Layer 2, Head 1:** Near-deterministic focus (entropy = 0.074) on engineered iron-status features
- **Layer 3, Head 5:** Specializes on B12/Folate markers
- **Layers 4–6:** Broad integration across all features

---



<div align="center">

**Made with ❤️ for clinical AI research**

If you find this work useful, please ⭐ the repository!

</div>
