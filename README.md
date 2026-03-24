# TVGM: Source Code for Time-Varying Graph Model

![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)
![Python](https://img.shields.io/badge/Python->=3.8-blue.svg?style=for-the-badge)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)

This repository contains the official PyTorch implementation for a structurally decoupled time-varying graph model applied to quantitative stock selection.

*(Note: To strictly comply with double-blind peer review policies, the exact paper title, full abstract, and author information have been temporarily removed. The complete details will be updated upon publication.)*

## 📊 Main Results (Top-K = 30)
The following table presents the out-of-sample backtesting performance of our proposed TVGM compared to its ablated single-branch variants. 

| Metric | TVGM (Hybrid) | ST-only | HAR-only |
| :--- | :--- | :--- | :--- |
| **Sharpe** | **2.2580 ± 1.4287** | 2.1275 ± 1.2327 | 1.2352 ± 0.6879 |
| **CumRet** | **8.20% ± 3.43%** | 7.87% ± 3.37% | 6.16% ± 3.67% |
| **AnnRet** | **15.76% ± 5.83%** | 15.17% ± 5.75% | 12.29% ± 6.63% |
| **AnnVol** | 9.64% ± 4.30% | **9.58% ± 4.32%** | 11.92% ± 4.19% |
| **MaxDD** | -4.59% ± 3.62% | **-4.53% ± 3.65%** | -7.02% ± 4.51% |
| **P@K** | **56.76% ± 2.76%** | **56.76% ± 2.54%** | 53.97% ± 2.18% |
| **LiftPct**| **3.70% ± 5.04%** | **3.70% ± 4.64%** | -1.40% ± 3.98% |

*(Note: The above results are evaluated on a dynamically rebalanced S&P 500 universe using a 4-fold rolling cross-validation protocol.)*

## 📂 Project Structure
```text
TVGM/
├── data/
│   ├── processed/          # Generated tensor/numpy data (Features & Masks)
│   └── raw/                # Raw OHLCV market data (e.g., all_stocks_5yr.csv)
├── models/                 # Core model architectures
│   ├── __init__.py
│   ├── layers.py           # Dense Chebyshev Convolution Layer
│   └── tvgm.py             # Complete TVGM Architecture (ST & HAR branches)
├── utils/                  # Helper functions
│   ├── data_loader.py      # Rolling window dataset & Dynamic Graph builder
│   ├── evaluator.py        # Top-K Backtesting Engine
│   ├── losses.py           # Risk-Averse Combined Loss
│   ├── metrics.py          # Financial Evaluation Metrics
│   └── static_gen.py       # Static Spectral Basis Generator
├── feature.py              # Feature Engineering & Preprocessing Pipeline
├── train.py                # Main Training & Evaluation Loop
├── README.md               # Project documentation
└── requirements.txt        # Dependencies
```
# ⚙️ Installation

1.Clone this repository:
```text
git clone [https://github.com/AnonymousAuthor/TVGM.git](https://github.com/AnonymousAuthor/TVGM.git)
cd TVGM
```
2.Install the required dependencies:
```text
pip install -r requirements.txt
```

# 🚀 Quick Start (How to Run)

The pipeline consists of two mandatory steps: Data Preprocessing and Model Training.

Step 1: Feature Engineering

Before training, you must process the raw CSV data to calculate technical indicators, align dates, and generate dynamic masks.
```text
python feature.py --task_type 5D --top_n 505
```
This will create the necessary .npy files inside the data/processed/ directory.

*(Note: Running ```python feature.py``` without arguments will automatically use the default settings: 5-day horizon and 505 stocks.)*

Step 2: Model Training & Evaluation

Once the data is processed, you can train the TVGM model and evaluate it using the rolling-window backtest protocol.
```text
python train.py --seq_len 20 --sparsity 0.08 --penalty 4.0 --epochs 15
```
*(Note: Running ```python train.py``` without arguments will automatically execute the training loop using the optimal default hyperparameters: seq_len=20, sparsity=0.08, and penalty=4.0.)*

# 🔬 Ablation Studies

You can easily reproduce the ablation studies from the paper by specifying the --ablation_mode argument:
1. Remove risk-averse penalty (Standard MSE)
```text
    python train.py --ablation_mode wo-risk
```

2. Remove pairwise ranking loss
```text
python train.py --ablation_mode wo-rank
```
