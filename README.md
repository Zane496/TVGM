# TVGM: Source Code for Time-Varying Graph Model

![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)
![Python](https://img.shields.io/badge/Python->=3.8-blue.svg?style=for-the-badge)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)

This repository contains the official PyTorch implementation for a structurally decoupled time-varying graph model applied to quantitative stock selection.

*(Note: To strictly comply with double-blind peer review policies, the exact paper title, full abstract, and author information have been temporarily removed. The complete details will be updated upon publication.)*

## 📊 Main Results (Top-K = 30)

The following table reports the out-of-sample backtesting performance of the full TVGM model (mean ± standard deviation across rolling evaluation folds).

| Metric | TVGM (Full Model) |
| :--- | :--- |
| **Sharpe** | **2.2580 ± 1.4287** |
| **CumRet** | **8.20% ± 3.43%** |
| **AnnRet** | **15.76% ± 5.83%** |
| **AnnVol** | **9.64% ± 4.30%** |
| **MaxDD** | **-4.59% ± 3.62%** |
| **P@K** | **56.76% ± 2.76%** |
| **LiftPct** | **3.70% ± 5.04%** |

*Note: The reported results use a dynamically rebalanced S&P 500 stock universe and a four-fold rolling evaluation protocol.*

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

1. Clone this repository (replace `<REPOSITORY_URL>` with the actual GitHub clone URL):
```text
git clone <REPOSITORY_URL>
cd TVGM
```
2. Install the required dependencies:
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
*(Note: Running ```python train.py``` without arguments will automatically execute the training loop using the default hyperparameters: seq_len=20, sparsity=0.08, and penalty=4.0.)*
