import os
import gc
import random
import argparse
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.tvgm import TVGM
from utils.static_gen import get_dynamic_static_adj
from utils.data_loader import add_market_context_and_norm, get_causal_dynamic_adjs, RollingStockDataset
from utils.losses import risk_averse_loss
from utils.evaluator import evaluate_backtest


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def run_experiment(args):
    set_seed(args.seed)
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using Device: {DEVICE}")

    # Load preprocessed data
    try:
        raw_data = np.load(os.path.join(args.data_dir, 'feature_data.npy'))
        raw_masks = np.load(os.path.join(args.data_dir, 'valid_mask.npy'))
    except FileNotFoundError:
        print("[ERROR] Processed data not found. Please run feature.py first.")
        return

    TOTAL_T = raw_data.shape[1]
    num_folds = (TOTAL_T - args.train_window) // args.step_size
    fold_results = []

    k_list = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]

    for fold in range(num_folds):
        t_s = fold * args.step_size
        t_e_train = t_s + args.train_window
        t_e_test = min(t_e_train + args.test_window, TOTAL_T)

        print(f"\n" + "=" * 20 + f" Fold {fold + 1}/{num_folds} " + "=" * 20)

        # Dynamic Universe Selection (Top 300 by volume ratio)
        avg_vols = np.nanmean(raw_data[:, t_s:t_e_train, 16], axis=1)
        top_indices = np.argsort(avg_vols)[-args.top_k_stocks:]

        f_data = np.transpose(raw_data[top_indices, t_s:t_e_test, :], (1, 0, 2))
        f_mask = np.transpose(raw_masks[top_indices, t_s:t_e_test], (1, 0))

        X_norm = add_market_context_and_norm(f_data[:, :, :16])
        Y_target, Y_raw_ret = f_data[:, :, 17] * 100.0, f_data[:, :, 18]

        A_static = get_dynamic_static_adj(f_data[:args.train_window, :, :16], target_sparsity=args.sparsity, device=DEVICE)
        A_dynamic = get_causal_dynamic_adjs(f_data[:, :, 0], window_size=args.seq_len, sparsity=args.sparsity)

        train_end_idx = args.train_window - args.horizon + 1
        test_start_idx = args.train_window - args.seq_len + 1
        test_end_idx = args.train_window + args.test_window

        train_loader = DataLoader(RollingStockDataset(
            X_norm[:train_end_idx], Y_target[:train_end_idx], Y_raw_ret[:train_end_idx],
            f_mask[:train_end_idx], A_dynamic[:train_end_idx], args.seq_len
        ), batch_size=args.batch_size, shuffle=True)

        test_loader = DataLoader(RollingStockDataset(
            X_norm[test_start_idx:test_end_idx], Y_target[test_start_idx:test_end_idx],
            Y_raw_ret[test_start_idx:test_end_idx], f_mask[test_start_idx:test_end_idx],
            A_dynamic[test_start_idx:test_end_idx], args.seq_len
        ), batch_size=args.batch_size, shuffle=False)

        model = TVGM(in_features=X_norm.shape[-1], hidden_features=args.hidden_dim, out_features=1, A_static=A_static, K=3).to(DEVICE)
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)

        # Training Loop
        for epoch in range(args.epochs):
            model.train()
            pbar = tqdm(train_loader, desc=f"  Epoch {epoch + 1:02d}/{args.epochs}", leave=False)
            for x, y, _, adj, mask in pbar:
                x, y, adj, mask = x.to(DEVICE), y.to(DEVICE), adj.to(DEVICE), mask.to(DEVICE)
                optimizer.zero_grad()

                loss, _, _ = risk_averse_loss(
                    model(x, adj).squeeze(-1), y, mask,
                    penalty_val=args.penalty, ablation_mode=args.ablation_mode
                )

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
                pbar.set_postfix({"Loss": f"{loss.item():.4f}"})

        # Evaluation
        res, _ = evaluate_backtest(model, test_loader, k_list=k_list, horizon=args.horizon, device=DEVICE)
        fold_results.append(res)

        print(f"  Fold {fold + 1} | [K=30] Sharpe: {res['hybrid'][30][0]:.4f} | CumRet: {res['hybrid'][30][1]:.2%}")

        del model, optimizer, A_static, A_dynamic, train_loader, test_loader
        torch.cuda.empty_cache()
        gc.collect()

    # Aggregate results across folds
    mean_results = {m: {} for m in ["hybrid", "st", "har"]}
    std_results = {m: {} for m in ["hybrid", "st", "har"]}
    for m_name in ["hybrid", "st", "har"]:
        for k in k_list:
            metrics_matrix = np.array([f[m_name][k] for f in fold_results])
            mean_results[m_name][k] = np.nanmean(metrics_matrix, axis=0)
            std_results[m_name][k] = np.nanstd(metrics_matrix, axis=0)

    # Print Final Academic Table (Ref_K = 30)
    print(f"\n>>> 📊 Final Evaluation (Top-K = 30)")
    print("-" * 105)
    print(f"{'Metric':<10} | {'TVGM (Hybrid)':<25} | {'ST-only':<25} | {'HAR-only':<25}")
    print("-" * 105)

    metric_names = ["Sharpe", "CumRet", "AnnRet", "AnnVol", "MaxDD", "P@K", "LiftPct"]
    is_pct_list = [False, True, True, True, True, True, True]

    def fmt(m_val, s_val, is_pct):
        if np.isnan(m_val): return "NaN"
        return f"{m_val:.2%}±{s_val:.2%}" if is_pct else f"{m_val:.4f}±{s_val:.4f}"

    for i, m_name in enumerate(metric_names):
        is_pct = is_pct_list[i]
        val_h = fmt(mean_results["hybrid"][30][i], std_results["hybrid"][30][i], is_pct)
        val_s = fmt(mean_results["st"][30][i], std_results["st"][30][i], is_pct)
        val_a = fmt(mean_results["har"][30][i], std_results["har"][30][i], is_pct)
        print(f"{m_name:<10} | {val_h:<25} | {val_s:<25} | {val_a:<25}")
    print("-" * 105 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TVGM Training Pipeline")

    # Core hyperparameters
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--seq_len", type=int, default=20, help="Temporal sequence length")
    parser.add_argument("--sparsity", type=float, default=0.08, help="Graph sparsity target")
    parser.add_argument("--penalty", type=float, default=4.0, help="Risk aversion penalty weight")
    parser.add_argument("--ablation_mode", type=str, default="full", choices=["full", "wo-risk", "wo-rank", "wo-mse"], help="Ablation testing mode")

    # System and Data configs
    parser.add_argument("--data_dir", type=str, default="data/processed", help="Path to preprocessed data")
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--hidden_dim", type=int, default=32, help="Hidden dimensions")
    parser.add_argument("--horizon", type=int, default=5, help="Prediction horizon")

    # Backtest configs
    parser.add_argument("--train_window", type=int, default=700, help="Rolling train window length")
    parser.add_argument("--test_window", type=int, default=126, help="Rolling test window length")
    parser.add_argument("--step_size", type=int, default=126, help="Rolling step size")
    parser.add_argument("--top_k_stocks", type=int, default=300, help="Dynamic universe size")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    print("=== TVGM Training Setup ===")
    print(f"LR: {args.lr} | Seq Len: {args.seq_len} | Sparsity: {args.sparsity} | Penalty: {args.penalty}")

    run_experiment(args)