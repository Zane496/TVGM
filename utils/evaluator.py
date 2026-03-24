import torch
import numpy as np
from .metrics import calculate_metrics

def evaluate_backtest(model, loader, k_list, horizon, device):
    """
    Core backtesting engine. Iterates over the test loader, constructs dynamic
    Top-K portfolios, and evaluates risk-adjusted performance.
    """
    model.eval()
    modes = ["hybrid", "st", "har"]
    metrics_tracker = {
        m: {k: {"returns": [], "precisions": []} for k in k_list}
        for m in modes
    }
    market_precisions = []
    step_counter = 0

    with torch.no_grad():
        for x, _, ret_true, adj, mask in loader:
            x, adj, mask, ret_true = x.to(device), adj.to(device), mask.to(device), ret_true.to(device)
            out, pred_st, pred_har, _, _ = model(x, adj, return_parts=True)
            pred_map = {"hybrid": out.squeeze(-1), "st": pred_st.squeeze(-1), "har": pred_har.squeeze(-1)}

            for i in range(ret_true.shape[0]):
                if step_counter % horizon != 0:
                    step_counter += 1
                    continue

                r_day, m_day = ret_true[i], mask[i]
                valid_idx = m_day > 0.5
                if valid_idx.sum() < (max(k_list) + 5):
                    step_counter += 1
                    continue

                r_v = r_day[valid_idx]
                market_precisions.append((r_v > 0).float().mean().item())

                for mname in modes:
                    p_day = pred_map[mname][i][valid_idx]
                    dynamic_threshold = min(0.0, p_day.mean().item())
                    pos_mask = p_day > dynamic_threshold

                    if pos_mask.sum() >= 5:
                        p_sel, r_sel = p_day[pos_mask], r_v[pos_mask]
                        sorted_indices = torch.argsort(p_sel, descending=True)
                        p_sel_sorted = p_sel[sorted_indices]
                        r_sel_sorted = r_sel[sorted_indices]

                        for k in k_list:
                            cur_k = min(k, len(p_sel_sorted))
                            if cur_k == 0:
                                metrics_tracker[mname][k]["returns"].append(0.0)
                                metrics_tracker[mname][k]["precisions"].append(np.nan)
                                continue

                            r_topk_tensor = r_sel_sorted[:cur_k]
                            metrics_tracker[mname][k]["returns"].append(torch.mean(r_topk_tensor).item())
                            metrics_tracker[mname][k]["precisions"].append((r_topk_tensor > 0).float().mean().item())
                    else:
                        for k in k_list:
                            metrics_tracker[mname][k]["returns"].append(0.0)
                            metrics_tracker[mname][k]["precisions"].append(np.nan)
                step_counter += 1

    results = {m: {} for m in modes}
    for m in modes:
        for k in k_list:
            results[m][k] = calculate_metrics(
                metrics_tracker[m][k]["returns"],
                metrics_tracker[m][k]["precisions"],
                market_precisions,
                horizon=horizon
            )
    return results, metrics_tracker