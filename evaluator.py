import torch
import numpy as np
from .metrics import calculate_metrics


def evaluate_backtest(model, loader, k_list, horizon, device):
    """
    Core backtesting engine for the full TVGM model.
    Constructs dynamic Top-K portfolios and evaluates
    risk-adjusted performance.
    """
    model.eval()

    metrics_tracker = {
        "hybrid": {
            k: {"returns": [], "precisions": []}
            for k in k_list
        }
    }

    market_precisions = []
    step_counter = 0

    with torch.no_grad():
        for x, _, ret_true, adj, mask in loader:
            x = x.to(device)
            adj = adj.to(device)
            mask = mask.to(device)
            ret_true = ret_true.to(device)

            # Full TVGM prediction
            predictions = model(x, adj).squeeze(-1)

            for i in range(ret_true.shape[0]):
                if step_counter % horizon != 0:
                    step_counter += 1
                    continue

                r_day = ret_true[i]
                m_day = mask[i]

                valid_idx = m_day > 0.5

                if valid_idx.sum() < (max(k_list) + 5):
                    step_counter += 1
                    continue

                r_v = r_day[valid_idx]
                p_day = predictions[i][valid_idx]

                market_precisions.append(
                    (r_v > 0).float().mean().item()
                )

                # Dynamic prediction threshold
                dynamic_threshold = min(
                    0.0, p_day.mean().item()
                )

                pos_mask = p_day > dynamic_threshold

                if pos_mask.sum() >= 5:
                    p_sel = p_day[pos_mask]
                    r_sel = r_v[pos_mask]

                    sorted_indices = torch.argsort(
                        p_sel, descending=True
                    )
                    r_sel_sorted = r_sel[sorted_indices]

                    for k in k_list:
                        cur_k = min(k, len(r_sel_sorted))

                        if cur_k == 0:
                            metrics_tracker["hybrid"][k]["returns"].append(0.0)
                            metrics_tracker["hybrid"][k]["precisions"].append(np.nan)
                            continue

                        r_topk_tensor = r_sel_sorted[:cur_k]

                        metrics_tracker["hybrid"][k]["returns"].append(
                            torch.mean(r_topk_tensor).item()
                        )

                        metrics_tracker["hybrid"][k]["precisions"].append(
                            (r_topk_tensor > 0).float().mean().item()
                        )

                else:
                    # Stay in cash when fewer than 5 stocks qualify
                    for k in k_list:
                        metrics_tracker["hybrid"][k]["returns"].append(0.0)
                        metrics_tracker["hybrid"][k]["precisions"].append(np.nan)

                step_counter += 1

    # Evaluate full TVGM only
    results = {"hybrid": {}}

    for k in k_list:
        results["hybrid"][k] = calculate_metrics(
            metrics_tracker["hybrid"][k]["returns"],
            metrics_tracker["hybrid"][k]["precisions"],
            market_precisions,
            horizon=horizon
        )

    return results, metrics_tracker