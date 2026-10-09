import torch
import torch.nn as nn


def risk_averse_loss(preds, targets_scaled, mask, penalty_val=4.0):
    """
    Risk-Averse Combined Loss Function.
    Imposes asymmetric penalties on high-risk prediction biases (Bull Traps).
    """
    mask = mask.float()
    valid_idx = mask > 0.5

    clipped_targets = torch.clamp(targets_scaled, min=-15.0, max=15.0)
    mse_loss = (preds - clipped_targets) ** 2

    batch_mean_pred = torch.clamp(preds.mean(), max=0.0)
    batch_mean_target = targets_scaled.mean()

    # Asymmetric penalty for Bull Traps
    penalty_mask = (clipped_targets < batch_mean_target) & (preds > batch_mean_pred)
    weights = torch.ones_like(mse_loss)
    weights[penalty_mask] = penalty_val

    weighted_mse = (mse_loss * weights * mask).sum() / (mask.sum() + 1e-8)


    p_v, t_v = preds[valid_idx], targets_scaled[valid_idx]

    if len(p_v) < 10:
        rank_loss = torch.tensor(0.0, device=preds.device)
    else:
        n_pairs = min(len(p_v), 2000)
        idx_1, idx_2 = torch.randperm(len(p_v))[:n_pairs], torch.randperm(len(p_v))[:n_pairs]
        target_diff = torch.sign(t_v[idx_1] - t_v[idx_2])
        v_mask = target_diff != 0
        if v_mask.sum() == 0:
            rank_loss = torch.tensor(0.0, device=preds.device)
        else:
            rank_loss = nn.MarginRankingLoss(margin=0.05)(p_v[idx_1][v_mask], p_v[idx_2][v_mask], target_diff[v_mask])

    total_loss = weighted_mse + rank_loss

    if total_loss.item() == 0.0:
        total_loss = torch.tensor(0.0, device=preds.device, requires_grad=True)

    return total_loss, weighted_mse.item(), rank_loss.item()