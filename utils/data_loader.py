import torch
from torch.utils.data import Dataset
import numpy as np

def add_market_context_and_norm(data_3d):
    """
    Cross-sectional Z-score normalization and Market Context Injection.
    """
    market_mean = np.nanmean(data_3d, axis=1, keepdims=True)
    mean = np.nanmean(data_3d, axis=1, keepdims=True)
    std = np.nanstd(data_3d, axis=1, keepdims=True)
    data_norm = (data_3d - mean) / (std + 1e-8)
    market_mean_expanded = np.tile(market_mean, (1, data_3d.shape[1], 1))
    return np.concatenate([data_norm, market_mean_expanded], axis=-1)

def get_causal_dynamic_adjs(returns, window_size, sparsity):
    """
    Construct dynamic rolling correlation graphs for the ST branch.
    """
    T, N = returns.shape
    adjs = []
    for t in range(T):
        if t < window_size:
            adjs.append(np.eye(N, dtype=np.float32))
        else:
            window = returns[t - window_size: t, :]
            with np.errstate(divide='ignore', invalid='ignore'):
                corr = np.nan_to_num(np.corrcoef(window, rowvar=False), 0.0)

            abs_corr = np.abs(corr)
            np.fill_diagonal(abs_corr, 0)
            k = int(N * N * sparsity)
            thr = np.partition(abs_corr.flatten(), -k)[-k] if k > 0 else 0.5
            adj = np.where(abs_corr >= thr, corr, 0.0).astype(np.float32)
            np.fill_diagonal(adj, 1.0)
            adjs.append(adj)
    return np.stack(adjs)

class RollingStockDataset(Dataset):
    """
    PyTorch Dataset for generating spatio-temporal sliding windows.
    """
    def __init__(self, features, targets, rets_raw, masks, adjs, seq_len):
        self.features = torch.FloatTensor(features)
        self.targets = torch.FloatTensor(targets)
        self.rets_raw = torch.FloatTensor(rets_raw)
        self.masks = torch.FloatTensor(masks)
        self.adjs = torch.FloatTensor(adjs)
        self.seq_len = seq_len

    def __len__(self):
        return len(self.features) - self.seq_len + 1

    def __getitem__(self, idx):
        x = self.features[idx: idx + self.seq_len]
        t_idx = idx + self.seq_len - 1
        return x.permute(1, 0, 2), self.targets[t_idx], self.rets_raw[t_idx], self.adjs[t_idx], self.masks[t_idx]