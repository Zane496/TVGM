import numpy as np
import torch

def get_dynamic_static_adj(data_train, threshold=None, last_days=700, target_sparsity=0.12, return_tensor=True, device='cpu'):
    """
    Construct the static correlation graph based on the historical training window.
    This provides the topological foundation for the spectral HAR branch.

    Args:
        data_train (np.array): Shape (T_train, N, F), where column 0 is log_ret.
        threshold (float or None): Fixed threshold for edge retention.
        last_days (int or None): Window size to compute the static graph.
        target_sparsity (float): Desired non-zero edge ratio.
        return_tensor (bool): Whether to return a PyTorch tensor.
        device (str): Device to place the tensor.

    Returns:
        adj (Tensor or np.array): The sparsified static adjacency matrix of shape (N, N).
    """
    # 1. Extract log returns (assume index 0 is log_ret)
    returns = data_train[:, :, 0] + 1e-8  # (T, N)

    # Use only the most recent 'last_days'
    if last_days is not None and returns.shape[0] > last_days:
        returns_used = returns[-last_days:]
    else:
        returns_used = returns

    # 2. Compute Pearson Correlation
    with np.errstate(divide='ignore', invalid='ignore'):
        corr_matrix = np.corrcoef(returns_used, rowvar=False)

    corr_matrix = np.nan_to_num(corr_matrix, 0.0)
    N = corr_matrix.shape[0]

    # 3. Adaptive Thresholding based on target sparsity
    abs_corr = np.abs(corr_matrix)

    if threshold is None and target_sparsity is not None:
        # Control sparsity only on off-diagonal elements
        off_diag_mask = ~np.eye(N, dtype=bool)
        off_vals = abs_corr[off_diag_mask]

        k = int(target_sparsity * off_vals.size)
        if k <= 0:
            thr = np.quantile(off_vals, 0.99)
        else:
            sorted_vals = np.sort(off_vals)
            thr = sorted_vals[-k]
    else:
        thr = 0.45 if threshold is None else threshold

    # 4. Apply mask and force self-loops
    adj = np.zeros((N, N), dtype=np.float32)
    mask = abs_corr > thr
    adj[mask] = corr_matrix[mask]
    np.fill_diagonal(adj, 1.0)

    # 5. Logging
    edge_count = np.count_nonzero(adj)
    sparsity = edge_count / (N * N)
    print(f"  [Static Graph] thr={thr:.4f} | sparsity={sparsity:.2%} | edges={edge_count}")

    if return_tensor:
        return torch.tensor(adj, dtype=torch.float32).to(device)
    else:
        return adj