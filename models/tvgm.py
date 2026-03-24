import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers import DenseChebConv  # Import the base graph convolution layer


class SpatioTemporalModel(nn.Module):
    """
    The ST-Branch (Dynamic Spatio-Temporal).
    Captures short-term local evolutionary characteristics among assets
    using Chebyshev graph convolution followed by temporal LSTM integration.
    """

    def __init__(self, in_features, hidden_features, out_features, K, dropout=0.5):
        super(SpatioTemporalModel, self).__init__()
        self.hidden_features = hidden_features

        self.dense_cheb = DenseChebConv(in_features, hidden_features, K=K)
        self.norm = nn.LayerNorm(hidden_features)

        self.lstm = nn.LSTM(
            input_size=hidden_features,
            hidden_size=hidden_features,
            num_layers=1,
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_features, out_features)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, dynamic_adj):
        B, N, T, F_in = x.shape
        x_flat = x.permute(0, 2, 1, 3).reshape(B * T, N, F_in)
        adj_flat = dynamic_adj.unsqueeze(1).repeat(1, T, 1, 1).reshape(B * T, N, N)

        # Spatial Convolution on dynamic graphs
        st_out = self.dense_cheb(x_flat, adj_flat)
        st_out = F.relu(st_out)
        st_out = self.norm(st_out)
        st_out = self.dropout(st_out)

        # Temporal Integration
        st_out = st_out.reshape(B, T, N, self.hidden_features)
        lstm_in = st_out.permute(0, 2, 1, 3).reshape(B * N, T, self.hidden_features)

        lstm_out, _ = self.lstm(lstm_in)
        last_step = lstm_out[:, -1, :]
        last_step = self.dropout(last_step)

        out = self.fc(last_step)
        return out.reshape(B, N, out.shape[-1])


class GSP_HAR_Model(nn.Module):
    """
    The HAR-Branch (Static Spectral).
    Anchors long-term structural trends by projecting signals onto an orthogonal
    feature space spanned by the static Laplacian basis (GFT).
    """

    def __init__(self, in_features, out_features, num_nodes, A_static, q=0.1):
        super(GSP_HAR_Model, self).__init__()
        device = A_static.device

        # Construct static normalized Laplacian
        W_s = 0.5 * (A_static + A_static.T)
        D_s_diag = torch.sum(W_s, dim=1)
        D_inv = torch.diag(torch.pow(D_s_diag, -0.5))
        D_inv[torch.isinf(D_inv)] = 0.
        L = torch.eye(num_nodes, device=device) - (D_inv @ W_s @ D_inv)

        # Eigendecomposition for spectral basis
        eigvals, U = torch.linalg.eigh(L)
        self.register_buffer('U', U)
        self.register_buffer('U_T', U.T)

        self.har_real = nn.Linear(in_features * 3, out_features)

        self.aggregation = nn.Sequential(
            nn.Linear(out_features, out_features * 2),
            nn.ReLU(),
            nn.Linear(out_features * 2, out_features)
        )

    def forward(self, x):
        """
        x: (B, T, N, F)
        return: (B, N, out_features)
        """
        # Graph Fourier Transform (GFT)
        x_gft = torch.einsum('nm,btnf->btmf', self.U_T, x)  # (B,T,N,F)

        # Multi-scale HAR Aggregation in spectral domain
        x1 = x_gft[:, -1, :, :]
        x5 = x_gft[:, -5:, :, :].mean(dim=1) if x_gft.size(1) >= 5 else x_gft.mean(dim=1)
        x20 = x_gft[:, -20:, :, :].mean(dim=1) if x_gft.size(1) >= 20 else x_gft.mean(dim=1)

        feat = torch.cat([x1, x5, x20], dim=-1)  # (B,N,3F)
        filtered = self.har_real(feat)  # (B,N,out_features)

        # Inverse GFT and final aggregation
        out = torch.einsum('nm,bmf->bnf', self.U, filtered)  # (B,N,out_features)
        out = self.aggregation(out)  # (B,N,out_features)
        return out


class TVGM(nn.Module):
    """
    Structurally Decoupled Time-Varying Graph Ensemble Model (TVGM).
    Fuses the ST-branch (dynamic) and HAR-branch (static) via a gated residual network.
    """

    def __init__(self, in_features, hidden_features, out_features, A_static, K=3):
        super(TVGM, self).__init__()
        self.st_model = SpatioTemporalModel(in_features, hidden_features, out_features, K)
        self.har_model = GSP_HAR_Model(in_features, out_features, A_static.shape[0], A_static)

        # Gating Mechanism
        self.gate = nn.Linear(out_features * 2, 1)
        self.delta_proj = nn.Linear(out_features, out_features)

    def forward(self, x, dynamic_adj, return_parts=False):
        # 1. Short-term dynamic prediction
        pred_st = self.st_model(x, dynamic_adj)

        # 2. Long-term structural prediction
        x_permuted = x.permute(0, 2, 1, 3)
        pred_har = self.har_model(x_permuted)

        # 3. Regime-adaptive Gated Fusion
        combined = torch.cat([pred_st, pred_har], dim=-1)
        z = torch.sigmoid(self.gate(combined))

        diff = pred_har - pred_st
        delta = self.delta_proj(diff)

        # Final prediction: Base (ST) + Adaptive Residual (HAR)
        out = pred_st + (1 - z) * delta

        if return_parts:
            return out, pred_st, pred_har, z, delta
        return out