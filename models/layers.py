import torch
import torch.nn as nn


class DenseChebConv(nn.Module):
    """
    Standard ChebNet (Defferrard et al., 2016) implemented on dense adjacency.

    We build:
        S = D^{-1/2} A D^{-1/2}
        L = I - S
        L_tilde = 2/lambda_max * L - I
    and use Chebyshev recursion:
        T0(x)=x, T1(x)=L_tilde x, Tk(x)=2 L_tilde T_{k-1}(x) - T_{k-2}(x)

    With lambda_max≈2 (common for normalized Laplacian), L_tilde = L - I = -S,
    which matches your previous operator `laplacian = -norm_adj`, hence it will
    not change information flow, only standardizes the formulation.
    """

    def __init__(self, in_channels, out_channels, K, lambda_max=2.0, eps=1e-12):
        super(DenseChebConv, self).__init__()
        self.K = K
        self.lambda_max = float(lambda_max)
        self.eps = eps

        self.weight = nn.Parameter(torch.Tensor(K, in_channels, out_channels))
        self.bias = nn.Parameter(torch.Tensor(out_channels))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        nn.init.zeros_(self.bias)

    def forward(self, x, adj):
        B, N, In = x.shape

        deg = torch.sum(torch.abs(adj), dim=2)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        deg_mat = torch.diag_embed(deg_inv_sqrt)

        norm_adj = torch.bmm(torch.bmm(deg_mat, adj), deg_mat)

        L_tilde = -norm_adj
        Tx_0 = x
        Tx_1 = torch.bmm(L_tilde, x)

        score = torch.matmul(Tx_0, self.weight[0])
        if self.K > 1:
            score = score + torch.matmul(Tx_1, self.weight[1])

        prev_Tx = Tx_1
        prev_prev_Tx = Tx_0
        for k in range(2, self.K):
            term1 = 2 * torch.bmm(L_tilde, prev_Tx)
            Tx_k = term1 - prev_prev_Tx
            score = score + torch.matmul(Tx_k, self.weight[k])
            prev_prev_Tx = prev_Tx
            prev_Tx = Tx_k

        return score + self.bias
