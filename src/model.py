"""CGCNN model (Xie & Grossman, PRL 120, 145301 (2018)).

- GaussianSmearing : bond features u_ij = exp(-(d-mu_k)^2/2sigma^2)
- SimpleConv       : original convolution, Eq. (4)
- CGCNNConv        : modified sigmoid-gated convolution, Eq. (5)
- CGCNN            : R convs -> mean pooling -> [aux fusion] -> FC(Linear-BN-Softplus)
                     -> scalar; optional interpretable linear-pooling head.
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool


class GaussianSmearing(nn.Module):
    def __init__(self, cutoff=6.0, bins=41):
        super().__init__()
        self.register_buffer("centers", torch.linspace(0, cutoff, bins))
        self.register_buffer("width", torch.tensor(cutoff / (bins - 1)))

    def forward(self, d):                                # d: [E]
        x = (d.unsqueeze(-1) - self.centers) / self.width
        return torch.exp(-0.5 * x ** 2)                  # [E, bins]


class CGCNNConv(nn.Module):
    """Eq. (5): v_i <- v_i + sum_{j,k} sigma(z W_f + b_f) ⊙ g(z W_s + b_s),
    z = v_i ⊕ v_j ⊕ u_ij,  g = softplus."""
    def __init__(self, h, e_dim):
        super().__init__()
        self.lin_f = nn.Linear(2 * h + e_dim, h)         # gate
        self.lin_s = nn.Linear(2 * h + e_dim, h)         # message

    def forward(self, x, edge_index, u):
        src, dst = edge_index[0], edge_index[1]          # src=j (nbr), dst=i (center)
        z = torch.cat([x[dst], x[src], u], dim=-1)
        m = torch.sigmoid(self.lin_f(z)) * F.softplus(self.lin_s(z))
        return x.index_add(0, dst, m)                    # residual aggregation


class SimpleConv(nn.Module):
    """Eq. (4): v_i <- g(sum_j,k [v_j ⊕ u_ij] W_c + v_i W_s + b)."""
    def __init__(self, h, e_dim):
        super().__init__()
        self.lin_c = nn.Linear(h + e_dim, h)
        self.lin_s = nn.Linear(h, h)

    def forward(self, x, edge_index, u):
        src, dst = edge_index[0], edge_index[1]
        msg = self.lin_c(torch.cat([x[src], u], dim=-1))
        agg = torch.zeros_like(x).index_add(0, dst, msg)
        return F.softplus(agg + self.lin_s(x))


class CGCNN(nn.Module):
    def __init__(self, cfg, aux_dim=0, num_elements=119):
        super().__init__()
        h = cfg.hidden
        self.use_aux = bool(cfg.use_aux and aux_dim > 0)
        self.linear_pool = bool(cfg.linear_pool)
        self.z_emb = nn.Embedding(num_elements, h)       # ≡ one-hot -> linear (paper)
        self.smear = GaussianSmearing(cfg.cutoff, cfg.gauss_bins)
        conv_cls = CGCNNConv if cfg.conv_type == "modified" else SimpleConv
        self.convs = nn.ModuleList([conv_cls(h, cfg.gauss_bins)
                                    for _ in range(cfg.n_conv)])
        self.conv_bn = nn.BatchNorm1d(h)
        if self.linear_pool:                             # interpretable head
            self.site_out = nn.Linear(h, 1)
        else:
            din = h + (aux_dim if self.use_aux else 0)
            layers = []
            for f_dim in cfg.fc_hidden:
                layers += [nn.Linear(din, f_dim), nn.BatchNorm1d(f_dim),
                           nn.Softplus(), nn.Dropout(cfg.dropout)]
                din = f_dim
            self.fc = nn.Sequential(*layers)
            self.out = nn.Linear(din, 1)

    def forward(self, data, return_sites=False):
        x = self.z_emb(data.z.clamp(max=118))            # [N, h]
        u = self.smear(data.edge_dist)                   # [E, bins]
        for conv in self.convs:
            x = conv(x, data.edge_index, u)
        if self.linear_pool:
            site = self.site_out(x).squeeze(-1)          # per-atom contribution (eV)
            pred = global_mean_pool(site, data.batch)    # size-invariant pooling
            return (pred, site) if return_sites else pred
        pooled = F.softplus(self.conv_bn(global_mean_pool(x, data.batch)))
        if self.use_aux:
            pooled = torch.cat([pooled, data.aux], dim=-1)
        return self.out(self.fc(pooled)).squeeze(-1)
