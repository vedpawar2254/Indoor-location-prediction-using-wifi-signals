"""Phase 2 Deep Learning Architectures & Multitask Loss Formulations.

Implements a principled progression of deep learning architectures for Wi-Fi
fingerprint indoor localization:
  1. SingleTaskMLP  : Three unshared MLPs (ablation confirming multitask gain).
  2. MTL_MLP        : Phase-1 proposed plain shared MLP trunk feeding 3 heads.
  3. SAE_MTL        : Stacked Denoising Autoencoder + Multitask fine-tuning
                      (Nowicki & Wietrzykowski, ICRA 2017 baseline).
  4. CNN1D_MTL      : 1D Convolutional Multitask Network over 3-channel AP spectrum
                      (inspired by Song et al., IEEE Access 2019 CNNLoc).
  5. HierResMTL     : Flagship Hierarchical Residual Multitask Network featuring:
                        - Strongest-AP Embedding
                        - Pre-LayerNorm Residual MLP Trunk
                        - FiLM Building-to-Floor/Coordinate Conditioning
                        - 64-d Spatial Bottleneck
                        - Kendall et al. (2018) Homoscedastic Uncertainty Loss
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    """Pre-LayerNorm Residual Block for sparse tabular RF representations."""

    def __init__(self, dim: int, dropout: float = 0.25):
        super().__init__()
        self.block = nn.Sequential(
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class SingleTaskBranch(nn.Module):
    """Single-target MLP branch used inside SingleTaskMLP."""

    def __init__(self, in_dim: int, out_dim: int, hidden: int = 256, dropout: float = 0.25):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.LayerNorm(hidden // 2),
            nn.GELU(),
            nn.Dropout(dropout * 0.6),
            nn.Linear(hidden // 2, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SingleTaskMLP(nn.Module):
    """Three completely independent single-task MLPs (no shared trunk)."""

    def __init__(self, in_dim: int, n_waps: int, hidden: int = 256, dropout: float = 0.25, emb_dim: int = 16):
        super().__init__()
        self.wap_emb_b = nn.Embedding(n_waps + 1, emb_dim)
        self.wap_emb_f = nn.Embedding(n_waps + 1, emb_dim)
        self.wap_emb_c = nn.Embedding(n_waps + 1, emb_dim)
        d = in_dim + emb_dim
        self.bld_net = SingleTaskBranch(d, 3, hidden=hidden, dropout=dropout)
        self.flr_net = SingleTaskBranch(d, 5, hidden=hidden, dropout=dropout)
        self.crd_net = SingleTaskBranch(d, 2, hidden=hidden, dropout=dropout)
        self.log_vars = nn.Parameter(torch.zeros(3))

    def forward(self, x: torch.Tensor, wap_id: torch.Tensor):
        lb = self.bld_net(torch.cat([x, self.wap_emb_b(wap_id)], dim=-1))
        lf = self.flr_net(torch.cat([x, self.wap_emb_f(wap_id)], dim=-1))
        coords = self.crd_net(torch.cat([x, self.wap_emb_c(wap_id)], dim=-1))
        return lb, lf, coords, x[:, :64]


class MTL_MLP(nn.Module):
    """Plain shared Multitask MLP trunk feeding Building, Floor, and Coordinate heads."""

    def __init__(self, in_dim: int, n_waps: int, hidden: int = 384, dropout: float = 0.25, emb_dim: int = 16):
        super().__init__()
        self.wap_emb = nn.Embedding(n_waps + 1, emb_dim)
        self.trunk = nn.Sequential(
            nn.Linear(in_dim + emb_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 128),
            nn.LayerNorm(128),
            nn.GELU(),
        )
        self.bld_head = nn.Linear(128, 3)
        self.flr_head = nn.Linear(128, 5)
        self.crd_head = nn.Linear(128, 2)
        self.log_vars = nn.Parameter(torch.zeros(3))

    def forward(self, x: torch.Tensor, wap_id: torch.Tensor):
        h = self.trunk(torch.cat([x, self.wap_emb(wap_id)], dim=-1))
        return self.bld_head(h), self.flr_head(h), self.crd_head(h), h


class SAE_MTL(nn.Module):
    """Stacked Denoising Autoencoder + Multitask Fine-Tuning (Nowicki & Wietrzykowski 2017)."""

    def __init__(self, in_dim: int, n_waps: int, bottleneck: int = 128, dropout: float = 0.25, emb_dim: int = 16):
        super().__init__()
        self.in_dim = in_dim
        self.wap_emb = nn.Embedding(n_waps + 1, emb_dim)
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout * 0.7),
            nn.Linear(256, bottleneck),
            nn.LayerNorm(bottleneck),
            nn.GELU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck, 256),
            nn.GELU(),
            nn.Linear(256, 512),
            nn.GELU(),
            nn.Linear(512, in_dim),
        )
        head_in = bottleneck + emb_dim
        self.bld_head = nn.Sequential(nn.Linear(head_in, 64), nn.GELU(), nn.Linear(64, 3))
        self.flr_head = nn.Sequential(nn.Linear(head_in, 96), nn.GELU(), nn.Linear(96, 5))
        self.crd_head = nn.Sequential(nn.Linear(head_in, 96), nn.GELU(), nn.Linear(96, 2))
        self.log_vars = nn.Parameter(torch.zeros(3))

    def reconstruct(self, x_noisy: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x_noisy))

    def forward(self, x: torch.Tensor, wap_id: torch.Tensor):
        z = self.encoder(x)
        h = torch.cat([z, self.wap_emb(wap_id)], dim=-1)
        return self.bld_head(h), self.flr_head(h), self.crd_head(h), z


class CNN1D_MTL(nn.Module):
    """1D Convolutional Multitask Network over 3-channel AP spectra (CNNLoc-inspired)."""

    def __init__(self, in_dim: int, n_waps: int, dropout: float = 0.25, emb_dim: int = 16):
        super().__init__()
        self.n_waps = n_waps
        self.n_agg = in_dim - 3 * n_waps
        self.wap_emb = nn.Embedding(n_waps + 1, emb_dim)

        self.conv = nn.Sequential(
            nn.Conv1d(3, 48, kernel_size=7, stride=2, padding=3),
            nn.GroupNorm(8, 48),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Conv1d(48, 96, kernel_size=5, stride=2, padding=2),
            nn.GroupNorm(8, 96),
            nn.GELU(),
            nn.Dropout(dropout * 0.6),
            nn.Conv1d(96, 160, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, 160),
            nn.GELU(),
        )
        # Adaptive avg + max pooling over sequence -> 320 + agg + emb
        feat_dim = 320 + self.n_agg + emb_dim
        self.proj = nn.Sequential(
            nn.Linear(feat_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.GELU(),
        )
        self.bld_head = nn.Linear(128, 3)
        self.flr_head = nn.Linear(128, 5)
        self.crd_head = nn.Linear(128, 2)
        self.log_vars = nn.Parameter(torch.zeros(3))

    def forward(self, x: torch.Tensor, wap_id: torch.Tensor):
        b = x.size(0)
        ap_part = x[:, : 3 * self.n_waps].view(b, 3, self.n_waps)
        agg_part = x[:, 3 * self.n_waps :]
        c = self.conv(ap_part)
        c_pool = torch.cat([c.mean(dim=-1), c.amax(dim=-1)], dim=-1)
        h = self.proj(torch.cat([c_pool, agg_part, self.wap_emb(wap_id)], dim=-1))
        return self.bld_head(h), self.flr_head(h), self.crd_head(h), h


class HierResMTL(nn.Module):
    """Hierarchical Residual Multitask Network with FiLM Conditioning & Kendall Loss."""

    def __init__(
        self,
        in_dim: int,
        n_waps: int,
        hidden: int = 384,
        n_blocks: int = 2,
        dropout: float = 0.25,
        emb_dim: int = 16,
        use_film: bool = True,
    ):
        super().__init__()
        self.use_film = use_film
        self.wap_emb = nn.Embedding(n_waps + 1, emb_dim)
        self.input_proj = nn.Sequential(
            nn.Linear(in_dim + emb_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
        )
        self.trunk = nn.Sequential(*[ResBlock(hidden, dropout=dropout) for _ in range(n_blocks)])
        self.trunk_norm = nn.LayerNorm(hidden)

        self.bld_head = nn.Sequential(
            nn.Linear(hidden, 128),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(128, 3),
        )
        self.bld_film = nn.Linear(3, hidden * 2) if use_film else None

        self.flr_head = nn.Sequential(
            nn.Linear(hidden + 3, 192),
            nn.LayerNorm(192),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(192, 5),
        )
        self.bottleneck = nn.Sequential(
            nn.Linear(hidden + 3 + 5, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(dropout * 0.4),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.GELU(),
        )
        self.crd_head = nn.Linear(64, 2)
        self.log_vars = nn.Parameter(torch.zeros(3))

    def forward(self, x: torch.Tensor, wap_id: torch.Tensor):
        h = self.input_proj(torch.cat([x, self.wap_emb(wap_id)], dim=-1))
        h = self.trunk_norm(self.trunk(h))
        logits_b = self.bld_head(h)
        p_b = F.softmax(logits_b, dim=-1)

        if self.use_film:
            gamma, beta = self.bld_film(p_b).chunk(2, dim=-1)
            h_cond = h * (1.0 + gamma) + beta
        else:
            h_cond = h

        logits_f = self.flr_head(torch.cat([h_cond, p_b], dim=-1))
        p_f = F.softmax(logits_f, dim=-1)
        z = self.bottleneck(torch.cat([h_cond, p_b, p_f], dim=-1))
        coords = self.crd_head(z)
        return logits_b, logits_f, coords, z

