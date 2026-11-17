"""Reusable training, regularization, and inference engine for Phase 2 DL models."""
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

from config import FLOOR_PENALTY_METERS, BUILDING_PENALTY_METERS
from phase2_features import CoordScaler


def set_seed(seed: int = 42):
    """Seed NumPy + all torch backends for reproducible runs across CPU/CUDA/MPS."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def _apply_domain_augmentation(
    xb: torch.Tensor,
    n_waps: int,
    rep_mode: str,
    ap_drop_p: float,
    noise_std: float,
) -> torch.Tensor:
    """Apply AP-block masking dropout + Gaussian jitter on the RSSI channels only.

    Each representation lays out its channels as [RSSI ... | presence/absence | aggregates].
    Gaussian jitter must perturb only the continuous RSSI channels; adding it to the
    binary presence/absence mask would turn flags into non-binary garbage, and adding it
    to the standardized aggregates double-counts the signal perturbation. `rssi_dim`
    below is the width of the leading continuous block for each mode.
    """
    xb = xb.clone()

    if rep_mode == "R3":
        rssi_dim = 2 * n_waps          # [pow | rel | presence | agg]
    elif rep_mode == "R3_no_rel":
        rssi_dim = n_waps              # [pow | presence | agg]
    elif rep_mode == "R2":
        rssi_dim = n_waps              # [pow]
    elif rep_mode == "R1":
        rssi_dim = n_waps              # [rssi | absence | agg]
    else:
        rssi_dim = n_waps

    if ap_drop_p > 0:
        mask_ap = (torch.rand(xb.size(0), n_waps, device=xb.device) > ap_drop_p).float()
        if rep_mode == "R3":
            # Zero the dropped AP across pow, rel AND its presence flag so the model
            # never sees a "heard" flag for an AP whose RSSI was just masked out.
            xb[:, :n_waps] *= mask_ap
            xb[:, n_waps : 2 * n_waps] *= mask_ap
            xb[:, 2 * n_waps : 3 * n_waps] *= mask_ap
        elif rep_mode == "R3_no_rel":
            xb[:, :n_waps] *= mask_ap
            xb[:, n_waps : 2 * n_waps] *= mask_ap   # presence flag
        elif rep_mode == "R2":
            xb[:, :n_waps] *= mask_ap
        elif rep_mode == "R1":
            # R1 absence indicator sits in [n_waps : 2*n_waps]; drop RSSI and flip its flag.
            xb[:, :n_waps] *= mask_ap
            xb[:, n_waps : 2 * n_waps] = torch.maximum(
                xb[:, n_waps : 2 * n_waps], 1.0 - mask_ap
            )

    if noise_std > 0:
        noise = torch.randn_like(xb[:, :rssi_dim]) * noise_std
        xb[:, :rssi_dim] = xb[:, :rssi_dim] + noise
    return xb


def train_dl_model(
    model: nn.Module,
    X_tr: np.ndarray,
    wid_tr: np.ndarray,
    yb_tr: np.ndarray,
    yf_tr: np.ndarray,
    yc_tr: np.ndarray,
    X_es: np.ndarray,
    wid_es: np.ndarray,
    yb_es: np.ndarray,
    yf_es: np.ndarray,
    yc_es: np.ndarray,
    n_waps: int,
    rep_mode: str = "R3",
    epochs: int = 50,
    batch_size: int = 256,
    lr: float = 1.2e-3,
    weight_decay: float = 1e-2,
    ap_drop_p: float = 0.12,
    noise_std: float = 0.02,
    use_kendall: bool = True,
    use_floor_weights: bool = True,
    sae_pretrain_epochs: int = 0,
    seed: int = 42,
):
    """Train a multitask DL model with early-stopping strictly on `X_es`."""
    set_seed(seed)
    cscaler = CoordScaler().fit(yc_tr)
    yc_tr_s = cscaler.transform(yc_tr)

    if use_floor_weights:
        counts = np.bincount(yf_tr, minlength=5).astype(np.float32)
        flr_weights = torch.tensor((counts.mean() / np.maximum(counts, 1.0)) ** 0.3, dtype=torch.float32)
        ce_f = nn.CrossEntropyLoss(weight=flr_weights, label_smoothing=0.02)
    else:
        ce_f = nn.CrossEntropyLoss(label_smoothing=0.02)
    ce_b = nn.CrossEntropyLoss(label_smoothing=0.02)

    ds_tr = TensorDataset(
        torch.tensor(X_tr, dtype=torch.float32),
        torch.tensor(wid_tr, dtype=torch.long),
        torch.tensor(yb_tr, dtype=torch.long),
        torch.tensor(yf_tr, dtype=torch.long),
        torch.tensor(yc_tr_s, dtype=torch.float32),
    )
    dl_tr = DataLoader(ds_tr, batch_size=batch_size, shuffle=True)

    # Optional Stage-0 Denoising Autoencoder pretraining for SAE_MTL
    if sae_pretrain_epochs > 0 and hasattr(model, "reconstruct"):
        pre_opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        for _ in range(sae_pretrain_epochs):
            model.train()
            for xb, _, _, _, _ in dl_tr:
                noisy = xb * (torch.rand_like(xb) > 0.20).float()
                pre_opt.zero_grad()
                recon = model.reconstruct(noisy)
                loss_rec = F.mse_loss(recon, xb)
                loss_rec.backward()
                pre_opt.step()

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-5)

    X_es_t = torch.tensor(X_es, dtype=torch.float32)
    wid_es_t = torch.tensor(wid_es, dtype=torch.long)

    best_evaal = 1e9
    best_epoch = 0
    best_state = None
    history = {"train_loss": [], "es_evaal": [], "es_floor_acc": [], "es_coord_mean": [], "kendall_sigmas": []}

    t0 = time.time()
    for ep in range(1, epochs + 1):
        model.train()
        ep_loss = 0.0
        n_batches = 0
        for xb, wb, bb, fb, cb in dl_tr:
            xb = _apply_domain_augmentation(xb, n_waps, rep_mode, ap_drop_p, noise_std)
            opt.zero_grad()
            lb, lf, pc_s, _ = model(xb, wb)
            loss_b = ce_b(lb, bb)
            loss_f = ce_f(lf, fb)
            diff = pc_s - cb
            loss_c = torch.sqrt((diff ** 2).sum(dim=-1) + 1e-6).mean()

            if use_kendall and hasattr(model, "log_vars"):
                s0, s1, s2 = model.log_vars[0], model.log_vars[1], model.log_vars[2]
                loss = (
                    (torch.exp(-s0) * loss_b + 0.5 * s0)
                    + (torch.exp(-s1) * loss_f + 0.5 * s1)
                    + (torch.exp(-s2) * loss_c + 0.5 * s2)
                )
            else:
                loss = loss_b + loss_f + 2.0 * loss_c

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step()
            ep_loss += float(loss.item())
            n_batches += 1
        sched.step()

        model.eval()
        with torch.no_grad():
            lb, lf, pc_s, _ = model(X_es_t, wid_es_t)
            pb = lb.argmax(dim=-1).numpy()
            pf = lf.argmax(dim=-1).numpy()
            pc = cscaler.inverse_transform(pc_s.numpy())
            err = np.sqrt(((pc - yc_es) ** 2).sum(axis=1))
            evaal = float((err + FLOOR_PENALTY_METERS * (pf != yf_es) + BUILDING_PENALTY_METERS * (pb != yb_es)).mean())
            flr_acc = float((pf == yf_es).mean())
            coord_mean = float(err.mean())

        history["train_loss"].append(round(ep_loss / max(n_batches, 1), 4))
        history["es_evaal"].append(round(evaal, 4))
        history["es_floor_acc"].append(round(flr_acc, 4))
        history["es_coord_mean"].append(round(coord_mean, 4))
        if hasattr(model, "log_vars"):
            sigmas = [round(float(torch.exp(0.5 * s).item()), 4) for s in model.log_vars]
            history["kendall_sigmas"].append(sigmas)

        if evaal < best_evaal:
            best_evaal = evaal
            best_epoch = ep
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    history["best_epoch"] = best_epoch
    history["best_evaal"] = round(best_evaal, 4)
    history["train_sec"] = round(time.time() - t0, 2)
    return model, cscaler, history


def predict_dl_model(model: nn.Module, X: np.ndarray, wid: np.ndarray, cscaler: CoordScaler):
    """Run inference and return calibrated softmax probabilities, coordinates (m), and latent z."""
    model.eval()
    with torch.no_grad():
        lb, lf, pc_s, z = model(torch.tensor(X, dtype=torch.float32), torch.tensor(wid, dtype=torch.long))
        pb_prob = F.softmax(lb, dim=-1).numpy().astype(np.float32)
        pf_prob = F.softmax(lf, dim=-1).numpy().astype(np.float32)
        pc = cscaler.inverse_transform(pc_s.numpy())
    return pb_prob, pf_prob, pc, z.numpy().astype(np.float32)

