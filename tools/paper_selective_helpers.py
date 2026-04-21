"""
Shared tensor logic for paper figures: entropy vs LS-CRC acceptance, per-image localized risk.
Mirrors evaluate.py / calibrate.py conventions.
"""
from __future__ import annotations

import torch

from utils.losses import compute_spatial_weight_map, get_localized_selective_loss_components


def clamp_prob(prob: torch.Tensor) -> torch.Tensor:
    return torch.clamp(prob, 1e-7, 1 - 1e-7)


def entropy_and_norm(prob: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    p = clamp_prob(prob)
    ent = -(p * torch.log(p) + (1 - p) * torch.log(1 - p))
    max_ent = 0.693
    norm = ent / max_ent
    return ent, norm


def accept_entropy(prob: torch.Tensor, tau: float) -> torch.Tensor:
    _, norm = entropy_and_norm(prob)
    return ((1.0 - norm) >= tau).float()


def accept_lscrc(rejector, features: torch.Tensor, prob: torch.Tensor, tau: float) -> tuple[torch.Tensor, torch.Tensor]:
    scores = rejector(features, prob)
    acc = (scores >= tau).float()
    return acc, scores


def per_image_localized_risk_and_aux(
    y: torch.Tensor, pred: torch.Tensor, accept: torch.Tensor, w_u: torch.Tensor, eps: float = 1e-7
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Returns per-image (B,):
      risk, mean_coverage, fg_accept_ratio, committed_err_frac
    fg_accept_ratio = sum(A*y)/sum(y)  (0 if no FG)
    committed_err_frac = sum(A*|pred-y|)/sum(A+eps)  (fraction of accepted pixels wrong)
    """
    numer, denom = get_localized_selective_loss_components(y, pred, accept, w_u)
    risk = numer / (denom + eps)
    cov = accept.view(accept.size(0), -1).mean(dim=1)
    y_flat = y.view(y.size(0), -1)
    a_flat = accept.view(accept.size(0), -1)
    fg = y_flat.sum(dim=1).clamp_min(eps)
    fg_accept = (a_flat * y_flat).sum(dim=1) / fg
    wrong = (pred - y).abs()
    w_flat = wrong.view(wrong.size(0), -1)
    a_sum = a_flat.sum(dim=1).clamp_min(eps)
    committed_err_frac = (a_flat * w_flat).sum(dim=1) / a_sum
    return risk, cov, fg_accept, committed_err_frac


def committed_error_mask(pred: torch.Tensor, y: torch.Tensor, accept: torch.Tensor) -> torch.Tensor:
    """B,1,H,W bool: accepted and prediction differs from GT."""
    wrong = (pred - y).abs() > 0.5
    acc = accept > 0.5
    return acc & wrong


def accepted_prediction_rgb(
    img_rgb_hwc: torch.Tensor | "np.ndarray",
    pred: torch.Tensor,
    accept: torch.Tensor,
) -> "np.ndarray":
    """Blend: show prediction as green tint on accepted pixels; dim elsewhere. img_rgb_hwc float 0-1 H,W,3 on CPU."""
    import numpy as np

    if isinstance(img_rgb_hwc, torch.Tensor):
        base = img_rgb_hwc.detach().cpu().numpy()
    else:
        base = img_rgb_hwc
    pr = pred[0, 0].detach().cpu().numpy() > 0.5
    acc = accept[0, 0].detach().cpu().numpy() > 0.5
    out = base.copy()
    dim = 0.35
    out = out * dim
    green = np.array([0.15, 0.85, 0.25], dtype=np.float32)
    m = acc & pr
    out[m] = 0.55 * out[m] + 0.45 * green
    m = acc & (~pr)
    out[m] = 0.65 * base[m]  # accepted background: slightly less dimmed
    return np.clip(out, 0, 1)


def error_overlay_rgb(
    img_rgb_hwc,
    pred: torch.Tensor,
    y: torch.Tensor,
    accept: torch.Tensor,
    red_alpha: float = 0.55,
) -> "np.ndarray":
    import numpy as np

    if isinstance(img_rgb_hwc, torch.Tensor):
        base = img_rgb_hwc.detach().cpu().numpy()
    else:
        base = img_rgb_hwc
    cm = committed_error_mask(pred, y, accept)[0, 0].detach().cpu().numpy()
    out = base.copy()
    red = np.array([1.0, 0.05, 0.05], dtype=np.float32)
    out[cm] = (1.0 - red_alpha) * out[cm] + red_alpha * red
    return np.clip(out, 0, 1)


def spatial_weights(y: torch.Tensor, bnd_weight: float = 2.0) -> torch.Tensor:
    return compute_spatial_weight_map(y, bnd_weight=bnd_weight)
