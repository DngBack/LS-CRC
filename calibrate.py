import torch
import torch.nn.functional as F
import numpy as np
from utils.losses import compute_spatial_weight_map, get_localized_selective_loss_components


def evaluate_risk(backbone, rejector, dataloader, device, tau, method="lscrc"):
    """
    Evaluates empirical risk expected for a given threshold tau using a specific method.
    methods:
        'lscrc': using rejector spatial scores >= tau
        'entropy': accept if (1 - norm_entropy) >= tau
        'max_softmax': accept if max(prob, 1-prob) >= tau
        'standard_crc': global image accept if mean max-prob >= tau (else reject all pixels)
        'spatial_weighted_cp': accept if blended(max-prob, spatial smoothness of prob) >= tau
    """
    backbone.eval()
    if rejector:
        rejector.eval()

    risks = []
    coverages = []

    with torch.no_grad():
        for x, y, tags, _ in dataloader:
            x, y = x.to(device), y.to(device)
            _, prob, features = backbone(x)
            pred = (prob >= 0.5).float()

            p_safe = torch.clamp(prob, 1e-7, 1 - 1e-7)
            entropy = -(p_safe * torch.log(p_safe) + (1 - p_safe) * torch.log(1 - p_safe))

            if method == "lscrc":
                scores = rejector(features, prob)
                A_u = (scores >= tau).float()
            elif method == "entropy":
                max_entropy = 0.693  # roughly log(2)
                norm_entropy = entropy / max_entropy
                A_u = ((1.0 - norm_entropy) >= tau).float()
            elif method == "max_softmax":
                max_prob = torch.max(prob, 1 - prob)
                A_u = (max_prob >= tau).float()
            elif method == "standard_crc":
                max_prob = torch.max(prob, 1 - prob)
                conf = max_prob.view(max_prob.size(0), -1).mean(dim=1)
                accept_image = (conf >= tau).float().view(-1, 1, 1, 1)
                A_u = accept_image * torch.ones_like(prob)
            elif method == "spatial_weighted_cp":
                max_prob = torch.max(prob, 1 - prob)
                gx = prob[:, :, :, 1:] - prob[:, :, :, :-1]
                gy = prob[:, :, 1:, :] - prob[:, :, :-1, :]
                gx = F.pad(gx, (0, 1, 0, 0), mode="replicate")
                gy = F.pad(gy, (0, 0, 0, 1), mode="replicate")
                grad_mag = torch.sqrt(gx * gx + gy * gy + 1e-8)
                gm_flat = grad_mag.view(grad_mag.size(0), -1)
                gm_max = gm_flat.max(dim=1, keepdim=True).values.clamp(min=1e-8)
                gm_norm = grad_mag / gm_max.view(grad_mag.size(0), 1, 1, 1)
                spatial_conf = 1.0 - gm_norm
                score = 0.5 * max_prob + 0.5 * spatial_conf
                A_u = (score >= tau).float()
            else:
                A_u = torch.ones_like(prob)

            w_u = compute_spatial_weight_map(y, bnd_weight=2.0)
            numerators, denominators = get_localized_selective_loss_components(y, pred, A_u, w_u)

            risk = numerators / denominators
            coverage = A_u.view(A_u.size(0), -1).mean(dim=1)

            risks.extend(risk.cpu().numpy())
            coverages.extend(coverage.cpu().numpy())

    return np.mean(risks), np.mean(coverages)


def calibrate_threshold(
    backbone, rejector, dataloader, device, alpha, method="lscrc", num_thresholds=500
):
    """
    Split Conformal Risk Control: largest coverage such that cal risk <= alpha - 1/n.
    num_thresholds: grid resolution on [0.01, 0.99] (higher = less saturation between alphas).
    """
    n = len(dataloader.dataset)
    target = alpha - (1.0 / n)

    thresholds = np.linspace(0.01, 0.99, int(num_thresholds))
    best_tau = None
    best_cov = -1

    for tau in thresholds:
        risk, cov = evaluate_risk(backbone, rejector, dataloader, device, tau, method)
        if risk <= target:
            if cov > best_cov:
                best_cov = cov
                best_tau = tau

    if best_tau is None:
        return 0.999, 0.0, 0.0
    return best_tau, risk, best_cov
