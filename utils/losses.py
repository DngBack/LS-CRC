import torch
import torch.nn.functional as F

def compute_spatial_weight_map(y_true, bnd_weight=2.0, bnd_band_size=3):
    """
    Computes a spatially localized weight map emphasizing the foreground boundary.
    Args:
        y_true: Ground truth mask (B, 1, H, W)
        bnd_weight: Extra importance assigned to the boundary pixels
        bnd_band_size: Kernel size for morphological operations (dilation/erosion)
    Returns:
        w_u: Weight map (B, 1, H, W)
    """
    # Morphological dilation and erosion using MaxPool to find boundary band
    kernel_size = bnd_band_size
    pad = kernel_size // 2
    
    dilation = F.max_pool2d(y_true, kernel_size=kernel_size, stride=1, padding=pad)
    erosion = -F.max_pool2d(-y_true, kernel_size=kernel_size, stride=1, padding=pad)
    
    boundary_band = dilation - erosion
    w_u = torch.ones_like(y_true) + bnd_weight * boundary_band
    
    return w_u

def get_localized_selective_loss_components(y_true, pred, A_u, w_u):
    """
    Returns the numerator and denominator for the localized loss for each image in the batch.
    Args:
        y_true: Ground truth (B, 1, H, W)
        pred: Hard binary prediction (B, 1, H, W)
        A_u: Binary acceptance mask or continuous acceptance score (B, 1, H, W)
        w_u: Spatial weight map (B, 1, H, W)
    Returns:
        numerator: Bounded error penalty per item (B)
        denominator: Normalization factor per item (B)
    """
    # L_loc = [sum w_u * 1{y_u=1} * A_u * 1{y_hat=0}] / sum[w_u * 1{y_u=1}]
    # which is essentially weighted false negative rate limited to accepted regions.
    
    false_negatives = y_true * (1 - pred)
    
    # Numerator counts error on accepted pixels
    numerator = (w_u * false_negatives * A_u).view(y_true.size(0), -1).sum(dim=1)
    
    # Denominator measures total weight of foreground
    denominator = (w_u * y_true).view(y_true.size(0), -1).sum(dim=1) + 1e-7
    
    return numerator, denominator

def smoothness_loss(s_phi):
    """
    Penalty to enforce spatial coherence on the acceptance scores.
    Args:
        s_phi: Acceptance score map (B, 1, H, W)
    """
    diff_x = torch.abs(s_phi[:, :, :, :-1] - s_phi[:, :, :, 1:])
    diff_y = torch.abs(s_phi[:, :, :-1, :] - s_phi[:, :, 1:, :])
    return diff_x.mean() + diff_y.mean()

def localized_surrogate_risk(y_true, prob, s_phi, w_u):
    """
    Differentiable surrogate for the localized risk, used for joint fine-tuning in Ablation E.
    """
    surrogate_fn = w_u * y_true * s_phi * (1 - prob)
    numerator = surrogate_fn.view(y_true.size(0), -1).sum(dim=1)
    denominator = (w_u * y_true).view(y_true.size(0), -1).sum(dim=1) + 1e-7
    return (numerator / denominator).mean()
