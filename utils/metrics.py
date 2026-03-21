import torch
import numpy as np

def compute_dice_iou(y_true, y_pred, epsilon=1e-5):
    """
    Computes generic Dice and IoU over the batch.
    """
    intersection = (y_true * y_pred).sum(dim=(1,2,3))
    union = y_true.sum(dim=(1,2,3)) + y_pred.sum(dim=(1,2,3)) - intersection
    
    iou = (intersection + epsilon) / (union + epsilon)
    dice = 2.0 * (intersection + epsilon) / (y_true.sum(dim=(1,2,3)) + y_pred.sum(dim=(1,2,3)) + epsilon)
    
    return dice.mean(), iou.mean()

def compute_selective_metrics(y_true, y_pred, A_u, epsilon=1e-5):
    """
    Computes metrics specific to the selective predictions.
    Args:
        A_u: Binary acceptance mask
    """
    # Coverage is simply the percentage of pixels accepted
    coverage = A_u.mean(dim=(1,2,3))
    
    # Retained Dice is computed only on the accepted pixels
    intersection = (y_true * y_pred * A_u).sum(dim=(1,2,3))
    total_area = (y_true * A_u).sum(dim=(1,2,3)) + (y_pred * A_u).sum(dim=(1,2,3))
    retained_dice = 2.0 * (intersection + epsilon) / (total_area + epsilon)
    
    return coverage.mean(), retained_dice.mean()

def cvar_risk(risks, alpha=0.9):
    """
    Compute Tail Conditional Value At Risk (CVaR).
    Args:
        risks: List or array of image-level risks Z_target.
        alpha: The quantile definition (e.g. 0.9 for CVaR_0.9)
    """
    if len(risks) == 0:
        return 0.0
    arr = np.array(risks)
    q = np.quantile(arr, alpha)
    tail_risks = arr[arr >= q]
    if len(tail_risks) == 0:
        return np.max(arr)
    return np.mean(tail_risks)

def worst_percentile_risk(risks, q=0.9):
    """
    Returns the exact percentile worst risk (e.g. Worst-10% risk -> 90th percentile).
    """
    if len(risks) == 0:
        return 0.0
    return np.quantile(np.array(risks), q)
