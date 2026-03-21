import torch

def generate_pseudo_labels(prob, target_mask, high_conf_thresh=0.8, low_conf_thresh=0.2):
    """
    Creates pseudo-labels 'r*' for abstention.
    1 (safe to accept): correct prediction with high confidence.
    0 (unsafe): incorrect prediction or prediction with high uncertainty.
    -1 (ignore): neither.
    """
    with torch.no_grad():
        pred = (prob >= 0.5).float()
        correct = (pred == target_mask)
        
        # Determine confident areas
        high_conf = (prob > high_conf_thresh) | (prob < low_conf_thresh)
        
        unsafe = (~correct) | (~high_conf)
        safe = correct & high_conf
        
        pseudo_labels = torch.full_like(prob, -1.0)
        # We mark unsafe first, and safe overwrites
        pseudo_labels[unsafe] = 0.0
        pseudo_labels[safe] = 1.0
        
    return pseudo_labels
