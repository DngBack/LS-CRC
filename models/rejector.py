import torch
import torch.nn as nn
import torch.nn.functional as F

class Rejector(nn.Module):
    """
    Structured Spatial Rejector Network (g_phi).
    Inputs the features h_theta, probability mask p_theta, and uncertainty maps u_theta.
    Predicts pixel-wise acceptance scores.
    """
    def __init__(self, feature_channels=32):
        super().__init__()
        # Inputs: features (e.g. 32) + probability (1) + entropy (1) = feature_channels + 2
        in_channels = feature_channels + 2
        
        # A shallow convolutional decoder to predict the acceptance map
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, kernel_size=3, padding=1)
        )

    def forward(self, features, prob):
        """
        Args:
            features: Tensor of shape (B, C, H, W)
            prob: Tensor of shape (B, 1, H, W)
        """
        # Calculate pixel-wise uncertainty (Entropy)
        # Avoid log(0) with clamping
        p_safe = torch.clamp(prob, 1e-7, 1 - 1e-7)
        entropy = - (p_safe * torch.log(p_safe) + (1 - p_safe) * torch.log(1 - p_safe))
        
        # We can also add margin: abs(prob - 0.5)
        # However, the paper explicitly lists entropy.
        
        # DeepLabV3+ decoder features can be lower resolution than prob.
        # Align spatial size before channel concatenation.
        if features.shape[-2:] != prob.shape[-2:]:
            features = F.interpolate(features, size=prob.shape[-2:], mode="bilinear", align_corners=False)

        # Concatenate features, probability, and entropy
        x = torch.cat([features, prob, entropy], dim=1)
        
        # Predict acceptance score
        logits = self.net(x)
        acceptance_score = torch.sigmoid(logits)
        
        return acceptance_score
