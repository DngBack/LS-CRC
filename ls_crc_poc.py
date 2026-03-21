import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import scipy.ndimage
import matplotlib.pyplot as plt
import os

torch.manual_seed(42)
np.random.seed(42)

# ==========================================
# 1. Dataset
# ==========================================
class SyntheticShapesDataset(Dataset):
    def __init__(self, num_samples, img_size=64):
        self.num_samples = num_samples
        self.img_size = img_size
        self.images, self.masks = self._generate_data()

    def _generate_data(self):
        images = []
        masks = []
        for _ in range(self.num_samples):
            img = np.zeros((self.img_size, self.img_size), dtype=np.float32)
            mask = np.zeros((self.img_size, self.img_size), dtype=np.float32)
            
            # Add a random circle
            cx, cy = np.random.randint(20, 44, size=2)
            r = np.random.randint(10, 18)
            y, x = np.ogrid[-cy:self.img_size-cy, -cx:self.img_size-cx]
            mask_region = x*x + y*y <= r*r
            mask[mask_region] = 1.0
            
            # Create ambiguous boundaries with blur and noise
            img_blurred = scipy.ndimage.gaussian_filter(mask, sigma=3.0)
            noise = np.random.normal(0, 0.4, (self.img_size, self.img_size))
            img_final = np.clip(img_blurred + noise, 0, 1)

            images.append(img_final[np.newaxis, ...])
            masks.append(mask[np.newaxis, ...])

        return torch.tensor(np.array(images, dtype=np.float32)), torch.tensor(np.array(masks, dtype=np.float32))

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.images[idx], self.masks[idx]

# ==========================================
# 2. Models
# ==========================================
class SimpleUNet(nn.Module):
    """A tiny CNN for segmentation."""
    def __init__(self):
        super().__init__()
        self.enc1 = nn.Conv2d(1, 16, 3, padding=1)
        self.enc2 = nn.Conv2d(16, 32, 3, padding=1, stride=2)
        self.dec1 = nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1)
        self.dec2 = nn.Conv2d(16, 1, 3, padding=1)

    def forward(self, x):
        e1 = F.relu(self.enc1(x))
        e2 = F.relu(self.enc2(e1))
        d1 = F.relu(self.dec1(e2))
        logits = self.dec2(d1)
        prob = torch.sigmoid(logits)
        return prob, e1 # return low-level features for the rejector

class Rejector(nn.Module):
    """Predicts pixel-wise acceptance score in [0, 1]."""
    def __init__(self):
        super().__init__()
        # Input features: backbone features (16) + prob (1) + entropy (1) = 18
        self.conv1 = nn.Conv2d(18, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 1, 3, padding=1)

    def forward(self, features, prob):
        # Compute entropy
        entropy = - (prob * torch.log(prob + 1e-8) + (1 - prob) * torch.log(1 - prob + 1e-8))
        x = torch.cat([features, prob, entropy], dim=1)
        x = F.relu(self.conv1(x))
        acc_score = torch.sigmoid(self.conv2(x))
        return acc_score

# ==========================================
# 3. Training
# ==========================================
def create_pseudo_labels(prob, target):
    """
    Creates pseudo-labels for rejector training.
    1 (safe): correct prediction, high confidence
    0 (unsafe): incorrect prediction or high uncertainty
    -1 (ignore): otherwise
    """
    with torch.no_grad():
        pred = (prob >= 0.5).float()
        correct = (pred == target)
        
        # High confidence: prob > 0.8 or prob < 0.2
        high_conf = (prob > 0.8) | (prob < 0.2)
        
        unsafe = (~correct) | (~high_conf)
        safe = correct & high_conf
        
        pseudo_labels = torch.full_like(prob, -1)
        pseudo_labels[unsafe] = 0.0
        pseudo_labels[safe] = 1.0
        
    return pseudo_labels

def main():
    print("Generating synthetic datasets...")
    train_ds = SyntheticShapesDataset(500)
    val_ds = SyntheticShapesDataset(100)
    cal_ds = SyntheticShapesDataset(150)
    test_ds = SyntheticShapesDataset(100)

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)
    cal_loader = DataLoader(cal_ds, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False) # batch=1 for visualization

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    backbone = SimpleUNet().to(device)
    rejector = Rejector().to(device)

    opt_backbone = optim.AdamW(backbone.parameters(), lr=1e-3)
    opt_rejector = optim.AdamW(rejector.parameters(), lr=1e-3)

    print("\n--- Training Backbone ---")
    backbone.train()
    for epoch in range(20):
        total_loss = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt_backbone.zero_grad()
            prob, _ = backbone(x)
            loss = F.binary_cross_entropy(prob, y)
            loss.backward()
            opt_backbone.step()
            total_loss += loss.item()
        if (epoch+1) % 5 == 0:
            print(f"Epoch {epoch+1}/20, Loss: {total_loss/len(train_loader):.4f}")

    print("\n--- Training Rejector ---")
    backbone.eval()
    rejector.train()
    for epoch in range(20):
        total_loss = 0
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            with torch.no_grad():
                prob, features = backbone(x)
                pseudo_labels = create_pseudo_labels(prob, y)
            
            opt_rejector.zero_grad()
            acc_score = rejector(features, prob)
            
            # Masked BCE Loss
            mask = (pseudo_labels != -1).float()
            if mask.sum() > 0:
                loss = F.binary_cross_entropy(acc_score, pseudo_labels.clamp(0, 1), reduction='none')
                loss = (loss * mask).sum() / mask.sum()
                loss.backward()
                opt_rejector.step()
                total_loss += loss.item()
        if (epoch+1) % 5 == 0:
            print(f"Epoch {epoch+1}/20, Loss: {total_loss/len(val_loader):.4f}")

    print("\n--- Conformal Calibration (Split CRC) ---")
    rejector.eval()
    
    alpha = 0.05 # Target risk: 5% unweighted accepted false negative rate
    thresholds = np.linspace(0.01, 0.99, 100)
    
    # We calibrate the threshold tau over the calibration set
    cal_risks = {tau: [] for tau in thresholds}
    cal_coverages = {tau: [] for tau in thresholds}
    
    for x, y in cal_loader:
        x, y = x.to(device), y.to(device)
        with torch.no_grad():
            prob, features = backbone(x)
            acc_score = rejector(features, prob)
            pred = (prob >= 0.5).float()
            
            # shape: [B, 1, H, W]
            for tau in thresholds:
                A = (acc_score >= tau).float()
                
                # Localized selective miss risk: 
                # (Weighted Accepted FNR). Here we use unweighted w_u = 1.
                # Numerator: error on accepted foreground
                numerator = (y * A * (1 - pred)).sum(dim=(1, 2, 3))
                # Denominator: total foreground + epsilon
                denominator = y.sum(dim=(1, 2, 3)) + 1e-5
                
                risk = numerator / denominator
                coverage = A.mean(dim=(1, 2, 3))
                
                cal_risks[tau].extend(risk.cpu().numpy())
                cal_coverages[tau].extend(coverage.cpu().numpy())
                
    optimal_tau = None
    max_coverage = -1
    
    # Calculate finite sample correction (CRC style)
    n = len(cal_ds)
    # empirical risk bound: R_hat(tau) + delta <= alpha
    # For bounded loss in [0, 1] and valid calibration, we want R_hat <= alpha - delta_n
    # Standard CRC says expected risk <= expected empirical risk + 1/n approx.
    # We will just pick the smallest threshold where empirical risk <= alpha for this PoC.
    target = alpha - (1.0 / n) # simple correction
    
    for tau in thresholds:
        mean_risk = np.mean(cal_risks[tau])
        mean_cov = np.mean(cal_coverages[tau])
        if mean_risk <= target:
            if mean_cov > max_coverage:
                max_coverage = mean_cov
                optimal_tau = tau
                
    if optimal_tau is None:
        print("Could not satisfy the risk target! Using default tau=0.5")
        optimal_tau = 0.5
    else:
        print(f"Calibrated tau*: {optimal_tau:.3f} | Expected Risk: {np.mean(cal_risks[optimal_tau]):.4f} <= {alpha} | Coverage: {max_coverage:.4f}")


    print("\n--- Evaluation & Visualization on Test Set ---")
    # Take ONE sample to plot
    x, y = test_loader.dataset[0]
    x = x.unsqueeze(0).to(device)
    y = y.unsqueeze(0).to(device)
    
    with torch.no_grad():
        prob, features = backbone(x)
        acc_score = rejector(features, prob)
        pred = (prob >= 0.5).float()
        
        A = (acc_score >= optimal_tau).float()
        selective_pred = pred.clone()
        selective_pred[A == 0] = 0.5 # Represent abstention as gray (0.5)

    x_np = x[0, 0].cpu().numpy()
    y_np = y[0, 0].cpu().numpy()
    pred_np = pred[0, 0].cpu().numpy()
    acc_np = acc_score[0, 0].cpu().numpy()
    A_np = A[0, 0].cpu().numpy()
    sel_pred_np = selective_pred[0, 0].cpu().numpy()

    plt.figure(figsize=(15, 6))
    plt.subplot(1, 6, 1)
    plt.title("Input image")
    plt.imshow(x_np, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 6, 2)
    plt.title("Ground Truth")
    plt.imshow(y_np, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 6, 3)
    plt.title("Prediction")
    plt.imshow(pred_np, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 6, 4)
    plt.title("Acceptance Score")
    plt.imshow(acc_np, cmap='viridis', vmin=0, vmax=1)
    plt.axis('off')

    plt.subplot(1, 6, 5)
    plt.title(f"Acceptance Mask\n(tau={optimal_tau:.2f})")
    plt.imshow(A_np, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 6, 6)
    plt.title("Selective Pred\n(Gray=Abstain)")
    plt.imshow(sel_pred_np, cmap='gray', vmin=0, vmax=1)
    plt.axis('off')

    os.makedirs(os.path.dirname(os.path.abspath(__file__)), exist_ok=True)
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ls_crc_poc_result.png")
    plt.tight_layout()
    plt.savefig(out_path)
    print(f"Saved visualization to {out_path}")

if __name__ == "__main__":
    main()
