import os
import torch
import torch.nn.functional as F
import torch.optim as optim
from data.dataset import get_dataloader
from models.backbone import get_backbone
from models.rejector import Rejector
from utils.pseudo_labels import generate_pseudo_labels
from utils.losses import compute_spatial_weight_map, localized_surrogate_risk, smoothness_loss

def train_backbone(backbone, train_loader, val_loader, device, epochs=1):
    """Stage 1: Train the segmentation backbone."""
    print("--- Training Backbone ---")
    optimizer = optim.AdamW(backbone.parameters(), lr=3e-4)
    backbone.train()
    
    for epoch in range(epochs):
        total_loss = 0
        for x, y, tags, _ in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits, prob, _ = backbone(x)
            
            # Simple BCE + Dice
            loss_bce = F.binary_cross_entropy(prob, y)
            intersection = (prob * y).sum((1,2,3))
            dice = 2.0 * intersection / (prob.sum((1,2,3)) + y.sum((1,2,3)) + 1e-5)
            loss_dice = 1.0 - dice.mean()
            loss = loss_bce + loss_dice
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        print(f"Epoch {epoch+1}/{epochs} | Backbone Loss: {total_loss/len(train_loader):.4f}")
    return backbone

def train_rejector(backbone, rejector, train_loader, val_loader, device, epochs=1):
    """Stage 2: Train the rejector head using pseudo-labels on val_loader or train_loader."""
    print("--- Training Rejector ---")
    optimizer = optim.AdamW(rejector.parameters(), lr=3e-4)
    backbone.eval()
    rejector.train()
    
    # We train the rejector on the train_loader or val_loader
    for epoch in range(epochs):
        total_loss = 0
        for x, y, tags, _ in train_loader:
            x, y = x.to(device), y.to(device)
            
            with torch.no_grad():
                _, prob, features = backbone(x)
                pseudo_labels = generate_pseudo_labels(prob, y)
                
            optimizer.zero_grad()
            acc_score = rejector(features, prob)
            
            mask = (pseudo_labels != -1.0).float()
            if mask.sum() > 0:
                loss_bce = F.binary_cross_entropy(acc_score, pseudo_labels.clamp(0, 1), reduction='none')
                loss_bce = (loss_bce * mask).sum() / mask.sum()
                
                loss_smooth = smoothness_loss(acc_score)
                loss = loss_bce + 0.1 * loss_smooth
                
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                
        print(f"Epoch {epoch+1}/{epochs} | Rejector Loss: {total_loss/len(train_loader):.4f}")
    return rejector

def joint_finetune(backbone, rejector, train_loader, device, epochs=1):
    """Stage 3: Joint fine-tuning with surrogate localized risk."""
    print("--- Joint Fine-Tuning ---")
    optimizer = optim.AdamW(list(backbone.parameters()) + list(rejector.parameters()), lr=1e-4)
    backbone.train()
    rejector.train()
    
    for epoch in range(epochs):
        total_loss = 0
        for x, y, tags, _ in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            
            logits, prob, features = backbone(x)
            acc_score = rejector(features, prob)
            
            w_u = compute_spatial_weight_map(y, bnd_weight=2.0)
            
            loss_surrogate = localized_surrogate_risk(y, prob, acc_score, w_u)
            loss_bce = F.binary_cross_entropy(prob, y)
            loss_smooth = smoothness_loss(acc_score)
            
            loss = loss_bce + 0.5 * loss_surrogate + 0.1 * loss_smooth
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        print(f"Epoch {epoch+1}/{epochs} | Fine-tune Loss: {total_loss/len(train_loader):.4f}")
    return backbone, rejector

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using device:", device)
    
    # In a real run, this would be a large number of epochs. 
    # For testing the codebase execution, we use small loops.
    train_loader = get_dataloader('data/Kvasir-SEG', split='train')
    val_loader = get_dataloader('data/Kvasir-SEG', split='val')
    
    backbone = get_backbone('unet').to(device)
    rejector = Rejector(feature_channels=32).to(device)
    
    # Ablation logic
    # Variant A: Backbone only (trained)
    # Variant D: Backbone + Rejector (with smoothness)
    # Variant E: Backbone + Rejector + Joint Fine-tuning
    
    # Execute full pipeline for the sake of completeness
    backbone = train_backbone(backbone, train_loader, val_loader, device, epochs=15)
    rejector = train_rejector(backbone, rejector, train_loader, val_loader, device, epochs=15)
    backbone, rejector = joint_finetune(backbone, rejector, train_loader, device, epochs=10)
    
    os.makedirs("checkpoints", exist_ok=True)
    torch.save(backbone.state_dict(), "checkpoints/backbone.pth")
    torch.save(rejector.state_dict(), "checkpoints/rejector.pth")
    print("Saved models locally.")

if __name__ == "__main__":
    main()
