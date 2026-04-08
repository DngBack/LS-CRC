import argparse
import copy
import os
import random

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from data.dataset import get_dataloader
from models.backbone import get_backbone
from models.rejector import Rejector
from utils.pseudo_labels import generate_pseudo_labels
from utils.losses import compute_spatial_weight_map, localized_surrogate_risk, smoothness_loss


def set_seed(seed: int):
    """Best-effort reproducibility for training (GPU still has minor nondeterminism on some ops)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _dice_score(prob, y, eps=1e-5):
    intersection = (prob * y).sum((1, 2, 3))
    denom = prob.sum((1, 2, 3)) + y.sum((1, 2, 3)) + eps
    return (2.0 * intersection / denom).mean()


@torch.no_grad()
def validate_backbone(backbone, val_loader, device):
    backbone.eval()
    total_loss = 0.0
    total_dice = 0.0
    n_batches = 0
    for batch in val_loader:
        x, y = batch[0].to(device), batch[1].to(device)
        logits, prob, _ = backbone(x)
        loss_bce = F.binary_cross_entropy(prob, y)
        loss_dice = 1.0 - _dice_score(prob, y)
        total_loss += (loss_bce + loss_dice).item()
        total_dice += _dice_score(prob, y).item()
        n_batches += 1
    if n_batches == 0:
        return 0.0, float("inf")
    return total_dice / n_batches, total_loss / n_batches


@torch.no_grad()
def validate_rejector(backbone, rejector, val_loader, device, lambda_smooth):
    backbone.eval()
    rejector.eval()
    total_loss = 0.0
    n_batches = 0
    for batch in val_loader:
        x, y = batch[0].to(device), batch[1].to(device)
        _, prob, features = backbone(x)
        pseudo_labels = generate_pseudo_labels(prob, y)
        acc_score = rejector(features, prob)
        mask = (pseudo_labels != -1.0).float()
        if mask.sum() == 0:
            continue
        loss_bce = F.binary_cross_entropy(acc_score, pseudo_labels.clamp(0, 1), reduction="none")
        loss_bce = (loss_bce * mask).sum() / mask.sum()
        loss_smooth = smoothness_loss(acc_score)
        total_loss += (loss_bce + lambda_smooth * loss_smooth).item()
        n_batches += 1
    if n_batches == 0:
        return float("inf")
    return total_loss / n_batches


@torch.no_grad()
def validate_joint(backbone, rejector, val_loader, device, lambda_smooth, lambda_surrogate):
    backbone.eval()
    rejector.eval()
    total_loss = 0.0
    n_batches = 0
    for batch in val_loader:
        x, y = batch[0].to(device), batch[1].to(device)
        _, prob, features = backbone(x)
        acc_score = rejector(features, prob)
        w_u = compute_spatial_weight_map(y, bnd_weight=2.0)
        loss_surrogate = localized_surrogate_risk(y, prob, acc_score, w_u)
        loss_bce = F.binary_cross_entropy(prob, y)
        loss_smooth = smoothness_loss(acc_score)
        total_loss += (loss_bce + lambda_surrogate * loss_surrogate + lambda_smooth * loss_smooth).item()
        n_batches += 1
    if n_batches == 0:
        return float("inf")
    return total_loss / n_batches


def _clone_state_dict(module):
    return copy.deepcopy(module.state_dict())


def train_backbone(backbone, train_loader, val_loader, device, epochs, lr, val_loader_nonempty):
    print("--- Training Backbone ---")
    optimizer = optim.AdamW(backbone.parameters(), lr=lr)
    best_dice = -1.0
    best_state = None

    for epoch in range(epochs):
        backbone.train()
        total_loss = 0.0
        for x, y, *_ in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits, prob, _ = backbone(x)
            loss_bce = F.binary_cross_entropy(prob, y)
            loss_dice = 1.0 - _dice_score(prob, y)
            loss = loss_bce + loss_dice
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        msg = f"Epoch {epoch + 1}/{epochs} | Backbone Loss: {total_loss / len(train_loader):.4f}"
        if val_loader_nonempty:
            val_dice, val_loss = validate_backbone(backbone, val_loader, device)
            msg += f" | Val Dice: {val_dice:.4f} | Val Loss: {val_loss:.4f}"
            if val_dice > best_dice:
                best_dice = val_dice
                best_state = _clone_state_dict(backbone)
        print(msg)

    if val_loader_nonempty and best_state is not None:
        backbone.load_state_dict(best_state)
        print(f"Restored best backbone (Val Dice: {best_dice:.4f}).")

    return backbone


def train_rejector(backbone, rejector, train_loader, val_loader, device, epochs, lr, lambda_smooth, val_loader_nonempty):
    print("--- Training Rejector ---")
    optimizer = optim.AdamW(rejector.parameters(), lr=lr)
    backbone.eval()
    best_val = float("inf")
    best_state = None

    for epoch in range(epochs):
        rejector.train()
        total_loss = 0.0
        for x, y, *_ in train_loader:
            x, y = x.to(device), y.to(device)
            with torch.no_grad():
                _, prob, features = backbone(x)
                pseudo_labels = generate_pseudo_labels(prob, y)

            optimizer.zero_grad()
            acc_score = rejector(features, prob)
            mask = (pseudo_labels != -1.0).float()
            if mask.sum() > 0:
                loss_bce = F.binary_cross_entropy(acc_score, pseudo_labels.clamp(0, 1), reduction="none")
                loss_bce = (loss_bce * mask).sum() / mask.sum()
                loss_smooth = smoothness_loss(acc_score)
                loss = loss_bce + lambda_smooth * loss_smooth
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

        msg = f"Epoch {epoch + 1}/{epochs} | Rejector Loss: {total_loss / len(train_loader):.4f}"
        if val_loader_nonempty:
            v = validate_rejector(backbone, rejector, val_loader, device, lambda_smooth)
            msg += f" | Val Loss: {v:.4f}"
            if v < best_val:
                best_val = v
                best_state = _clone_state_dict(rejector)
        print(msg)

    if val_loader_nonempty and best_state is not None:
        rejector.load_state_dict(best_state)
        print(f"Restored best rejector (Val Loss: {best_val:.4f}).")

    return rejector


def joint_finetune(
    backbone,
    rejector,
    train_loader,
    val_loader,
    device,
    epochs,
    lr,
    lambda_smooth,
    lambda_surrogate,
    val_loader_nonempty,
):
    print("--- Joint Fine-Tuning ---")
    optimizer = optim.AdamW(list(backbone.parameters()) + list(rejector.parameters()), lr=lr)
    best_val = float("inf")
    best_backbone = None
    best_rejector = None

    for epoch in range(epochs):
        backbone.train()
        rejector.train()
        total_loss = 0.0
        for x, y, *_ in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits, prob, features = backbone(x)
            acc_score = rejector(features, prob)
            w_u = compute_spatial_weight_map(y, bnd_weight=2.0)
            loss_surrogate = localized_surrogate_risk(y, prob, acc_score, w_u)
            loss_bce = F.binary_cross_entropy(prob, y)
            loss_smooth = smoothness_loss(acc_score)
            loss = loss_bce + lambda_surrogate * loss_surrogate + lambda_smooth * loss_smooth
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        msg = f"Epoch {epoch + 1}/{epochs} | Fine-tune Loss: {total_loss / len(train_loader):.4f}"
        if val_loader_nonempty:
            v = validate_joint(backbone, rejector, val_loader, device, lambda_smooth, lambda_surrogate)
            msg += f" | Val Loss: {v:.4f}"
            if v < best_val:
                best_val = v
                best_backbone = _clone_state_dict(backbone)
                best_rejector = _clone_state_dict(rejector)
        print(msg)

    if val_loader_nonempty and best_backbone is not None:
        backbone.load_state_dict(best_backbone)
        rejector.load_state_dict(best_rejector)
        print(f"Restored best joint checkpoint (Val Loss: {best_val:.4f}).")

    return backbone, rejector


def parse_args():
    p = argparse.ArgumentParser(description="LS-CRC three-stage training (backbone, rejector, joint).")
    p.add_argument("--data-root", type=str, default="data/Kvasir-SEG", help="Dataset root with images/, masks/, splits.")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--epochs-backbone", type=int, default=100)
    p.add_argument("--epochs-rejector", type=int, default=50)
    p.add_argument("--epochs-joint", type=int, default=50)
    p.add_argument("--lr-backbone", type=float, default=1e-4)
    p.add_argument("--lr-rejector", type=float, default=1e-4)
    p.add_argument("--lr-joint", type=float, default=5e-5)
    p.add_argument(
        "--lambda-smooth",
        type=float,
        default=0.5,
        help="Weight for smoothness loss (lambda_2); used in rejector and joint stages.",
    )
    p.add_argument(
        "--lambda-surrogate",
        type=float,
        default=1.0,
        help="Weight for localized surrogate risk (lambda_3) in joint fine-tuning.",
    )
    p.add_argument("--backbone", type=str, default="unet", choices=["unet", "deeplabv3plus"])
    p.add_argument("--encoder-name", type=str, default="resnet50")
    p.add_argument(
        "--encoder-weights",
        type=str,
        default="imagenet",
        help="Use 'imagenet' for pre-trained encoder, or 'none' for random init.",
    )
    p.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    p.add_argument(
        "--resume-backbone",
        type=str,
        default=None,
        help="Load backbone state_dict before training (e.g. fine-tune rejector on another domain).",
    )
    p.add_argument(
        "--resume-rejector",
        type=str,
        default=None,
        help="Load rejector weights before rejector/joint stages (same feature_channels as backbone).",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=None,
        help="If set, fixes Python/NumPy/torch RNG and train-loader shuffle for repeatable runs.",
    )
    return p.parse_args()


def main():
    args = parse_args()
    if args.seed is not None:
        set_seed(args.seed)
        print(f"Using random seed: {args.seed}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    encoder_weights = None if args.encoder_weights.lower() in ("none", "null", "") else args.encoder_weights

    train_loader = get_dataloader(
        args.data_root,
        split="train",
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
    )
    val_loader = get_dataloader(
        args.data_root, split="val", batch_size=args.batch_size, num_workers=args.num_workers
    )
    val_nonempty = len(val_loader) > 0

    backbone = get_backbone(
        model_name=args.backbone,
        encoder_name=args.encoder_name,
        encoder_weights=encoder_weights,
    ).to(device)
    rejector = Rejector(feature_channels=backbone.feature_channels).to(device)

    if args.resume_backbone:
        backbone.load_state_dict(torch.load(args.resume_backbone, map_location=device))
        print(f"Loaded backbone from {args.resume_backbone}")
    if args.resume_rejector:
        rejector.load_state_dict(torch.load(args.resume_rejector, map_location=device))
        print(f"Loaded rejector from {args.resume_rejector}")

    if args.epochs_backbone > 0:
        backbone = train_backbone(
            backbone, train_loader, val_loader, device, args.epochs_backbone, args.lr_backbone, val_nonempty
        )
    else:
        print("Skipping backbone training (--epochs-backbone 0).")

    if args.epochs_rejector > 0:
        rejector = train_rejector(
            backbone,
            rejector,
            train_loader,
            val_loader,
            device,
            args.epochs_rejector,
            args.lr_rejector,
            args.lambda_smooth,
            val_nonempty,
        )
    else:
        print("Skipping rejector training (--epochs-rejector 0).")

    if args.epochs_joint > 0:
        backbone, rejector = joint_finetune(
            backbone,
            rejector,
            train_loader,
            val_loader,
            device,
            args.epochs_joint,
            args.lr_joint,
            args.lambda_smooth,
            args.lambda_surrogate,
            val_nonempty,
        )
    else:
        print("Skipping joint fine-tuning (--epochs-joint 0).")

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    torch.save(backbone.state_dict(), os.path.join(args.checkpoint_dir, "backbone.pth"))
    torch.save(rejector.state_dict(), os.path.join(args.checkpoint_dir, "rejector.pth"))
    print(f"Saved models to {args.checkpoint_dir}/")


if __name__ == "__main__":
    main()
