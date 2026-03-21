import argparse
import os

import numpy as np
import pandas as pd
import torch
from calibrate import calibrate_threshold
from data.dataset import get_dataloader
from models.backbone import get_backbone
from models.rejector import Rejector
from utils.losses import compute_spatial_weight_map, get_localized_selective_loss_components
from utils.metrics import compute_dice_iou, cvar_risk, worst_percentile_risk


def evaluate_model(backbone, rejector, dataloader, device, tau, method="lscrc"):
    """
    Evaluates all metrics on the test set for a given calibration threshold tau.
    """
    backbone.eval()
    if rejector:
        rejector.eval()

    all_risks = []
    all_coverages = []
    all_dice = []
    all_iou = []

    subgroup_risks = {"size": {0: [], 1: [], 2: []}, "complexity": {0: [], 1: []}}

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
                max_entropy = 0.693
                norm_entropy = entropy / max_entropy
                A_u = ((1.0 - norm_entropy) >= tau).float()
            elif method == "max_softmax":
                max_prob = torch.max(prob, 1 - prob)
                A_u = (max_prob >= tau).float()
            else:
                A_u = torch.ones_like(prob)

            dice, iou = compute_dice_iou(y, pred)
            all_dice.append(dice.item())
            all_iou.append(iou.item())

            w_u = compute_spatial_weight_map(y, bnd_weight=2.0)
            numerators, denominators = get_localized_selective_loss_components(y, pred, A_u, w_u)

            risk = numerators / denominators
            coverage = A_u.view(A_u.size(0), -1).mean(dim=1)

            for i in range(x.size(0)):
                r_val = risk[i].item()
                c_val = coverage[i].item()
                size_tag = tags["size"][i].item()
                comp_tag = tags["complexity"][i].item()
                all_risks.append(r_val)
                all_coverages.append(c_val)
                subgroup_risks["size"][size_tag].append(r_val)
                subgroup_risks["complexity"][comp_tag].append(r_val)

    metrics = {
        "Dice": np.mean(all_dice),
        "IoU": np.mean(all_iou),
        "Coverage": np.mean(all_coverages),
        "Risk_Mean": np.mean(all_risks),
        "Risk_Std": np.std(all_risks),
        "Worst_10%_Risk": worst_percentile_risk(all_risks, q=0.9),
        "CVaR_0.9": cvar_risk(all_risks, alpha=0.9),
        "Subgroups": subgroup_risks,
    }
    return metrics


def _parse_scenario(spec: str):
    """
    Format: label,cal_root,test_root
    Example: Cross-Kvasir-Cal-CVC-Test,data/Kvasir-SEG,data/CVC-ClinicDB
    """
    parts = [p.strip() for p in spec.split(",")]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(
            f"Scenario must be 'label,cal_root,test_root' (3 comma-separated fields); got {spec!r}"
        )
    return parts[0], parts[1], parts[2]


def _run_one_scenario(
    backbone,
    rejector,
    device,
    cal_root,
    test_root,
    batch_size,
    num_workers,
    alpha,
    experiments,
):
    cal_loader = get_dataloader(cal_root, split="cal", batch_size=batch_size, num_workers=num_workers)
    test_loader = get_dataloader(test_root, split="test", batch_size=batch_size, num_workers=num_workers)
    results = []

    print(f"\nCalibration: {cal_root} | Test: {test_root}")
    for exp in experiments:
        name = exp["name"]
        method = exp["method"]
        if exp["requires_cal"]:
            tau_star, _, _ = calibrate_threshold(
                backbone, rejector if method == "lscrc" else None, cal_loader, device, alpha, method=method
            )
            print(f"[{name}] Calibrated tau* = {tau_star:.3f}")
        else:
            tau_star = 0.0
            print(f"[{name}] Fixed tau* = 0.0")

        metrics = evaluate_model(
            backbone, rejector if method == "lscrc" else None, test_loader, device, tau_star, method=method
        )

        size_risks = [
            np.mean(metrics["Subgroups"]["size"][k]) if len(metrics["Subgroups"]["size"][k]) > 0 else 0
            for k in range(3)
        ]
        comp_risks = [
            np.mean(metrics["Subgroups"]["complexity"][k]) if len(metrics["Subgroups"]["complexity"][k]) > 0 else 0
            for k in range(2)
        ]
        worst_group = max(max(size_risks), max(comp_risks))

        results.append(
            {
                "Method": name,
                "Dice": metrics["Dice"],
                "Coverage": metrics["Coverage"],
                "Expected Risk": metrics["Risk_Mean"],
                "Risk Std": metrics["Risk_Std"],
                "Worst 10%": metrics["Worst_10%_Risk"],
                "CVaR_0.9": metrics["CVaR_0.9"],
                "Worst Group": worst_group,
            }
        )
    return pd.DataFrame(results)


def parse_args():
    p = argparse.ArgumentParser(
        description="Evaluate selective segmentation with optional cross-dataset calibration."
    )
    p.add_argument(
        "--scenario",
        action="append",
        type=_parse_scenario,
        metavar="LABEL,CAL_ROOT,TEST_ROOT",
        help=(
            "Repeatable. Calibrate on CAL_ROOT (split=cal), evaluate on TEST_ROOT (split=test). "
            "Default if omitted: CVC-ClinicDB,data/CVC-ClinicDB,data/CVC-ClinicDB"
        ),
    )
    p.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--alpha", type=float, default=0.05, help="Risk budget for calibration.")
    p.add_argument("--backbone", type=str, default="unet", choices=["unet", "deeplabv3plus"])
    p.add_argument("--encoder-name", type=str, default="resnet50")
    p.add_argument(
        "--encoder-weights",
        type=str,
        default="imagenet",
        help="Must match training (e.g. imagenet or none).",
    )
    p.add_argument(
        "--results-csv",
        type=str,
        default=None,
        help="If set, append all scenario tables with a Dataset column.",
    )
    return p.parse_args()


def run_experiments(args=None):
    if args is None:
        args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder_weights = None if args.encoder_weights.lower() in ("none", "null", "") else args.encoder_weights

    scenarios = args.scenario
    if not scenarios:
        scenarios = [("CVC-ClinicDB", "data/CVC-ClinicDB", "data/CVC-ClinicDB")]

    experiments = [
        {"name": "Plain Segmentation", "method": "plain", "requires_cal": False},
        {"name": "Entropy Threshold", "method": "entropy", "requires_cal": True},
        {"name": "Max-Softmax Threshold", "method": "max_softmax", "requires_cal": True},
        {"name": "LS-CRC (Ours)", "method": "lscrc", "requires_cal": True},
    ]

    print("Loading models...")
    backbone = get_backbone(
        model_name=args.backbone,
        encoder_name=args.encoder_name,
        encoder_weights=encoder_weights,
    ).to(device)
    rejector = Rejector(feature_channels=backbone.feature_channels).to(device)

    ckpt_b = os.path.join(args.checkpoint_dir, "backbone.pth")
    ckpt_r = os.path.join(args.checkpoint_dir, "rejector.pth")
    try:
        backbone.load_state_dict(torch.load(ckpt_b, map_location=device))
        rejector.load_state_dict(torch.load(ckpt_r, map_location=device))
        print("Loaded trained weights.")
    except Exception as e:
        print(f"Warning: Could not load weights ({e}). Using fresh initialization. Run train.py first.")

    all_frames = []
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 1000)

    for label, cal_root, test_root in scenarios:
        print(f"\n================== TABLE: {label} ==================")
        df = _run_one_scenario(
            backbone,
            rejector,
            device,
            cal_root,
            test_root,
            args.batch_size,
            args.num_workers,
            args.alpha,
            experiments,
        )
        out = df.copy()
        out.insert(0, "Dataset", label)
        all_frames.append(out)
        print(df.to_string(index=False))

    if args.results_csv and all_frames:
        combined = pd.concat(all_frames, ignore_index=True)
        combined.to_csv(args.results_csv, index=False)
        print(f"\nWrote combined results to {args.results_csv}")


if __name__ == "__main__":
    run_experiments()
