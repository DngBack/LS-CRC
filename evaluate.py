import torch
import numpy as np
import pandas as pd
from data.dataset import get_dataloader
from models.backbone import get_backbone
from models.rejector import Rejector
from calibrate import calibrate_threshold, evaluate_risk
from utils.metrics import compute_dice_iou, compute_selective_metrics, cvar_risk, worst_percentile_risk
from utils.losses import compute_spatial_weight_map, get_localized_selective_loss_components

def evaluate_model(backbone, rejector, dataloader, device, tau, method='lscrc'):
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
    
    # Subgroup tracking
    # Tags: 'size' in [0, 1, 2], 'complexity' in [0, 1]
    subgroup_risks = {'size': {0:[], 1:[], 2:[]}, 'complexity': {0:[], 1:[]}}
    
    with torch.no_grad():
        for x, y, tags, _ in dataloader:
            x, y = x.to(device), y.to(device)
            _, prob, features = backbone(x)
            pred = (prob >= 0.5).float()
            
            p_safe = torch.clamp(prob, 1e-7, 1 - 1e-7)
            entropy = - (p_safe * torch.log(p_safe) + (1 - p_safe) * torch.log(1 - p_safe))
            
            # Acceptance logic
            if method == 'lscrc':
                scores = rejector(features, prob)
                A_u = (scores >= tau).float()
            elif method == 'entropy':
                max_entropy = 0.693 
                norm_entropy = entropy / max_entropy
                A_u = ((1.0 - norm_entropy) >= tau).float()
            elif method == 'max_softmax':
                max_prob = torch.max(prob, 1 - prob)
                A_u = (max_prob >= tau).float()
            else:
                A_u = torch.ones_like(prob)

            # Standard Metrics
            dice, iou = compute_dice_iou(y, pred)
            all_dice.append(dice.item())
            all_iou.append(iou.item())
            
            # Localized Risk computation
            w_u = compute_spatial_weight_map(y, bnd_weight=2.0)
            numerators, denominators = get_localized_selective_loss_components(y, pred, A_u, w_u)
            
            risk = numerators / denominators
            coverage = A_u.view(A_u.size(0), -1).mean(dim=1)
            
            # Subgroup bookkeeping (assuming batch=1 or scalar tag format matches length)
            # For simplicity, if batch size > 1, we zip through it
            for i in range(x.size(0)):
                r_val = risk[i].item()
                c_val = coverage[i].item()
                
                size_tag = tags['size'][i].item()
                comp_tag = tags['complexity'][i].item()
                
                all_risks.append(r_val)
                all_coverages.append(c_val)
                
                subgroup_risks['size'][size_tag].append(r_val)
                subgroup_risks['complexity'][comp_tag].append(r_val)

    # Compute aggregate metrics
    metrics = {
        'Dice': np.mean(all_dice),
        'IoU': np.mean(all_iou),
        'Coverage': np.mean(all_coverages),
        'Risk_Mean': np.mean(all_risks),
        'Risk_Std': np.std(all_risks),
        'Worst_10%_Risk': worst_percentile_risk(all_risks, q=0.9),
        'CVaR_0.9': cvar_risk(all_risks, alpha=0.9),
        'Subgroups': subgroup_risks
    }
    return metrics

def run_experiments():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Loading datasets...")
    
    # We use synthetic dataset fallback internally so this script always runs.
    cal_loader = get_dataloader('data/Kvasir-SEG', split='cal')
    test_loader = get_dataloader('data/Kvasir-SEG', split='test')
    
    # Load Models
    print("Loading Models...")
    backbone = get_backbone('unet').to(device)
    rejector = Rejector(feature_channels=32).to(device)
    
    try:
        backbone.load_state_dict(torch.load("checkpoints/backbone.pth", map_location=device))
        rejector.load_state_dict(torch.load("checkpoints/rejector.pth", map_location=device))
        print("Loaded trained weights.")
    except Exception as e:
        print("Warning: Could not load weights. Using untrained initialization. Run train.py first.")
        
    # Baselines
    experiments = [
        {'name': 'Plain Segmentation', 'method': 'plain', 'requires_cal': False},
        {'name': 'Entropy Threshold', 'method': 'entropy', 'requires_cal': True},
        {'name': 'Max-Softmax Threshold', 'method': 'max_softmax', 'requires_cal': True},
        {'name': 'LS-CRC (Ours)', 'method': 'lscrc', 'requires_cal': True}
    ]
    
    alpha = 0.05 # 5% Risk budget
    results = []
    
    print("\nStarting Evaluation & Calibration...")
    for exp in experiments:
        name = exp['name']
        method = exp['method']
        
        if exp['requires_cal']:
            # Calibrate threshold
            tau_star, cal_risk, cal_cov = calibrate_threshold(backbone, rejector if method=='lscrc' else None, cal_loader, device, alpha, method=method)
            print(f"[{name}] Calibrated tau* = {tau_star:.3f}")
        else:
            tau_star = 0.0 # Everything accepted
            print(f"[{name}] Fixed tau* = 0.0")
            
        # Evaluate on Test Set
        metrics = evaluate_model(backbone, rejector if method=='lscrc' else None, test_loader, device, tau_star, method=method)
        
        # Format subgroup max risk
        size_risks = [np.mean(metrics['Subgroups']['size'][k]) if len(metrics['Subgroups']['size'][k]) > 0 else 0 for k in range(3)]
        comp_risks = [np.mean(metrics['Subgroups']['complexity'][k]) if len(metrics['Subgroups']['complexity'][k]) > 0 else 0 for k in range(2)]
        
        worst_group = max(max(size_risks), max(comp_risks))
        
        res_row = {
            'Method': name,
            'Dice': metrics['Dice'],
            'Coverage': metrics['Coverage'],
            'Expected Risk': metrics['Risk_Mean'],
            'Risk Std': metrics['Risk_Std'],
            'Worst 10%': metrics['Worst_10%_Risk'],
            'CVaR_0.9': metrics['CVaR_0.9'],
            'Worst Group': worst_group
        }
        results.append(res_row)

    df = pd.DataFrame(results)
    
    print("\n================== TABLE 1: KVASIR-SEG RESULTS ==================")
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    print(df.to_string(index=False))
    
if __name__ == "__main__":
    run_experiments()
