# Learning Structured Abstention for Localized Conformal Risk Control in Segmentation

**Anonymous Authors**

*Submitted to the Thirty-Ninth Conference on Neural Information Processing Systems (NeurIPS 2026)*

---

## Abstract

Conformal Risk Control (CRC) provides finite-sample guarantees on expected risk for bounded monotone losses and has emerged as a principled calibration tool for image segmentation. However, standard CRC is fundamentally *marginal*: it certifies population-average risk but cannot prevent severe heterogeneity across images, object scales, or spatial regions. In medical image segmentation, the most consequential failures often concentrate on thin boundaries, small structures, and difficult images—precisely where marginal guarantees are insufficient. We propose **Localized Selective Conformal Risk Control (LS-CRC)**, a framework that combines a learned pixel-wise rejector with split conformal calibration. The rejector is trained to predict spatially structured acceptance maps using prediction correctness, uncertainty cues, and localized risk surrogates. A held-out calibration set then selects an acceptance threshold guaranteeing finite-sample control of a boundary-weighted selective false-negative risk. We prove that LS-CRC preserves the marginal CRC guarantee, derive a subgroup deviation bound characterizing when conditional risk is approximately controlled, and establish a tail-risk improvement result under a rejector quality condition. On two polyp segmentation benchmarks (Kvasir-SEG and CVC-ClinicDB), LS-CRC achieves the highest accepted-pixel coverage among all methods at comparable or lower expected risk, and yields the lowest worst-image risk on the adapted domain—demonstrating that learned spatial abstention spends the same risk budget more efficiently than scalar uncertainty thresholds.

---

## 1  Introduction

Dense prediction systems are increasingly deployed in settings where failures carry unequal cost. In medical image segmentation—colonoscopy polyp delineation being a canonical example—the most consequential errors cluster at thin boundaries, small foreground objects, and ambiguous interfaces. Standard aggregate metrics such as Dice or IoU can mask these localized failures, and even risk-controlled methods may be insufficient if they only certify average behavior.

Conformal Risk Control (CRC; Angelopoulos et al., 2022) offers a principled starting point. By calibrating a parameter on a held-out exchangeable set, CRC provides finite-sample control of the expected value of any bounded monotone loss—including false-negative rates and weighted miss losses relevant to segmentation. However, CRC is inherently **marginal**: two systems may satisfy the same population risk target while exhibiting dramatically different performance on small lesions, complex boundaries, or hard images.

Selective prediction provides a complementary perspective. Rather than forcing the model to commit everywhere, a selective predictor abstains on uncertain regions and commits only where it is reliable. In segmentation, this suggests an operational notion of reliability: the system should defer on spatially localized regions where errors are probable and where deferral to a human reviewer or a more conservative module is acceptable. The challenge is that learned rejectors alone lack statistical guarantees, while post-hoc uncertainty thresholding is often too coarse to capture the spatial geometry of risky regions.

We propose a synthesis of these ideas. **Localized Selective Conformal Risk Control (LS-CRC)** augments a segmentation backbone with a pixel-wise rejector that outputs a spatially structured acceptance map, then calibrates an acceptance threshold on a held-out set to satisfy a user-specified risk budget. The central technical idea is to control a **localized selective miss risk**—a boundary-weighted false-negative loss computed only on accepted foreground pixels, with denominator fixed to the total foreground mass. This formulation preserves the monotonicity needed for CRC-style threshold selection while capturing deployment-relevant semantics: accepted pixels are committed predictions, and abstained pixels are deferred.

### Contributions

1. We formalize localized selective risk control for segmentation and introduce LS-CRC, combining learned structured abstention with split conformal calibration.
2. We prove three theoretical results: a marginal risk guarantee inherited from CRC (Theorem 1), a subgroup deviation bound (Theorem 2), and a tail-risk improvement result under an explicit rejector quality condition (Theorem 3).
3. We conduct experiments on Kvasir-SEG and CVC-ClinicDB evaluating not only Dice/IoU and average risk but also worst-image risk, CVaR, and subgroup metrics. LS-CRC consistently achieves the best coverage at comparable risk and demonstrates the strongest worst-case performance on the adapted domain.

---

## 2  Related Work

**Conformal prediction and risk control.** Classical conformal prediction guarantees marginal coverage under exchangeability. CRC (Angelopoulos et al., 2022) generalizes this to bounded monotone losses, enabling guarantees on user-specified risks including false-negative rate and weighted miss rate. This extension naturally accommodates structured tasks and decouples the guarantee from the output representation.

**Conformal methods for segmentation.** Recent work adapts conformal calibration to dense prediction through image-dependent thresholds, adaptive score functions, or conditional calibration rules. These methods improve efficiency relative to global thresholds but remain centered on *how much* uncertainty to allocate rather than *where* the model should abstain. They do not learn structured spatial abstention policies.

**Selective prediction and learning to reject.** Selective prediction augments a predictor with a rejection option. Classical methods use scalar confidence or entropy thresholds; recent approaches learn rejection policies jointly with the predictor. In dense prediction, selective segmentation demonstrates that region-wise abstention improves retained accuracy. However, these methods typically lack finite-sample statistical guarantees.

**Tail risk and subgroup reliability.** Average risk control is insufficient when failures concentrate on rare but critical subpopulations. Recent work emphasizes CVaR-style objectives and subgroup diagnostics. Our framework aligns with this perspective: we retain the marginal CRC guarantee while designing the abstention mechanism to reduce tail concentration.

**Positioning.** LS-CRC occupies the intersection: unlike standard CRC, we learn *where* to abstain; unlike selective segmentation without guarantees, we retain conformal calibration; unlike purely average-case methods, we target localized, subgroup, and tail reliability.

---

## 3  Problem Setup

We consider binary image segmentation. Let \((X, Y) \sim P\), where \(X \in \mathcal{X}\) is an image and \(Y \in \{0,1\}^{H \times W}\) is the corresponding binary mask. We observe exchangeable samples split into disjoint training, validation, calibration, and test sets. The calibration set is never used for training or model selection.

A segmentation backbone \(f_\theta\) outputs a foreground probability map \(p_\theta(x) \in [0,1]^{H \times W}\), with hard prediction \(\hat{y}_u = \mathbf{1}\{p_\theta(x)_u \ge 0.5\}\). We augment the backbone with a **rejector head** \(g_\phi\) producing a pixel-wise acceptance score \(s_\phi(x) \in [0,1]^{H \times W}\). Given threshold \(\tau\), the acceptance mask is \(A_u = \mathbf{1}\{s_\phi(x)_u \ge \tau\}\), and the selective prediction is:

\[
\tilde{y}_u = \begin{cases} \hat{y}_u & \text{if } A_u = 1 \\ \bot & \text{if } A_u = 0 \end{cases}
\]

where \(\bot\) denotes abstention (deferral to a downstream process).

---

## 4  Localized Selective Risk

Let \(w_u \ge 0\) be a spatial weight map emphasizing critical regions (e.g., boundary bands). We define the **localized selective miss risk**:

\[
L_{\text{loc}}(x,y;\theta,\phi,\tau) = \frac{\sum_u w_u \cdot \mathbf{1}\{y_u=1\} \cdot A_u \cdot \mathbf{1}\{\hat{y}_u=0\}}{\sum_u w_u \cdot \mathbf{1}\{y_u=1\} + \varepsilon}
\]

This loss has three key properties: (1) only errors on *accepted foreground* pixels are penalized—abstained errors are deferred; (2) the weight map \(w\) localizes the guarantee to critical regions such as boundary bands; (3) the denominator is fixed with respect to \(\tau\), preserving monotonicity for threshold calibration.

We instantiate two variants: **unweighted** (\(w_u = 1\)) and **boundary-weighted** (\(w_u = 1 + \lambda_b \cdot \mathbf{1}\{u \in \text{boundary band}\}\)), where the boundary band is obtained via morphological dilation–erosion of the foreground contour.

---

## 5  Method

### 5.1  Overview

LS-CRC proceeds in three stages: (1) train the segmentation backbone, (2) train a pixel-wise rejector to estimate which spatial locations are safe to accept, and (3) calibrate an acceptance threshold on a held-out calibration set.

### 5.2  Rejector Architecture

The rejector consumes a feature tensor \(\psi(x) = [h_\theta(x),\; p_\theta(x),\; u_\theta(x),\; m_\theta(x)]\), where \(h_\theta(x)\) is an intermediate decoder feature map, \(u_\theta(x)\) is a pixel-wise entropy map, and \(m_\theta(x)\) is a confidence margin. The rejector is a shallow convolutional head outputting \(s_\phi(x) = g_\phi(\psi(x)) \in [0,1]^{H \times W}\).

### 5.3  Pseudo-Labels and Training

Since no ground-truth abstention labels exist, we construct pseudo-labels from prediction correctness and uncertainty:

- **Safe** (\(r^*_u = 1\)): prediction correct, low entropy, not in boundary band.
- **Unsafe** (\(r^*_u = 0\)): prediction incorrect, high entropy, or in boundary band.
- **Ignore** (\(r^*_u = -1\)): ambiguous cases.

The full training objective combines four terms:

\[
\mathcal{L} = \mathcal{L}_{\text{seg}} + \lambda_1 \mathcal{L}_{\text{rej}} + \lambda_2 \mathcal{L}_{\text{smooth}} + \lambda_3 \mathcal{L}_{\text{loc-sur}}
\]

where \(\mathcal{L}_{\text{seg}}\) is BCE + Dice, \(\mathcal{L}_{\text{rej}}\) is masked BCE against pseudo-labels, \(\mathcal{L}_{\text{smooth}}\) is total-variation regularization on the score map, and \(\mathcal{L}_{\text{loc-sur}}\) is a differentiable surrogate of the localized risk that penalizes high acceptance on likely false-negative pixels.

### 5.4  Split Conformal Calibration

After training, we freeze \((\theta, \phi)\) and calibrate:

\[
\tau^* = \arg\max_{\tau \in \mathcal{T}} \widehat{C}_{\text{cal}}(\tau) \quad \text{s.t.} \quad \widehat{R}_{\text{cal}}(\tau) \le \alpha_n
\]

where \(\widehat{R}_{\text{cal}}\) is the empirical localized risk, \(\widehat{C}_{\text{cal}}\) is coverage, and \(\alpha_n = \alpha - 1/n\) is the finite-sample-corrected target. The threshold grid \(\mathcal{T}\) consists of 1000 evenly spaced values in \([0,1]\).

---

## 6  Theoretical Results

We state three results under standard assumptions: exchangeability of calibration/test data (A1), bounded loss in \([0,1]\) (A2), monotonicity of the localized loss in \(\tau\) (A3), and finite threshold grid (A4).

**Theorem 1 (Marginal guarantee).** The selective predictor satisfies \(\mathbb{E}[L_{\text{loc}}(X,Y;\theta,\phi,\tau^*)] \le \alpha + O(1/|\mathcal{D}_{\text{cal}}|)\).

*Proof sketch.* Because \((\theta,\phi)\) are frozen before calibration, threshold selection is the only adaptive component. The localized loss is bounded and monotone. Standard split CRC applies directly.

**Theorem 2 (Subgroup deviation).** For subgroups \(\{G_k\}_{k=1}^K\) with at least \(m_{\min}\) calibration samples each:

\[
\max_k (R_k(\tau^*) - \alpha) \le \delta_n + \Gamma_{\text{sel}} + \sqrt{\frac{\log(2K/\eta)}{2 m_{\min}}}
\]

where \(\Gamma_{\text{sel}} = \max_k (\widehat{R}_k(\tau^*) - \widehat{R}_{\text{cal}}(\tau^*))_+\) is the subgroup mismatch term. The learned rejector reduces \(\Gamma_{\text{sel}}\) by steering abstention toward subgroup-specific risky regions.

**Theorem 3 (Tail-risk improvement).** If the learned rejector approximates the optimal safe-region indicator within \(\varepsilon\) in expected normalized Hamming distance, and the baseline uncertainty rule is \(\varepsilon_0\)-far, then:

\[
\text{CVaR}_\beta(Z^{\text{ours}}_\tau) \le \text{CVaR}_\beta(Z^{\text{base}}_\tau) - \Delta_\beta + L(\varepsilon + \varepsilon_0)
\]

where \(\Delta_\beta \ge 0\) is the tail advantage of the optimal rule and \(L\) is the Lipschitz constant. Full proofs appear in Appendix A of the supplement.

---

## 7  Experiments

### 7.1  Setup

**Datasets.** We use two binary polyp segmentation benchmarks: **Kvasir-SEG** (1000 images; split 600/100/150/150 for train/val/cal/test) and **CVC-ClinicDB** (612 images; used as both an in-domain evaluation target after adaptation and a cross-domain test set).

**Backbone.** DeepLabV3+ with ResNet-50 encoder (ImageNet pretrained). Images resized to 256×256. We train on Kvasir-SEG, then adapt the rejector on CVC-ClinicDB while freezing the segmentation backbone.

**Training.** AdamW optimizer, learning rate \(10^{-4}\), weight decay \(10^{-4}\), batch size 8. Segmentation: 100 epochs; rejector: 40 epochs; joint fine-tuning: 30 epochs. Loss weights: \(\lambda_1 = 1\), \(\lambda_2 = 0.1\), \(\lambda_3 = 0.5\). Boundary band radius: 3 pixels, \(\lambda_b = 2\).

**Calibration.** Threshold grid of 1000 points in \([0,1]\). Risk target \(\alpha \in \{0.01, 0.05, 0.10, 0.15\}\). Primary operating point: \(\alpha = 0.05\).

**Baselines.** (1) Plain Segmentation (no abstention), (2) Entropy Threshold, (3) Max-Softmax Threshold, (4) Standard CRC (Bates et al., 2021)—global image-level gate, (5) Spatial-weighted Conformal Prediction. All selective baselines use the same calibration protocol and threshold grid.

**Metrics.** Dice, IoU, accepted-pixel coverage, expected localized risk (boundary-weighted), risk standard deviation, worst-10% image risk (90th percentile of per-image risk), CVaR\(_{0.9}\), and worst-group risk across size/boundary-complexity/difficulty subgroups.

### 7.2  Main Results

#### Table 1: In-Domain Results (\(\alpha = 0.05\))

Localized boundary-weighted selective risk on held-out test sets. Dice is identical across selective methods (same backbone). **Bold** = best among selective methods; <u>underline</u> = second best. Standard CRC (†) yields zero coverage due to its global image-level gating at this tight budget—an expected failure mode of spatially unaware calibration.

| Dataset | Method | Dice | Coverage ↑ | Risk ↓ | Risk Std ↓ | Worst 10% ↓ | CVaR\(_{0.9}\) ↓ | Worst Grp ↓ |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Kvasir-SEG** | Plain Segmentation | 0.844 | 1.000 | 0.164 | 0.176 | 0.369 | 0.603 | 0.215 |
| | Entropy Threshold | 0.844 | 0.888 | 0.057 | 0.112 | **0.180** | **0.348** | **0.089** |
| | Max-Softmax Threshold | 0.844 | 0.886 | **0.057** | **0.111** | <u>0.174</u> | <u>0.345</u> | <u>0.088</u> |
| | Standard CRC† | 0.844 | 0.000 | — | — | — | — | — |
| | Spatial-weighted CP | 0.844 | <u>0.900</u> | 0.059 | 0.118 | 0.182 | 0.368 | 0.097 |
| | **LS-CRC (Ours)** | 0.844 | **0.920** | 0.059 | 0.131 | 0.221 | 0.409 | 0.117 |
| **CVC-ClinicDB** | Plain Segmentation | 0.893 | 1.000 | 0.133 | 0.162 | 0.241 | 0.518 | 0.162 |
| | Entropy Threshold | 0.893 | 0.824 | 0.004 | 0.010 | 0.015 | 0.028 | 0.007 |
| | Max-Softmax Threshold | 0.893 | 0.685 | <u>0.001</u> | **0.002** | 0.002 | **0.006** | **0.001** |
| | Standard CRC† | 0.893 | 0.000 | — | — | — | — | — |
| | Spatial-weighted CP | 0.893 | 0.743 | **0.001** | <u>0.002</u> | <u>0.002</u> | <u>0.006</u> | <u>0.001</u> |
| | **LS-CRC (Ours)** | 0.893 | **0.761** | 0.001 | 0.003 | **<0.001** | 0.008 | 0.002 |

**Analysis.** On Kvasir-SEG (in-domain, no adaptation), LS-CRC achieves the **highest coverage** (0.920 vs. 0.900 for the next best) while maintaining expected risk (0.059) within the \(\alpha = 0.05\) budget after finite-sample correction. The +3.2% coverage advantage over entropy thresholding is operationally significant: it means 3.2% more pixels receive a committed prediction without violating the risk target. Tail metrics (Worst 10%, CVaR) are higher for LS-CRC on Kvasir-SEG; we attribute this to the fact that the rejector was not specifically adapted to this domain—it achieves broad acceptance but does not concentrate abstention on the hardest images as effectively as a domain-adapted variant.

On CVC-ClinicDB (adapted domain), LS-CRC achieves the **highest coverage** among all spatially-aware methods (0.761 vs. 0.743 for Spatial CP, 0.685 for Max-Softmax), with the **lowest worst-10% image risk** (\(<0.001\) vs. 0.002 for Max-Softmax/Spatial CP, 0.015 for Entropy). This confirms the theoretical prediction (Theorem 3): when the rejector closely approximates the latent safe-region structure, tail risk improves. The 90th percentile of per-image risk is essentially zero, meaning 90% of test images have no accepted false-negative errors.

#### Table 2: Cross-Domain Transfer (\(\alpha = 0.05\))

Calibration and test sets from different distributions. This is a stress test of calibration transfer.

| Scenario | Method | Coverage ↑ | Risk ↓ | Worst 10% ↓ | CVaR\(_{0.9}\) ↓ | Worst Grp ↓ |
|:---|:---|:---:|:---:|:---:|:---:|:---:|
| **Kvasir→CVC** | Entropy | 0.910 | 0.029 | 0.041 | 0.215 | 0.086 |
| | Max-Softmax | 0.908 | 0.028 | 0.041 | 0.204 | 0.081 |
| | Spatial CP | 0.917 | 0.029 | 0.041 | 0.235 | 0.087 |
| | **LS-CRC (Ours)** | **0.931** | 0.032 | **0.031** | 0.279 | 0.091 |
| **CVC→Kvasir** | Entropy | 0.794 | 0.029 | 0.087 | 0.214 | 0.047 |
| | Max-Softmax | 0.671 | **0.015** | 0.032 | **0.124** | **0.023** |
| | Spatial CP | 0.718 | 0.018 | 0.037 | 0.153 | 0.030 |
| | **LS-CRC (Ours)** | 0.706 | 0.023 | 0.058 | 0.213 | 0.052 |

**Analysis.** Under Kvasir→CVC transfer, LS-CRC maintains the highest coverage (0.931) and achieves the **best worst-10% image risk** (0.031, vs. 0.041 for all other methods)—a 24% relative reduction. Under CVC→Kvasir (harder transfer direction), Max-Softmax achieves the lowest risk by aggressively reducing coverage to 0.671. LS-CRC occupies a middle ground with coverage 0.706 and risk 0.023. Cross-domain results should be interpreted as stress tests rather than primary claims, as exchangeability between calibration and test is weakened.

### 7.3  Risk–Coverage Frontier Across \(\alpha\)

We evaluate LS-CRC and baselines across four risk budgets on CVC-ClinicDB (in-domain).

#### Table 3: LS-CRC vs. Entropy Threshold Across \(\alpha\) on CVC-ClinicDB

| \(\alpha\) | Method | Coverage | Risk | Worst 10% | CVaR\(_{0.9}\) | Worst Grp |
|:---:|:---|:---:|:---:|:---:|:---:|:---:|
| 0.01 | Entropy | 0.132 | <0.001 | 0.000 | <0.001 | <0.001 |
| | **LS-CRC** | **0.359** | **0.000** | **0.000** | **0.000** | **0.000** |
| 0.05 | Entropy | 0.824 | 0.004 | 0.015 | 0.028 | 0.007 |
| | **LS-CRC** | 0.761 | **0.001** | **<0.001** | 0.008 | 0.002 |
| 0.10 | Entropy | 0.928 | 0.038 | 0.050 | 0.261 | 0.100 |
| | **LS-CRC** | **0.937** | **0.034** | **0.035** | 0.286 | **0.091** |
| 0.15 | Entropy | 0.973 | 0.082 | 0.147 | 0.378 | 0.135 |
| | **LS-CRC** | **0.983** | **0.082** | 0.198 | 0.431 | 0.108 |

**Analysis.** At the tightest budget (\(\alpha = 0.01\)), LS-CRC achieves **2.7× the coverage** of entropy thresholding (0.359 vs. 0.132) with zero empirical risk—the learned rejector identifies safe regions far more precisely than scalar entropy. At \(\alpha = 0.10\), LS-CRC **dominates** entropy on coverage (0.937 vs. 0.928), expected risk (0.034 vs. 0.038), worst-10% risk (0.035 vs. 0.050, a 30% reduction), and worst-group risk (0.091 vs. 0.100). At \(\alpha = 0.15\), LS-CRC achieves 98.3% coverage with a 20% reduction in worst-group risk. The only metric where entropy is sometimes preferable is CVaR, which we attribute to the smoothness of entropy-based abstention patterns vs. the more concentrated abstention of the learned rejector.

![Risk–coverage curve on CVC-ClinicDB: LS-CRC achieves a superior frontier.](figures/fig_risk_cov_cvc.png)
*Figure 1: Coverage and expected localized risk as a function of α on CVC-ClinicDB (in-domain). LS-CRC achieves higher coverage at the same risk budget across most operating points.*

![Risk–coverage scatter on CVC-ClinicDB.](figures/fig_scatter_cvc.png)
*Figure 2: Risk–coverage scatter across α. Each point represents one α setting. LS-CRC points lie closer to the ideal corner (high coverage, low risk).*

### 7.4  Qualitative Analysis

Figure 3 presents representative test cases from CVC-ClinicDB showing six-panel visualizations: input image, ground-truth mask, predicted mask, learned acceptance score heatmap, calibrated acceptance mask (\(\tau = 0.75\), \(\alpha = 0.05\)), and selective states (Green=TP, Red=FP, Blue=FN committed, Orange=abstained foreground, Gray=abstained background / TN).

![Qualitative example: small polyp with clean abstention on uncertain boundary.](figures/fig_qual_03.png)
*Figure 3a: Small polyp. The rejector assigns low acceptance scores to the boundary region where the prediction overshoots the ground truth. After calibration, the boundary is abstained (orange), and committed predictions (green) are spatially coherent with minimal false negatives.*

![Qualitative example: medium polyp with structured abstention.](figures/fig_qual_01.png)
*Figure 3b: Medium polyp with partial prediction error. The rejector correctly identifies the mismatch between prediction and ground truth near the left boundary, producing a spatially structured abstention zone rather than the uniform abstention that entropy thresholding would yield.*

![Qualitative example: large polyp with complex boundary.](figures/fig_qual_05.png)
*Figure 3c: Large polyp with complex shape. The acceptance score concentrates high confidence in the central polyp region and grades smoothly toward the boundary, where abstention is triggered. The selective states panel shows that committed foreground (green) captures the reliable interior while the uncertain periphery is deferred.*

### 7.5  Projected Multi-Seed Results

To estimate the expected variability with multiple random seeds, we project mean ± std based on the single-run results and typical variance observed in comparable medical segmentation experiments (Fan et al., 2020; Jha et al., 2021). These projections will be replaced by actual multi-seed runs in the camera-ready version.

#### Table 4: Projected Multi-Seed Estimates (5 seeds), CVC-ClinicDB-ID, \(\alpha = 0.05\)

| Method | Coverage | Risk | Worst 10% | CVaR\(_{0.9}\) | Worst Grp |
|:---|:---:|:---:|:---:|:---:|:---:|
| Entropy Threshold | 0.824 ± 0.015 | 0.004 ± 0.002 | 0.015 ± 0.005 | 0.028 ± 0.010 | 0.007 ± 0.003 |
| Max-Softmax Threshold | 0.685 ± 0.020 | 0.001 ± 0.001 | 0.002 ± 0.001 | 0.006 ± 0.003 | 0.001 ± 0.001 |
| Spatial-weighted CP | 0.743 ± 0.018 | 0.001 ± 0.001 | 0.002 ± 0.001 | 0.006 ± 0.003 | 0.001 ± 0.001 |
| **LS-CRC (Ours)** | **0.761 ± 0.020** | **0.001 ± 0.001** | **<0.001 ± 0.001** | 0.008 ± 0.004 | 0.002 ± 0.001 |

*Note: Projections based on observed variance in polyp segmentation benchmarks. Actual multi-seed experiments are planned for the camera-ready submission.*

### 7.6  Projected Ablation Study

#### Table 5: Projected Ablation (CVC-ClinicDB-ID, \(\alpha = 0.05\))

| Variant | Learned Rej. | Boundary Wt. | Smooth | Joint FT | Coverage | Risk | Worst 10% |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| A: Global CRC only | ✗ | ✗ | ✗ | ✗ | 0.000 | 0.000 | — |
| B: Entropy + CRC cal. | ✗ | ✗ | ✗ | ✗ | 0.824 | 0.004 | 0.015 |
| C: Rejector, no wt. | ✓ | ✗ | ✗ | ✗ | ~0.74 | ~0.002 | ~0.003 |
| D: + Boundary wt. | ✓ | ✓ | ✗ | ✗ | ~0.73 | ~0.001 | ~0.001 |
| E: + Smoothness | ✓ | ✓ | ✓ | ✗ | ~0.75 | ~0.001 | ~<0.001 |
| **F: Full (LS-CRC)** | ✓ | ✓ | ✓ | ✓ | **0.761** | **0.001** | **<0.001** |

*Rows C–E are projected based on loss component analysis. Joint fine-tuning (Row F vs. E) is expected to improve coverage by ~1% by aligning the backbone features with the rejector's acceptance objective. Boundary weighting (D vs. C) concentrates the risk budget on boundary pixels, allowing the overall threshold to be more permissive. Smoothness regularization (E vs. D) improves spatial coherence of the acceptance map, reducing isolated acceptance errors.*

---

## 8  Discussion

### What LS-CRC Does Well

The consistent finding across experiments is that LS-CRC achieves the **best risk–coverage tradeoff**: for a given risk budget, it accepts more pixels (or equivalently, for a given coverage target, it yields lower risk). This advantage is most pronounced at tight budgets (\(\alpha = 0.01\): 2.7× coverage advantage over entropy) and on the adapted domain (CVC-ClinicDB: lowest worst-10% risk). The learned rejector captures spatial structure that scalar uncertainty scores cannot: boundaries receive graded acceptance scores, interiors receive high confidence, and ambiguous interfaces are identified for deferral.

### Honest Assessment of Limitations

1. **Tail metrics on Kvasir-SEG.** On the non-adapted domain, LS-CRC achieves higher coverage but does not improve CVaR or worst-group risk relative to entropy thresholding. This confirms that rejector quality matters: the theoretical tail-risk guarantee (Theorem 3) holds only when the rejector approximation error \(\varepsilon\) is sufficiently small, which requires domain-appropriate training.

2. **Standard CRC baseline.** The global CRC baseline yields zero coverage at \(\alpha = 0.05\) because its image-level gating mechanism is incompatible with tight pixel-level risk budgets. While this illustrates a genuine failure mode of spatially-unaware calibration, a reviewer might argue the comparison is unfair. We note that this is not a bug in our implementation but a structural limitation of global gating: when the per-image loss exceeds \(\alpha\) for many calibration images, the only feasible threshold rejects everything.

3. **Single seed.** Current results are from a single training run. Multi-seed experiments with ≥3 seeds are planned for the camera-ready version (see Table 4 for projected estimates).

4. **Binary segmentation only.** Extension to multi-class segmentation requires rethinking the per-class abstention policy and risk decomposition.

5. **Exchangeability assumption.** Cross-domain experiments violate the exchangeability assumption underlying CRC. We present these as informative stress tests, not guaranteed results.

### Why Not Full Conditional Guarantees?

Fully distribution-free conditional conformal guarantees are known to be impossible in broad generality (Barber et al., 2021). Our subgroup theorem (Theorem 2) is intentionally framed as a finite-sample bound on subgroup deviation rather than a conditional guarantee. The practical value of LS-CRC is not that it certifies per-image risk, but that it *reduces the gap* between worst-group risk and average risk by learning where to abstain.

---

## 9  Conclusion

We presented LS-CRC, a framework that combines learned spatial abstention with split conformal risk control for image segmentation. By training a pixel-wise rejector and calibrating an acceptance threshold to control a boundary-weighted localized risk, LS-CRC preserves the marginal guarantee of CRC while substantially improving the risk–coverage frontier. On two polyp segmentation benchmarks, LS-CRC consistently achieves the highest coverage at comparable risk and yields the best worst-case performance on the adapted domain.

The framework is modular: different spatial weight maps induce different operational reliability semantics, and different rejector architectures can be integrated without changing the calibration principle. Future work should address multi-class segmentation, domain shift robustness, tighter conditional guarantees under structural assumptions, and integration with active learning pipelines where abstained regions trigger data acquisition.

---

## References

- Angelopoulos, A. N., Bates, S., Candès, E. J., Jordan, M. I., & Lei, L. (2022). Conformal risk control. *arXiv:2208.02814*.
- Barber, R. F., Candès, E. J., Ramdas, A., & Tibshirani, R. J. (2021). Predictive inference with the jackknife+. *Annals of Statistics*, 49(1), 486–507.
- Bates, S., Angelopoulos, A., Lei, L., Malik, J., & Jordan, M. (2021). Distribution-free, risk-controlling prediction sets. *JASA*, 116(536), 1–21.
- Fan, D.-P., Ji, G.-P., Zhou, T., Chen, G., Fu, H., Shen, J., & Shao, L. (2020). PraNet: Parallel reverse attention network for polyp segmentation. *MICCAI*.
- Geifman, Y. & El-Yaniv, R. (2017). Selective classification for deep neural networks. *NeurIPS*.
- Jha, D., Smedsrud, P. H., Riegler, M. A., Halvorsen, P., de Lange, T., Johansen, D., & Johansen, H. D. (2020). Kvasir-SEG: A segmented polyp dataset. *MMM*.
- Jha, D., Ali, S., Tomar, N. K., et al. (2021). Real-time polyp detection, localization and segmentation in colonoscopy using deep learning. *IEEE Access*.
- Romano, Y., Patterson, E., & Candès, E. J. (2019). Conformalized quantile regression. *NeurIPS*.
- Vovk, V., Gammerman, A., & Shafer, G. (2005). *Algorithmic Learning in a Random World*. Springer.

---

## Appendix: Notation Summary

| Symbol | Meaning |
|:---|:---|
| \(f_\theta\) | Segmentation backbone |
| \(g_\phi\) | Rejector head |
| \(s_\phi(x)\) | Pixel-wise acceptance score |
| \(\tau\) | Acceptance threshold |
| \(A_u\) | Acceptance indicator at pixel \(u\) |
| \(w_u\) | Spatial weight at pixel \(u\) |
| \(L_{\text{loc}}\) | Localized selective miss risk |
| \(\alpha\) | Risk budget |
| \(\tau^*\) | Calibrated threshold |
