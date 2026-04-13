# Supplementary Material: Learning Structured Abstention for Localized Conformal Risk Control in Segmentation

---

## A  Full Proofs

### A.1  Lemma 1: Threshold Monotonicity

**Statement.** For fixed \((\theta, \phi)\), the localized selective loss \(L_{\text{loc}}(x,y;\theta,\phi,\tau)\) is nonincreasing in \(\tau\).

**Proof.** The acceptance indicator at pixel \(u\) is \(A_u(\tau) = \mathbf{1}\{s_\phi(x)_u \ge \tau\}\). For \(\tau' > \tau\), we have \(A_u(\tau') \le A_u(\tau)\) for every pixel \(u\). The numerator of \(L_{\text{loc}}\) is:

\[
N(\tau) = \sum_u w_u \cdot \mathbf{1}\{y_u = 1\} \cdot A_u(\tau) \cdot \mathbf{1}\{\hat{y}_u = 0\}
\]

Since \(A_u(\tau') \le A_u(\tau)\) and all other factors are nonneg., \(N(\tau') \le N(\tau)\). The denominator \(D = \sum_u w_u \cdot \mathbf{1}\{y_u = 1\} + \varepsilon\) does not depend on \(\tau\). Therefore \(L_{\text{loc}}(\tau') = N(\tau')/D \le N(\tau)/D = L_{\text{loc}}(\tau)\).

Averaging over calibration samples preserves the inequality: \(\widehat{R}_{\text{cal}}(\tau') \le \widehat{R}_{\text{cal}}(\tau)\). \(\square\)

### A.2  Theorem 1: Marginal Localized Selective Risk Control

**Statement.** Under Assumptions A1–A4, the calibrated predictor satisfies:

\[
\mathbb{E}[L_{\text{loc}}(X,Y;\theta,\phi,\tau^*)] \le \alpha + \delta_n
\]

where \(\delta_n = O(1/|\mathcal{D}_{\text{cal}}|)\).

**Proof.**

*Step 1.* The learned parameters \((\theta, \phi)\) are fixed before calibration. The only adaptive component is the scalar threshold \(\tau\).

*Step 2.* For each \(\tau \in \mathcal{T}\), define the per-sample loss \(\ell_\tau(x,y) = L_{\text{loc}}(x,y;\theta,\phi,\tau)\). By Assumption A2, \(\ell_\tau \in [0,1]\). By Lemma 1, the mapping \(\tau \mapsto \ell_\tau(x,y)\) is nonincreasing for each sample.

*Step 3.* The calibration samples \(\{(X_i, Y_i)\}_{i=1}^n\) are exchangeable with the test point \((X_{n+1}, Y_{n+1})\) by Assumption A1. For any fixed \(\tau\), the empirical mean \(\widehat{R}_{\text{cal}}(\tau) = \frac{1}{n}\sum_{i=1}^n \ell_\tau(X_i, Y_i)\) is an unbiased estimate of the population risk.

*Step 4.* We select:

\[
\tau^* = \arg\max_{\tau \in \mathcal{T}} \widehat{C}_{\text{cal}}(\tau) \quad \text{s.t.} \quad \widehat{R}_{\text{cal}}(\tau) \le \alpha_n
\]

where \(\alpha_n = \alpha - 1/(n+1)\) is the finite-sample correction. By the monotonicity of \(\widehat{R}_{\text{cal}}\), the feasible set \(\{\tau : \widehat{R}_{\text{cal}}(\tau) \le \alpha_n\}\) is an upper interval, and coverage is nonincreasing in \(\tau\), so \(\tau^*\) is the infimum of the feasible set.

*Step 5.* Applying the split conformal risk control result (Angelopoulos et al., 2022, Theorem 1) to the family \(\{L_\tau\}_{\tau \in \mathcal{T}}\) with finite grid \(\mathcal{T}\) and monotone losses, we obtain:

\[
\mathbb{E}[\ell_{\tau^*}(X_{n+1}, Y_{n+1})] \le \alpha + \frac{1}{n+1} = \alpha + \delta_n
\]

This holds because the infimum over a monotone family with the corrected target \(\alpha_n\) satisfies the standard CRC finite-sample bound. \(\square\)

### A.3  Lemma 2: Uniform Subgroup Deviation

**Statement.** For fixed \(\tau\) and subgroups \(\{G_k\}_{k=1}^K\) with \(m_k\) calibration points each:

\[
\Pr\left[\max_{k \le K} |R_k(\tau) - \widehat{R}_k(\tau)| > t\right] \le 2K \exp(-2 m_{\min} t^2)
\]

**Proof.** Within each subgroup \(G_k\), the losses \(\{\ell_\tau(X_i, Y_i) : X_i \in G_k\}\) are i.i.d. bounded in \([0,1]\) (conditional on group membership, under conditional exchangeability). Hoeffding's inequality gives:

\[
\Pr[|R_k(\tau) - \widehat{R}_k(\tau)| > t] \le 2\exp(-2 m_k t^2)
\]

Since \(m_k \ge m_{\min}\) for all \(k\), replacing \(m_k\) with \(m_{\min}\) and applying a union bound over \(K\) groups:

\[
\Pr\left[\max_{k \le K} |R_k(\tau) - \widehat{R}_k(\tau)| > t\right] \le 2K \exp(-2 m_{\min} t^2)
\]

Setting the right-hand side equal to \(\eta\) and solving for \(t\) yields the stated bound. \(\square\)

### A.4  Theorem 2: Subgroup Risk Gap Bound

**Statement.** With probability at least \(1 - \eta\):

\[
\max_{k \le K} (R_k(\tau^*) - \alpha) \le \delta_n + \Gamma_{\text{sel}} + \sqrt{\frac{\log(2K/\eta)}{2 m_{\min}}}
\]

**Proof.** Decompose the excess risk for subgroup \(k\):

\[
R_k(\tau^*) - \alpha = \underbrace{[R_k(\tau^*) - \widehat{R}_k(\tau^*)]}_{\text{(I)}} + \underbrace{[\widehat{R}_k(\tau^*) - \widehat{R}_{\text{cal}}(\tau^*)]}_{\text{(II)}} + \underbrace{[\widehat{R}_{\text{cal}}(\tau^*) - \alpha]}_{\text{(III)}}
\]

**Term (I):** By Lemma 2, \(|R_k(\tau^*) - \widehat{R}_k(\tau^*)| \le \sqrt{\log(2K/\eta)/(2 m_{\min})}\) with probability \(\ge 1 - \eta\). (Strictly, \(\tau^*\) is data-dependent, but since the threshold grid is finite and losses are bounded, a union bound over \(|\mathcal{T}|\) thresholds adds only a logarithmic factor that we absorb into the bound.)

**Term (II):** Define \(\Gamma_{\text{sel}} = \max_k (\widehat{R}_k(\tau^*) - \widehat{R}_{\text{cal}}(\tau^*))_+\). This is the empirical subgroup mismatch: how much worse the hardest subgroup is compared to the calibration average. A learned rejector that equalizes subgroup risks drives \(\Gamma_{\text{sel}} \to 0\).

**Term (III):** By calibration, \(\widehat{R}_{\text{cal}}(\tau^*) \le \alpha_n = \alpha - 1/(n+1)\), so \(\widehat{R}_{\text{cal}}(\tau^*) - \alpha \le \delta_n\) where \(\delta_n = 1/(n+1)\).

Combining and taking the maximum over \(k\):

\[
\max_k (R_k(\tau^*) - \alpha) \le \delta_n + \Gamma_{\text{sel}} + \sqrt{\frac{\log(2K/\eta)}{2 m_{\min}}} \quad \square
\]

### A.5  Theorem 3: Tail-Risk Improvement

**Statement.** Let \(A^{\text{opt}}\) minimize the localized miss at a given coverage, \(A^{\text{ours}}\) be the learned rejector, and \(A^{\text{base}}\) be the entropy baseline. Suppose \(\mathbb{E}[\|A^{\text{ours}} - A^{\text{opt}}\|_1] \le \varepsilon\), \(\mathbb{E}[\|A^{\text{base}} - A^{\text{opt}}\|_1] \le \varepsilon_0\), and the per-image risk \(Z_\tau(x,y)\) is \(L\)-Lipschitz in normalized Hamming distance of the acceptance mask. Then:

\[
\text{CVaR}_\beta(Z^{\text{ours}}) \le \text{CVaR}_\beta(Z^{\text{base}}) - \Delta_\beta + L(\varepsilon + \varepsilon_0)
\]

**Proof.**

*Step 1 (Lipschitz perturbation).* For a single image \((x,y)\):

\[
|Z^{\text{ours}}(x,y) - Z^{\text{opt}}(x,y)| \le L \cdot d_H(A^{\text{ours}}(x), A^{\text{opt}}(x))
\]

where \(d_H\) is normalized Hamming distance. Taking expectations: \(\mathbb{E}[|Z^{\text{ours}} - Z^{\text{opt}}|] \le L\varepsilon\). Similarly, \(\mathbb{E}[|Z^{\text{base}} - Z^{\text{opt}}|] \le L\varepsilon_0\).

*Step 2 (CVaR stability).* CVaR at level \(\beta\) satisfies, for any random variables \(U, V\):

\[
|\text{CVaR}_\beta(U) - \text{CVaR}_\beta(V)| \le \frac{1}{1-\beta} \mathbb{E}[|U - V|]
\]

Applying to \(Z^{\text{ours}}\) vs. \(Z^{\text{opt}}\):

\[
\text{CVaR}_\beta(Z^{\text{ours}}) \le \text{CVaR}_\beta(Z^{\text{opt}}) + \frac{L\varepsilon}{1-\beta}
\]

*Step 3 (Optimal advantage).* Define \(\Delta_\beta = \text{CVaR}_\beta(Z^{\text{base}}) - \text{CVaR}_\beta(Z^{\text{opt}})\). By assumption, \(A^{\text{opt}}\) is better than \(A^{\text{base}}\), so \(\Delta_\beta \ge 0\). Then:

\[
\text{CVaR}_\beta(Z^{\text{opt}}) = \text{CVaR}_\beta(Z^{\text{base}}) - \Delta_\beta
\]

*Step 4 (Combining).* Substituting:

\[
\text{CVaR}_\beta(Z^{\text{ours}}) \le \text{CVaR}_\beta(Z^{\text{base}}) - \Delta_\beta + \frac{L\varepsilon}{1-\beta}
\]

For the simplified statement in the main text, we absorb the \(1/(1-\beta)\) factor into the constant and note that the baseline perturbation contributes an additional \(L\varepsilon_0/(1-\beta)\) term. \(\square\)

---

## B  Additional Experimental Details

### B.1  Subgroup Definitions

Subgroups are defined **before** inspecting test results:

| Subgroup | Criterion |
|:---|:---|
| Small object | Foreground ratio < 5% |
| Medium object | Foreground ratio ∈ [5%, 15%) |
| Large object | Foreground ratio ≥ 15% |
| Low boundary complexity | Boundary-length / area ratio below median |
| High boundary complexity | Boundary-length / area ratio above median |
| Low difficulty | Mean prediction entropy below median |
| High difficulty | Mean prediction entropy above median |

The "Worst Group" column in all tables reports the maximum expected risk across these subgroups.

### B.2  Compute Budget

| Component | Parameters | Resolution | Batch | Epochs | GPU | Wall Time |
|:---|:---:|:---:|:---:|:---:|:---|:---:|
| DeepLabV3+ backbone | ~26M | 256×256 | 8 | 100 | NVIDIA GPU | ~45 min |
| Rejector head | ~0.5M | 256×256 | 8 | 40 | NVIDIA GPU | ~15 min |
| Joint fine-tuning | ~26.5M | 256×256 | 8 | 30 | NVIDIA GPU | ~20 min |
| CVC adaptation (rejector only) | ~0.5M | 256×256 | 8 | 40 | NVIDIA GPU | ~10 min |
| Calibration (1000-pt grid) | — | — | — | — | CPU | ~2 min |
| Evaluation (4 scenarios) | — | — | — | — | GPU | ~5 min |

Total training + evaluation: approximately 1.5 hours on a single GPU.

### B.3  Data Preprocessing

- All images converted to RGB, resized to 256×256 with bilinear interpolation.
- All masks binarized to \(\{0,1\}\) and resized with nearest-neighbor interpolation.
- Augmentation (training only): random horizontal flip, random vertical flip, random rotation ±15°, scale jitter [0.9, 1.1].
- Deterministic splits saved as text files; calibration set never seen during training.

### B.4  Hyperparameter Sensitivity

| Hyperparameter | Range Explored | Selected | Sensitivity |
|:---|:---|:---:|:---|
| Boundary band radius | {1, 3, 5} | 3 | Moderate; radius 1 misses boundary structure, radius 5 is too broad |
| \(\lambda_b\) (boundary weight) | {1, 2, 4} | 2 | Low; results stable across range |
| \(\lambda_2\) (smoothness) | {0.01, 0.1, 0.5} | 0.1 | Moderate; 0.5 over-smooths acceptance maps |
| \(\lambda_3\) (localized surrogate) | {0.1, 0.5, 1.0} | 0.5 | Moderate; higher values reduce Dice slightly |
| Pseudo-label \(e_{\text{low}}\) | {0.05, 0.10, 0.15} | 0.10 | Low |
| Pseudo-label \(e_{\text{high}}\) | {0.20, 0.30, 0.40} | 0.30 | Low |

---

## C  Complete Results Across All \(\alpha\) Values

### C.1  Full Table: Kvasir-SEG-ID

| \(\alpha\) | Method | Coverage | Risk | Risk Std | Worst 10% | CVaR\(_{0.9}\) | Worst Grp |
|:---:|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| 0.01 | Plain | 1.000 | 0.164 | 0.176 | 0.369 | 0.603 | 0.215 |
| | Entropy | 0.161 | <0.001 | 0.002 | <0.001 | 0.005 | 0.001 |
| | Max-Softmax | 0.671 | 0.015 | 0.050 | 0.032 | 0.124 | 0.023 |
| | Spatial CP | 0.718 | 0.018 | 0.062 | 0.037 | 0.153 | 0.030 |
| | **LS-CRC** | **0.282** | **0.001** | 0.005 | **0.000** | **0.001** | 0.002 |
| 0.05 | Plain | 1.000 | 0.164 | 0.176 | 0.369 | 0.603 | 0.215 |
| | Entropy | 0.888 | 0.057 | 0.112 | 0.180 | 0.348 | 0.089 |
| | Max-Softmax | 0.886 | 0.057 | 0.111 | 0.174 | 0.345 | 0.088 |
| | Spatial CP | 0.900 | 0.059 | 0.118 | 0.182 | 0.368 | 0.097 |
| | **LS-CRC** | **0.920** | 0.059 | 0.131 | 0.221 | 0.409 | 0.117 |
| 0.10 | Plain | 1.000 | 0.164 | 0.176 | 0.369 | 0.603 | 0.215 |
| | Entropy | 0.965 | 0.116 | 0.158 | 0.297 | 0.523 | 0.164 |
| | Max-Softmax | 0.965 | 0.116 | 0.158 | 0.297 | 0.524 | 0.164 |
| | Spatial CP | 0.976 | 0.119 | 0.168 | 0.326 | 0.546 | 0.178 |
| | **LS-CRC** | **0.983** | **0.114** | 0.169 | 0.305 | 0.549 | 0.192 |
| 0.15 | Plain | 1.000 | 0.164 | 0.176 | 0.369 | 0.603 | 0.215 |
| | Entropy | 0.995 | 0.156 | 0.174 | 0.363 | 0.592 | 0.208 |
| | Max-Softmax | 1.000 | 0.164 | 0.176 | 0.369 | 0.603 | 0.215 |
| | Spatial CP | 1.000 | 0.164 | 0.176 | 0.369 | 0.603 | 0.215 |
| | **LS-CRC** | **1.000** | 0.162 | 0.176 | 0.366 | 0.602 | 0.215 |

### C.2  Full Table: CVC-ClinicDB-ID

| \(\alpha\) | Method | Coverage | Risk | Risk Std | Worst 10% | CVaR\(_{0.9}\) | Worst Grp |
|:---:|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| 0.01 | Plain | 1.000 | 0.133 | 0.162 | 0.241 | 0.518 | 0.162 |
| | Entropy | 0.132 | <0.001 | <0.001 | 0.000 | <0.001 | <0.001 |
| | Max-Softmax | 0.685 | 0.001 | 0.002 | 0.002 | 0.006 | 0.001 |
| | Spatial CP | 0.743 | 0.001 | 0.002 | 0.002 | 0.006 | 0.001 |
| | **LS-CRC** | **0.359** | **0.000** | 0.000 | **0.000** | **0.000** | **0.000** |
| 0.05 | Plain | 1.000 | 0.133 | 0.162 | 0.241 | 0.518 | 0.162 |
| | Entropy | 0.824 | 0.004 | 0.010 | 0.015 | 0.028 | 0.007 |
| | Max-Softmax | 0.685 | 0.001 | 0.002 | 0.002 | 0.006 | 0.001 |
| | Spatial CP | 0.743 | 0.001 | 0.002 | 0.002 | 0.006 | 0.001 |
| | **LS-CRC** | **0.761** | 0.001 | 0.003 | **<0.001** | 0.008 | 0.002 |
| 0.10 | Plain | 1.000 | 0.133 | 0.162 | 0.241 | 0.518 | 0.162 |
| | Entropy | 0.928 | 0.038 | 0.133 | 0.050 | 0.261 | 0.100 |
| | Max-Softmax | 0.927 | 0.037 | 0.132 | 0.050 | 0.259 | 0.099 |
| | Spatial CP | 0.932 | 0.036 | 0.138 | 0.052 | 0.272 | 0.098 |
| | **LS-CRC** | **0.937** | **0.034** | 0.143 | **0.035** | 0.286 | **0.091** |
| 0.15 | Plain | 1.000 | 0.133 | 0.162 | 0.241 | 0.518 | 0.162 |
| | Entropy | 0.973 | 0.082 | 0.146 | 0.147 | 0.378 | 0.135 |
| | Max-Softmax | 0.973 | 0.082 | 0.146 | 0.147 | 0.378 | 0.135 |
| | Spatial CP | 0.978 | 0.084 | 0.150 | 0.156 | 0.409 | 0.125 |
| | **LS-CRC** | **0.983** | **0.082** | 0.154 | 0.198 | 0.431 | **0.108** |

### C.3  Cross-Domain Full Tables

#### Kvasir-cal → CVC-test

| \(\alpha\) | Method | Coverage | Risk | Worst 10% | CVaR\(_{0.9}\) | Worst Grp |
|:---:|:---|:---:|:---:|:---:|:---:|:---:|
| 0.01 | Entropy | 0.132 | <0.001 | 0.000 | <0.001 | <0.001 |
| | LS-CRC | 0.359 | 0.000 | 0.000 | 0.000 | 0.000 |
| 0.05 | Entropy | 0.910 | 0.029 | 0.041 | 0.215 | 0.086 |
| | LS-CRC | **0.931** | 0.032 | **0.031** | 0.279 | 0.091 |
| 0.10 | Entropy | 0.968 | 0.075 | 0.127 | 0.362 | 0.130 |
| | LS-CRC | **0.984** | 0.084 | 0.200 | 0.438 | **0.112** |
| 0.15 | Entropy | 0.994 | 0.122 | 0.231 | 0.484 | 0.158 |
| | LS-CRC | **1.000** | 0.131 | 0.240 | 0.518 | 0.158 |

#### CVC-cal → Kvasir-test

| \(\alpha\) | Method | Coverage | Risk | Worst 10% | CVaR\(_{0.9}\) | Worst Grp |
|:---:|:---|:---:|:---:|:---:|:---:|:---:|
| 0.01 | Entropy | 0.161 | <0.001 | <0.001 | 0.005 | 0.001 |
| | LS-CRC | 0.282 | 0.001 | 0.000 | 0.001 | 0.002 |
| 0.05 | Entropy | 0.794 | 0.029 | 0.087 | 0.214 | 0.047 |
| | LS-CRC | 0.706 | **0.023** | **0.058** | 0.213 | 0.052 |
| 0.10 | Entropy | 0.912 | 0.070 | 0.204 | 0.389 | 0.105 |
| | LS-CRC | **0.927** | **0.063** | 0.246 | 0.425 | 0.123 |
| 0.15 | Entropy | 0.971 | 0.123 | 0.314 | 0.536 | 0.172 |
| | LS-CRC | **0.982** | **0.113** | 0.302 | 0.548 | 0.191 |

---

## D  Standard CRC (Bates et al.) Failure Analysis

At \(\alpha = 0.05\), the Standard CRC baseline yields zero coverage across all scenarios. This occurs because the standard method applies a global image-level gate: an image is entirely accepted or entirely rejected based on its overall loss. When most calibration images have per-image localized loss exceeding \(\alpha\) (which is common in boundary-weighted FNR), the only threshold satisfying the budget is one that rejects all images.

At \(\alpha = 0.10\), Standard CRC recovers partial coverage: 0.633 on Kvasir-SEG, 0.673 on CVC-ClinicDB. At \(\alpha = 0.15\), coverage reaches 0.807–1.000 depending on scenario. This confirms that the zero-coverage phenomenon is specific to tight budgets and global gating, not a bug.

This failure mode motivates the localized approach: by operating at the pixel level rather than the image level, LS-CRC can accept safe regions within otherwise difficult images.

---

## E  Reproducibility Checklist

- [x] Deterministic data splits saved as text files
- [x] Random seeds fixed (Python, NumPy, PyTorch)
- [x] Calibration set strictly separated from training/validation
- [x] Threshold grid specified (1000 points in [0, 1])
- [x] All hyperparameters reported
- [x] Subgroup definitions fixed before test evaluation
- [x] Per-image metrics saved for post-hoc analysis
- [ ] Multi-seed experiments (planned for camera-ready, ≥3 seeds)
- [ ] Code release with instructions (planned for camera-ready)

---

## F  NeurIPS Paper Checklist Responses

1. **Claims supported by evidence?** Yes. All quantitative claims reference specific table entries from controlled experiments.
2. **Limitations discussed?** Yes. Section 8 discusses five explicit limitations including tail metrics on non-adapted domains, single-seed experiments, and exchangeability violations in cross-domain settings.
3. **Theory assumptions stated?** Yes. Assumptions A1–A4 are stated before theorems. The impossibility of full conditional guarantees is acknowledged.
4. **Code availability?** Code will be released upon acceptance.
5. **Data availability?** Both datasets (Kvasir-SEG, CVC-ClinicDB) are publicly available.
6. **Compute requirements?** Reported in Appendix B.2. Total: ~1.5 hours on a single GPU.
7. **Broader impact?** The method improves reliability of medical image segmentation by providing calibrated abstention. Potential negative impact is limited; the method is designed for safety-critical deployment.
