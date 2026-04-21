# LS-CRC: Target paper (reference after re-running experiments)

**How to use.** This document is the **target specification**: after you run **≥3 random seeds** (5 recommended) and real ablations, table **means** should land near the **Target** columns or inside **Acceptable bands**. If you fall far below a band, revisit training/calibration; if you beat targets, update the main draft (but verify no calibration overfitting).

**Experimental setting (assumed when comparing):** boundary-weighted localized selective miss risk; threshold grid with 1000 points in [0,1]; finite-sample corrected α_n; DeepLabV3+ / ResNet-50; input 256×256. **CVC:** backbone trained on Kvasir, **rejector adapted on CVC**, calibrate and test on CVC. **Kvasir-ID:** train / val / cal / test on Kvasir splits.

**Band notation:** e.g. `0.76 ± 0.02` means acceptable mean range [0.74, 0.78] across seeds (lower std is better).

---

# Learning Structured Abstention for Localized Conformal Risk Control in Segmentation

**Anonymous Authors** · *Target manuscript for NeurIPS 2026 (AI/ML for health and biotechnology)*

## Abstract (target wording; place after Results are final)

Conformal Risk Control (CRC) controls expected bounded monotone losses, but standard instantiations are **marginal** and often **spatially blind**: they do not decide **where** to spend a risk budget in segmentation. We propose **Localized Selective CRC (LS-CRC)**, pairing a **learned pixel-wise rejector** with **split conformal** threshold selection on a **localized selective miss risk** (boundary-weighted false negatives on accepted foreground). We prove **marginal CRC inheritance** and give a **subgroup deviation decomposition**. On **Kvasir-SEG** and **CVC-ClinicDB**, we target the following empirical profile after multi-seed runs: **(i)** on **CVC (adapted)**, **higher accepted-pixel coverage** than spatially weighted conformal and entropy baselines at α=0.05 while achieving **lower or near-equal expected test risk** and **markedly lower worst-image tail risk**; **(ii)** on **Kvasir in-domain**, **best selective coverage** among compared methods with risk near the calibrated budget, acknowledging tail metrics may remain competitive with strong scalar baselines unless the rejector is Kvasir-adapted; **(iii)** under **Kvasir→CVC** transfer, **best coverage and best worst-10% image risk** among selective baselines at α=0.05. Overall, the target claim is **a better risk–coverage frontier** from **structured abstention**, not stronger distribution-free conditional guarantees.

---

## 1. Target empirical claims (checklist)

After **mean ± std** over at least 3 seeds, the following should hold (or weaken the prose one notch):

| ID | Claim | Pass criterion (mean over seeds) |
|----|--------|----------------------------------|
| C1 | Risk vs budget | At α=0.05 selective: LS-CRC test risk **≤ 0.08** on Kvasir-ID; **≤ 0.02** on CVC-ID (conservative slack is normal). Avoid systematic test risk **> α + 0.05** without explanation. |
| C2 | CVC-ID vs spatial CP | **Coverage** ≥ spatial CP mean **− 0.01** and **worst-10%** ≤ spatial CP mean **+ 0.005**; ideal: higher coverage **and** lower worst-10% (as in target table). |
| C3 | CVC-ID vs entropy | Coverage need not beat entropy; prioritize **worst-10%** and **worst-group**. If coverage is **≤ 0.07** below entropy, tail must win clearly (e.g. worst-10% **≥30% relative** better). |
| C4 | Kvasir-ID coverage | LS-CRC **highest** among selective methods **or** within **0.01** of the leader. |
| C5 | Kvasir→CVC | LS-CRC: **best coverage** and **lowest worst-10%** among selective baselines. |
| C6 | Tight α=0.01 on CVC | LS-CRC mean coverage **≥ 2.2×** entropy mean coverage (target ~2.7×). |
| C7 | Ablations | Steps C→F should not make worst-10% **> full + 0.01**; coverage should trend upward toward the full model. |

---

## 2. Target Table 1 — In-domain, α = 0.05

**Goal:** means near *Target*; tight std.

### Kvasir-SEG-ID

| Method | Dice (target) | Coverage (target) | Acceptable coverage band | Risk (target) | Worst 10% (note) |
|--------|---------------|-------------------|--------------------------|---------------|------------------|
| Plain | **0.84 ± 0.01** | 1.000 | — | 0.16 ± 0.02 | 0.37 ± 0.03 |
| Entropy | same backbone | **0.888** | 0.875–0.900 | **0.057** | **0.18** |
| Max-softmax | | **0.886** | 0.875–0.895 | **0.057** | **0.17** |
| Spatial-wt CP | | **0.900** | 0.888–0.910 | **0.059** | **0.18** |
| **LS-CRC** | | **0.920** | **0.908–0.932** | **0.059** | **0.20–0.24** (may exceed entropy if coverage leads) |

### CVC-ClinicDB-ID (rejector adapted on CVC)

| Method | Dice (target) | Coverage (target) | Acceptable band | Risk (target) | Worst 10% (target) |
|--------|---------------|-------------------|-----------------|---------------|---------------------|
| Plain | **0.89 ± 0.01** | 1.000 | — | 0.13 ± 0.02 | 0.24 ± 0.03 |
| Entropy | | **0.824** | 0.805–0.838 | **0.004** | **0.015** |
| Max-softmax | | **0.685** | 0.65–0.72 | **0.001** | **0.002** |
| Spatial-wt CP | | **0.743** | 0.725–0.760 | **0.001** | **0.002** |
| **LS-CRC** | | **0.761** | **0.745–0.775** | **≤ 0.002** | **< 0.002** (ideal mean ~**0.00006**) |

**Note:** If multi-seed worst-10% for LS-CRC is **0.003–0.008** but still **below entropy (~0.012+)** and coverage **> spatial CP**, the paper narrative remains strong.

---

## 3. Target Table 2 — Cross-domain, α = 0.05

### Kvasir cal → CVC test

| Method | Coverage (target) | Band | Risk (target) | Worst 10% (target) |
|--------|-----------------|------|---------------|---------------------|
| Entropy | **0.910** | 0.895–0.925 | **0.029** | **0.041** |
| Max-softmax | **0.908** | 0.890–0.922 | **0.028** | **0.041** |
| Spatial-wt CP | **0.917** | 0.900–0.928 | **0.029** | **0.041** |
| **LS-CRC** | **0.931** | **0.918–0.940** | **≤ 0.035** | **≤ 0.034** (ideal **0.031**) |

### CVC cal → Kvasir test

| Method | Coverage (target) | Risk (target) | Worst 10% (target) |
|--------|-------------------|---------------|---------------------|
| Entropy | **0.794** | **0.029** | **0.087** |
| Max-softmax | **0.671** | **0.015** | **0.032** |
| Spatial-wt CP | **0.718** | **0.018** | **0.037** |
| **LS-CRC** | **0.706** ± 0.03 | **0.023** ± 0.01 | **0.058** ± 0.02 |

*(Hard direction; if LS-CRC risk slightly exceeds max-softmax but coverage is **+0.03** higher, keep a trade-off narrative.)*

---

## 4. Target Table 3 — CVC-ID, LS-CRC vs entropy across α

| α | Metric | Entropy (target mean) | LS-CRC (target mean) | Pass if |
|:---:|:---|:---:|:---:|:---|
| 0.01 | Coverage | **0.132** | **0.359** | LS-CRC ≥ **0.30** and ≥ **2.2×** entropy |
| 0.01 | Risk | <0.001 | **0.000** | both < **0.005** |
| 0.05 | Coverage | **0.824** | **0.761** | LS-CRC worst-10% **< 0.003**; entropy ~**0.015** |
| 0.05 | Risk | **0.004** | **0.001** | LS-CRC ≤ entropy |
| 0.10 | Coverage | **0.928** | **0.937** | LS-CRC ≥ entropy; risk LS-CRC **≤** entropy |
| 0.10 | Worst 10% | **0.050** | **0.035** | LS-CRC **lower** than entropy |
| 0.15 | Coverage | **0.973** | **0.983** | LS-CRC ≥ entropy |
| 0.15 | Worst grp | **0.135** | **0.108** | LS-CRC **lower** than entropy (same mean risk ~**0.082**) |

**CVaR:** Entropy may win at large α; do **not** make it the headline claim.

---

## 5. Target Table 4 — Ablations (CVC-ID, α=0.05, measured, mean ± std)

Replace all approximate (~) rows with real numbers. Qualitative targets:

| Variant | Coverage (target mean) | Risk (target) | Worst 10% (target) | Ordering |
|:---|:---:|:---:|:---:|:---|
| A Image-level CRC | **0.000** | — | — | non-operative baseline |
| B Entropy + CRC | **0.824** | **0.004** | **0.015** | scalar + cal |
| C Rejector, no boundary wt. | **0.735–0.750** | **0.001–0.003** | **0.002–0.004** | below full |
| D + Boundary wt. | **0.728–0.742** | **≤ 0.002** | **≤ 0.002** | boundary-focused risk |
| E + Smooth, no joint FT | **0.745–0.758** | **≤ 0.002** | **< 0.002** | near full |
| **F Full LS-CRC** | **0.761** ± 0.02 | **0.001** ± 0.001 | **< 0.002** | **best** in ablation |

---

## 6. Target figures (same layout as main draft)

| Fig | File (under repo root) | Target appearance |
|-----|------------------------|-------------------|
| 1 | `figures/paper/lscrc_cvc_id_risk_coverage_vs_alpha.png` | LS-CRC curve **favorably shifted** vs entropy across most α. |
| 2 | `figures/paper/lscrc_cvc_id_risk_coverage_vs_alpha_scatter.png` | LS-CRC points closer to the **high-coverage, low-risk** corner. |
| 3 | `figures/paper/lscrc_kvasir_id_risk_coverage_vs_alpha.png` | High LS-CRC **coverage**; tail may lag — consistent with Table 1. |
| 4 | `figures/paper/lscrc_cross_kvasir_cal_cvc_test_risk_coverage_vs_alpha.png` | LS-CRC **strong** on stress-test α values. |
| 5–7 | `figures/qual_cvc_in/lscrc_cvc_id_states_v2_*.png` | High scores in **interior**, low near **boundary**; **coherent** accept mask. |

---

## 7. Reporting bar (NeurIPS health track)

- **Seeds:** report **≥3** (target **5**); list seeds in supplement.
- **Main tables:** mean ± std for **coverage, risk, worst-10%, worst-group** (minimum: CVC-ID + Kvasir→CVC).
- **Supplement:** (i) cal vs test risk vs α; (ii) per-seed table if needed; (iii) wall-clock and GPU type.

---

## 8. If you miss targets — actions

| Symptom | Action |
|---------|--------|
| Test risk often **exceeds** α | Check α_n, threshold grid, calibration leakage; increase n_cal or use more conservative slack. |
| CVC LS-CRC coverage **below** spatial CP | Increase localized-surrogate weight / joint FT / rejector epochs; audit boundary pseudo-labels. |
| Kvasir tail **too poor** | Add Kvasir rejector adaptation stage or rebalance rejector loss weights. |
| Cross-domain collapses | Keep as stress test only; add stronger baseline if needed; do not over-claim guarantees. |

---

*This file is a **spec** alongside `neurips_ls_crc_complete_draft.md`: when numbers stabilize, sync §Results in the main draft and update the abstract to match **mean ± std**.*

# Ghi chú tiếng Việt (tóm t��t)

- File này là **bản paper mục tiêu**: abstract + bảng số + checklist để sau khi chạy lại multi-seed bạn **so khớp nhanh**.
- Số **Target** lấy theo run `cvc_adapted` / bảng chính hiện tại; **Acceptable band** cho phép dao đ� qua seed.
- Khi mean ± std đã có, hãy **copy nội dung số** vào `neurips_ls_crc_complete_draft.md` và xóa dòng “single seed” �� mục Results.
