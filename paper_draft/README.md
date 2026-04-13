# Paper Draft — LS-CRC for NeurIPS 2026

## Structure

```
paper_draft/
├── main.md                 # Full paper (9 sections + appendix notation)
├── supplementary.md        # Proofs, full tables, ablation, checklist
├── README.md               # This file
└── figures/
    ├── fig_risk_cov_kvasir.png   → Risk–coverage curve, Kvasir-SEG
    ├── fig_risk_cov_cvc.png      → Risk–coverage curve, CVC-ClinicDB (Figure 1)
    ├── fig_scatter_kvasir.png    → Risk–coverage scatter, Kvasir-SEG
    ├── fig_scatter_cvc.png       → Risk–coverage scatter, CVC-ClinicDB (Figure 2)
    ├── fig_risk_cov_cross_k2c.png → Cross-domain Kvasir→CVC
    ├── fig_risk_cov_cross_c2k.png → Cross-domain CVC→Kvasir
    ├── fig_qual_01.png           → Qualitative: medium polyp (Figure 3b)
    ├── fig_qual_03.png           → Qualitative: small polyp  (Figure 3a)
    └── fig_qual_05.png           → Qualitative: large polyp  (Figure 3c)
```

All figures are symlinks to `../figures/paper/` and `../figures/qual_cvc_in/`.

## Data Source

All numeric results in the tables are extracted directly from:

```
figures/sweep/20260409_144715__ckpt-cvc_adapted__alpha{0p01,0p05,0p10,0p15}__grid1000__scen4.csv
```

## What to Do Next (Camera-Ready Prep)

1. **Multi-seed**: Run ≥3 seeds using `tools/run_paper_table_sweep.sh`, aggregate with `tools/aggregate_multiseed_csv.py`, update Tables 1–3 with mean ± std.
2. **Ablation**: Run variants (no smooth, no surrogate, no boundary weight) → replace projected Table 5.
3. **Standard CRC baseline**: Either fix score definition or add explicit discussion paragraph.
4. **Convert to LaTeX**: Use existing `docs/paper_assets/*.tex` as templates.
5. **Figures**: Regenerate at 300 DPI for print; add method overview diagram (Figure 0).
